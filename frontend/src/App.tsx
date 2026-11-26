import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiRequestError, api, draftKey, keepDraft } from './api'
import BrollPanel from './components/BrollPanel'
import PreviewPlayer from './components/PreviewPlayer'
import ProcessingPanel from './components/ProcessingPanel'
import RenderPanel from './components/RenderPanel'
import SocialPanel from './components/SocialPanel'
import TranscriptEditor from './components/TranscriptEditor'
import UploadPanel from './components/UploadPanel'
import type {
  BrollStatus,
  Capabilities,
  Job,
  Project,
  RenderOptions,
  Timeline,
} from './types'

interface Notice {
  kind: 'error' | 'warn' | 'info'
  message: string
  detail?: string
}

/** One point in the undo history: the *whole* edit state, not just the words. */
interface EditSnapshot {
  removed: number[]
  trimSilences: boolean
}

const INGEST_POLL_MS = 1200
const RENDER_POLL_MS = 900
const EDIT_DEBOUNCE_MS = 450
/** Network blips are retried; this is how long we wait before the first retry. */
const EDIT_RETRY_BASE_MS = 1500
const EDIT_MAX_RETRIES = 5
/** Consecutive poll failures tolerated before the render job is declared lost. */
const POLL_FAILURE_LIMIT = 5

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [project, setProject] = useState<Project | null>(null)
  const [ingestJob, setIngestJob] = useState<Job | null>(null)
  const [renderJob, setRenderJob] = useState<Job | null>(null)
  const [timeline, setTimeline] = useState<Timeline | null>(null)
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const [notice, setNotice] = useState<Notice | null>(null)
  const [activeWordId, setActiveWordId] = useState<number | null>(null)
  const [seekTo, setSeekTo] = useState<{ time: number; nonce: number } | null>(null)
  const [brollBusyId, setBrollBusyId] = useState<string | null>(null)
  const [savingEdit, setSavingEdit] = useState(false)
  const [unsavedEdit, setUnsavedEdit] = useState(false)
  const [trimSilences, setTrimSilences] = useState(true)
  const [isProofreading, setIsProofreading] = useState(false)
  const [recoveryDraft, setRecoveryDraft] = useState<EditSnapshot | null>(null)
  const [currentView, setCurrentView] = useState<'edit' | 'share'>('edit')

  const lastClickedRef = useRef<number | null>(null)
  const currentProjectIdRef = useRef<string | null>(null)
  const navigationRef = useRef(0)
  const editBusyRef = useRef(false)
  editBusyRef.current = savingEdit || unsavedEdit
  const skipNextSaveRef = useRef(true)

  // Edit persistence bookkeeping. `editSeqRef` orders the PUTs so a slow older
  // response can never overwrite a newer one; `pendingEditRef` holds whatever
  // the server has not acknowledged yet so it can be flushed or retried.
  const editSeqRef = useRef(0)
  const inFlightEditsRef = useRef(0)
  const pendingEditRef = useRef<EditSnapshot | null>(null)
  const retryHandleRef = useRef(0)
  const retryCountRef = useRef(0)

  const fail = useCallback((message: string, detail?: string) => {
    setNotice({ kind: 'error', message, detail })
  }, [])

  const warn = useCallback((message: string, detail?: string) => {
    setNotice((current) => current ?? { kind: 'warn', message, detail })
  }, [])

  const handleApiError = useCallback(
    (error: unknown, fallback: string) => {
      const err = error as { message?: string; detail?: string }
      fail(err?.message ?? fallback, err?.detail)
    },
    [fail],
  )

  // ------------------------------------------------------------- bootstrap
  const loadProjects = useCallback(() => {
    api
      .listProjects()
      .then(setProjects)
      .catch((error: unknown) => {
        // Swallowing this showed an empty "no projects yet" screen, which reads
        // as "your projects were deleted".
        warn(
          'Could not load your projects.',
          (error as { message?: string })?.message ??
            'The server did not answer /api/projects. Your projects are still on disk.',
        )
      })
  }, [warn])

  useEffect(() => {
    api
      .capabilities()
      .then(setCapabilities)
      .catch((error: unknown) => {
        // capabilities === null means "unknown", never "all clear": the
        // environment banner is the only place a missing API key surfaces
        // before an upload fails minutes later.
        warn(
          'Could not read the server environment.',
          (error as { message?: string })?.message ??
            'API keys, the FFmpeg build and the subtitle font could not be checked.',
        )
      })
    loadProjects()
  }, [loadProjects, warn])

  const openProject = useCallback(
    async (projectId: string) => {
      setRecoveryDraft(null)
      const generation = ++navigationRef.current
      ++editSeqRef.current
      window.clearTimeout(retryHandleRef.current)
      const oldId = currentProjectIdRef.current
      const queued = pendingEditRef.current
      if (oldId && queued) api.saveEditBeacon(oldId, queued.removed, queued.trimSilences)
      pendingEditRef.current = null
      currentProjectIdRef.current = projectId
      try {
        const fresh = await api.getProject(projectId)
        if (generation !== navigationRef.current) return
        skipNextSaveRef.current = true
        pendingEditRef.current = null
        setUnsavedEdit(false)
        setProject(fresh)
        setRemoved(new Set(fresh.edit.removed_word_ids))
        setTrimSilences(fresh.edit.trim_silences)
        setRenderJob(null)
        setIngestJob(null)
        setTimeline(null)
        setCurrentView('edit')

        if (fresh.status === 'ready') {
          const next = await api.getTimeline(projectId)
          if (generation !== navigationRef.current) return
          setTimeline(next)
          if (fresh.last_render_job_id) {
            const job = await api.getJob(fresh.last_render_job_id, projectId).catch(() => null)
            if (generation !== navigationRef.current) return
            setRenderJob(job)
          }
          try {
            const raw = localStorage.getItem(draftKey(projectId))
            const draft = raw ? JSON.parse(raw) : null
            if (draft && draft.revision === fresh.revision) {
              pendingEditRef.current = {removed:draft.removed, trimSilences:draft.trimSilences}
              setRemoved(new Set(draft.removed))
              setTrimSilences(draft.trimSilences)
              setUnsavedEdit(true)
              skipNextSaveRef.current = false
            } else if (draft && (draft.trimSilences !== fresh.edit.trim_silences ||
                       JSON.stringify([...draft.removed].sort()) !==
                       JSON.stringify([...fresh.edit.removed_word_ids].sort()))) {
              setRecoveryDraft({removed:draft.removed, trimSilences:draft.trimSilences})
              warn('Yerli qaralama saxlanılıb, amma layihə dəyişib. Mövcud redaktəni yoxlayın.')
            }
          } catch { warn('Yerli qaralama oxuna bilmədi.') }
        }
      } catch (error) {
        handleApiError(error, 'Could not open the project')
      }
    },
    [handleApiError],
  )

  const closeProject = useCallback(() => {
    ++navigationRef.current
    ++editSeqRef.current
    window.clearTimeout(retryHandleRef.current)
    const queued = pendingEditRef.current
    const id = currentProjectIdRef.current
    if (id && queued) api.saveEditBeacon(id, queued.removed, queued.trimSilences)
    pendingEditRef.current = null
    currentProjectIdRef.current = null
    setUnsavedEdit(false)
    setRecoveryDraft(null)
    setProject(null)
    setTimeline(null)
    setRenderJob(null)
    setIngestJob(null)
    setActiveWordId(null)
    loadProjects()
  }, [loadProjects])

  // --------------------------------------------------------- ingest polling
  const processing = project?.status === 'processing' || project?.status === 'uploaded'
  useEffect(() => {
    if (!project || !processing) return
    let cancelled = false
    const tick = async () => {
      try {
        const [fresh, job] = await Promise.all([
          api.getProject(project.id),
          api.ingestJob(project.id).catch(() => null),
        ])
        if (cancelled) return
        setIngestJob(job)
        setProject(fresh)
        if (fresh.status === 'ready') {
          skipNextSaveRef.current = true
          setRemoved(new Set(fresh.edit.removed_word_ids))
          setTrimSilences(fresh.edit.trim_silences)
          const next = await api.getTimeline(fresh.id)
          if (cancelled) return
          setTimeline(next)
          loadProjects()
        } else if (fresh.status === 'failed' && fresh.error) {
          fail(fresh.error.message, fresh.error.detail)
        }
      } catch (error) {
        if (cancelled) return
        const status = error instanceof ApiRequestError ? error.status : 0
        if (status === 404) {
          fail(
            'This project no longer exists on the server.',
            'It was deleted while it was still processing.',
          )
          closeProject()
        }
        /* otherwise transient - keep polling */
      }
      if (!cancelled) timeoutHandle = window.setTimeout(tick, INGEST_POLL_MS)
    }
    let timeoutHandle: number
    void tick()
    return () => {
      cancelled = true
      window.clearTimeout(timeoutHandle)
    }
  }, [closeProject, fail, loadProjects, processing, project?.id])

  // ---------------------------------------------------------- edit autosave
  /** Send one edit, ignoring the answer if a newer edit has since been sent. */
  const pushEdit = useCallback(
    async (projectId: string, snapshot: EditSnapshot) => {
      const seq = ++editSeqRef.current
      inFlightEditsRef.current += 1
      setSavingEdit(true)
      try {
        const result = await api.saveEdit(projectId, snapshot.removed, snapshot.trimSilences)
        // Out-of-order guard: PUT /edit rebuilds the whole timeline server-side,
        // so latency varies and an older answer can arrive last. Applying it
        // would leave the preview and the render stats describing an older cut.
        if (seq !== editSeqRef.current || currentProjectIdRef.current !== projectId) return
        if (pendingEditRef.current && JSON.stringify(pendingEditRef.current) !== JSON.stringify(snapshot)) return
        setTimeline(result.timeline)
        setProject(result.project)
        pendingEditRef.current = null
        localStorage.removeItem(draftKey(projectId))
        retryCountRef.current = 0
        setUnsavedEdit(false)
      } catch (error) {
        if (seq !== editSeqRef.current || currentProjectIdRef.current !== projectId) return
        if (pendingEditRef.current && JSON.stringify(pendingEditRef.current) !== JSON.stringify(snapshot)) return
        handleApiError(error, 'Could not save the edit')
        setUnsavedEdit(true)
        // Keep the edit queued and retry with backoff: the transcript is
        // showing strikes the server does not have, and reopening the project
        // would silently discard them.
        if (retryCountRef.current < EDIT_MAX_RETRIES) {
          const attempt = retryCountRef.current++
          window.clearTimeout(retryHandleRef.current)
          retryHandleRef.current = window.setTimeout(
            () => {
              const queued = pendingEditRef.current
              if (queued && currentProjectIdRef.current === projectId) void pushEdit(projectId, queued)
            },
            EDIT_RETRY_BASE_MS * 2 ** attempt,
          )
        }
      } finally {
        inFlightEditsRef.current -= 1
        if (inFlightEditsRef.current === 0) setSavingEdit(false)
      }
    },
    [handleApiError],
  )

  useEffect(() => {
    if (!project || project.status !== 'ready') return
    if (skipNextSaveRef.current) {
      skipNextSaveRef.current = false
      return
    }
    const projectId = project.id
    const snapshot: EditSnapshot = { removed: [...removed], trimSilences }
    pendingEditRef.current = snapshot
    setUnsavedEdit(true)
    const handle = window.setTimeout(() => void pushEdit(projectId, snapshot), EDIT_DEBOUNCE_MS)
    return () => window.clearTimeout(handle)
  }, [project?.id, project?.status, pushEdit, removed, trimSilences])

  // Flush an unsaved edit when the project is closed or the tab goes away. The
  // debounce timer alone was simply cleared, so an edit made within 450 ms of
  // navigating was lost with no feedback at all.
  useEffect(() => {
    const projectId = project?.id
    if (!projectId || project?.status !== 'ready') return
    const flush = () => {
      const queued = pendingEditRef.current
      if (!queued) return
      pendingEditRef.current = null
      api.saveEditBeacon(projectId, queued.removed, queued.trimSilences)
    }
    window.addEventListener('pagehide', flush)
    window.addEventListener('beforeunload', flush)
    return () => {
      window.removeEventListener('pagehide', flush)
      window.removeEventListener('beforeunload', flush)
      flush()
    }
  }, [project?.id, project?.status])

  useEffect(() => () => window.clearTimeout(retryHandleRef.current), [])

  // --------------------------------------------------------- render polling
  const renderJobId = renderJob?.id ?? null
  const renderRunning = renderJob?.status === 'queued' || renderJob?.status === 'running'
  const projectId = project?.id ?? null

  useEffect(() => {
    if (!projectId || !renderJobId || !renderRunning) return
    // A self-scheduling loop keyed on the job *id*, not the job object: the old
    // effect depended on the whole polled value, so every response tore the
    // interval down and rebuilt it, and with no cancellation guard a late
    // response could resurrect a finished job after the project was closed.
    let cancelled = false
    let handle = 0
    let failures = 0

    const lose = (detail: string) => {
      fail('Lost contact with the render job.', detail)
      setRenderJob((current) =>
        current && current.id === renderJobId ? { ...current, status: 'failed' } : current,
      )
    }

    const tick = async () => {
      try {
        const job = await api.getJob(renderJobId, projectId)
        if (cancelled) return
        failures = 0
        setRenderJob(job)
        if (job.status === 'failed' && job.error) fail(job.error.message, job.error.detail)
        if (job.status === 'succeeded') {
          const fresh = await api.getProject(projectId)
          if (cancelled) return
          setProject(fresh)
        }
        if (job.status !== 'queued' && job.status !== 'running') return
      } catch (error) {
        if (cancelled) return
        const status = error instanceof ApiRequestError ? error.status : 0
        if (status >= 400 && status < 500) {
          // Terminal: the server does not know this job (restarted, or the
          // project was deleted). Polling it forever behind a silent 404 left
          // the Render button disabled and reading "Rendering…" for good.
          lose('The server no longer has a record of it. Start the render again.')
          return
        }
        failures += 1
        if (failures >= POLL_FAILURE_LIMIT) {
          lose(`${failures} consecutive polling failures.`)
          return
        }
      }
      if (!cancelled) handle = window.setTimeout(tick, RENDER_POLL_MS)
    }

    handle = window.setTimeout(tick, RENDER_POLL_MS)
    return () => {
      cancelled = true
      window.clearTimeout(handle)
    }
  }, [fail, projectId, renderJobId, renderRunning])

  // ----------------------------------------------------------- edit actions
  // History is mutated in the event handler, never inside a setState updater:
  // StrictMode double-invokes updaters in development, which pushed every
  // snapshot twice and made the first Cmd+Z a no-op.
  const commit = useCallback(
    (nextRemoved: Set<number>, nextTrim: boolean = trimSilences) => {
      const id = currentProjectIdRef.current
      const snapshot = {removed: [...nextRemoved], trimSilences: nextTrim}
      pendingEditRef.current = snapshot
      setUnsavedEdit(true)
      if (id) keepDraft(id, snapshot.removed, snapshot.trimSilences)
      setRemoved(nextRemoved)
      if (nextTrim !== trimSilences) setTrimSilences(nextTrim)
    },
    [removed, trimSilences],
  )

  const toggleWord = useCallback(
    (wordId: number, extend: boolean) => {
      const words = project?.transcript?.words ?? []
      const next = new Set(removed)
      const anchor = lastClickedRef.current

      if (extend && anchor !== null) {
        const idx1 = words.findIndex((w) => w.id === anchor)
        const idx2 = words.findIndex((w) => w.id === wordId)
        if (idx1 !== -1 && idx2 !== -1) {
          const from = Math.min(idx1, idx2)
          const to = Math.max(idx1, idx2)
          const shouldRemove = !removed.has(wordId)
          words
            .slice(from, to + 1)
            .forEach((word) => (shouldRemove ? next.add(word.id) : next.delete(word.id)))
        }
      } else if (next.has(wordId)) {
        next.delete(wordId)
      } else {
        next.add(wordId)
      }
      lastClickedRef.current = wordId
      commit(next)
    },
    [commit, project?.transcript?.words, removed],
  )

  const toggleSentence = useCallback(
    (sentenceId: number, remove: boolean) => {
      const sentence = project?.transcript?.sentences.find((item) => item.id === sentenceId)
      if (!sentence) return
      const next = new Set(removed)
      sentence.word_ids.forEach((id) => (remove ? next.add(id) : next.delete(id)))
      commit(next)
    },
    [commit, project?.transcript?.sentences, removed],
  )

  const updateWordText = useCallback(
    async (wordId: number, newText: string) => {
      if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
      try {
        const result = await api.updateWord(project.id, wordId, newText)
        if (currentProjectIdRef.current !== project.id) return
        setProject(result.project)
        setTimeline(result.timeline)
      } catch (error) {
        handleApiError(error, 'Could not update word text')
      }
    },
    [handleApiError, project],
  )

  const handleMergeWords = useCallback(
    async (wordIds: number[], text?: string) => {
      if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
      try {
        const result = await api.mergeWords(project.id, wordIds, text)
        if (currentProjectIdRef.current !== project.id) return
        setProject(result.project)
        setTimeline(result.timeline)
        skipNextSaveRef.current = true
        setRemoved(new Set(result.project.edit.removed_word_ids))
        setNotice({
          kind: 'info',
          message: `Sözlər birləşdirildi: “${result.merged_word.text}”`,
        })
      } catch (error) {
        handleApiError(error, 'Could not merge words')
      }
    },
    [handleApiError, project],
  )

  const handleProofread = useCallback(async () => {
    if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
    setIsProofreading(true)
    try {
      const result = await api.proofreadTranscript(project.id)
      if (currentProjectIdRef.current !== project.id) return
      setProject(result.project)
      if (result.ai_status !== 'succeeded') {
        setNotice({kind:'warn', message:'AI yoxlaması tam başa çatmadı.', detail:(result.warnings || []).join(' ')})
      } else if (result.suggestions_count > 0) {
        setNotice({
          kind: 'info',
          message: `AI ${result.suggestions_count} söz üçün orfoqrafik düzəliş təklif etdi.`,
        })
      } else {
        setNotice({
          kind: 'info',
          message: 'Mətndə heç bir orfoqrafik xəta tapılmadı - hər şey qaydasındadır!',
        })
      }
    } catch (error) {
      handleApiError(error, 'Could not proofread transcript')
    } finally {
      setIsProofreading(false)
    }
  }, [handleApiError, project])

  const handleAcceptSuggestion = useCallback(
    async (wordId: number) => {
      if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
      try {
        const result = await api.acceptSuggestion(project.id, wordId)
        if (currentProjectIdRef.current !== project.id) return
        setProject(result.project)
        setTimeline(result.timeline)
      } catch (error) {
        handleApiError(error, 'Could not accept suggestion')
      }
    },
    [handleApiError, project],
  )

  const handleDismissSuggestion = useCallback(
    async (wordId: number) => {
      if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
      try {
        const result = await api.dismissSuggestion(project.id, wordId)
        if (currentProjectIdRef.current !== project.id) return
        setProject(result.project)
      } catch (error) {
        handleApiError(error, 'Could not dismiss suggestion')
      }
    },
    [handleApiError, project],
  )

  const handleAcceptAllSuggestions = useCallback(async () => {
    if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
    try {
      const result = await api.acceptAllSuggestions(project.id)
      if (currentProjectIdRef.current !== project.id) return
      setProject(result.project)
      setTimeline(result.timeline)
      setNotice({
        kind: 'info',
        message: `${result.accepted_count} AI düzəlişi uğurla tətbiq edildi.`,
      })
    } catch (error) {
      handleApiError(error, 'Could not accept all suggestions')
    }
  }, [handleApiError, project])

  const handleDismissAllSuggestions = useCallback(async () => {
    if (!project) return
      if (editBusyRef.current) { warn("Əvvəlki redaktənin saxlanmasını gözləyin."); return }
    try {
      const result = await api.dismissAllSuggestions(project.id)
      if (currentProjectIdRef.current !== project.id) return
      setProject(result.project)
    } catch (error) {
      handleApiError(error, 'Could not dismiss all suggestions')
    }
  }, [handleApiError, project])

  const applyHistory = useCallback(async (direction: 'undo' | 'redo') => {
    const id = currentProjectIdRef.current
    if (!id || editBusyRef.current) return
    try {
      const result = await api.history(id, direction)
      if (currentProjectIdRef.current !== id) return
      ++editSeqRef.current
      pendingEditRef.current = null
      skipNextSaveRef.current = true
      setProject(result.project)
      setTimeline(result.timeline)
      setRemoved(new Set(result.project.edit.removed_word_ids))
      setTrimSilences(result.project.edit.trim_silences)
      localStorage.removeItem(draftKey(id))
    } catch (error) { handleApiError(error, 'Redaktəni geri qaytarmaq mümkün olmadı') }
  }, [handleApiError])
  const undo = useCallback(() => { void applyHistory('undo') }, [applyHistory])
  const redo = useCallback(() => { void applyHistory('redo') }, [applyHistory])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey
      if (!meta || event.key.toLowerCase() !== 'z') return
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return
      event.preventDefault()
      if (event.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [redo, undo])

  // ---------------------------------------------------------- b-roll actions
  /** Recalculate the timeline through the edit endpoint, not GET /timeline.
   *
   * GET /timeline is built from the *stored* edit and knows nothing about the
   * local `removed` set, so using it after a b-roll decision reverted the
   * preview and the render stats to the pre-strike cut.
   */
  const refreshTimeline = useCallback(
    async (id: string) => {
      const fresh = await api.getProject(id)
      if (currentProjectIdRef.current !== id) return
      setProject(fresh)
      if (pendingEditRef.current) await pushEdit(id, pendingEditRef.current)
      else {
        const next = await api.getTimeline(id)
        if (currentProjectIdRef.current === id) setTimeline(next)
      }
    },
    [pushEdit, removed, trimSilences],
  )

  const decideBroll = useCallback(
    async (suggestionId: string, status: BrollStatus, clipId?: string) => {
      if (!project) return
      setBrollBusyId(suggestionId)
      try {
        const updated = await api.decideBroll(project.id, suggestionId, status, clipId)
        if (currentProjectIdRef.current !== project.id) return
        setProject((current) =>
          current
            ? {
                ...current,
                broll: current.broll.map((item) =>
                  item.id === updated.id ? updated : item,
                ),
              }
            : current,
        )
        await refreshTimeline(project.id)
      } catch (error) {
        handleApiError(error, 'Could not update the B-roll suggestion')
      } finally {
        setBrollBusyId(null)
      }
    },
    [handleApiError, project, refreshTimeline],
  )

  const searchBroll = useCallback(
    async (suggestionId: string, query: string) => {
      if (!project) return
      setBrollBusyId(suggestionId)
      try {
        const updated = await api.searchBroll(project.id, suggestionId, query)
        if (currentProjectIdRef.current !== project.id) return
        setProject((current) =>
          current
            ? {
                ...current,
                broll: current.broll.map((item) =>
                  item.id === updated.id ? updated : item,
                ),
              }
            : current,
        )
        await refreshTimeline(project.id)
      } catch (error) {
        handleApiError(error, 'Stock search failed')
      } finally {
        setBrollBusyId(null)
      }
    },
    [handleApiError, project],
  )

  // ---------------------------------------------------------------- render
  const startRender = useCallback(
    async (options: RenderOptions) => {
      if (!project) return
      try {
        const job = await api.render(project.id, [...removed], options)
        if (currentProjectIdRef.current !== project.id) return
        setRenderJob(job)
        setNotice(null)
      } catch (error) {
        handleApiError(error, 'Could not start the render')
      }
    },
    [handleApiError, project, removed],
  )

  const deleteProject = useCallback(
    async (id: string) => {
      if (!window.confirm('Are you sure you want to delete this project? This cannot be undone.')) return
      try {
        await api.deleteProject(id)
        setProjects((current) => current.filter((item) => item.id !== id))
        if (project?.id === id) {
          currentProjectIdRef.current = null
          ++navigationRef.current
          ++editSeqRef.current
          localStorage.removeItem(draftKey(id))
          pendingEditRef.current = null
          setUnsavedEdit(false)
          setRecoveryDraft(null)
    setProject(null)
          setTimeline(null)
          setRenderJob(null)
          setIngestJob(null)
        }
      } catch (error) {
        handleApiError(error, 'Could not delete the project')
      }
    },
    [handleApiError, project?.id],
  )

  const configWarnings = useMemo(() => {
    if (!capabilities) return []
    const list = [...capabilities.warnings]
    if (!capabilities.groq_configured) list.push('GROQ_API_KEY is missing - uploads will fail.')
    if (capabilities.normalizer_provider === 'agy' && !capabilities.agy_available)
      list.push('AGY CLI tapılmadı; mətn AI yoxlaması işləməyəcək.')
    if (!capabilities.pexels_configured) list.push('PEXELS_API_KEY is missing - B-roll disabled.')
    if (!capabilities.font_file) list.push('No subtitle font file was found on this machine.')
    else if (!capabilities.font_covers_language)
      list.push('The subtitle font cannot render every Azerbaijani letter (ə ğ ı ş ç ö ü).')
    return list
  }, [capabilities])

  const ready = project?.status === 'ready' && project.transcript

  /** Source geometry as the viewer will see it: rotation metadata swaps axes. */
  const displayGeometry = useMemo(() => {
    const media = project?.media
    if (!media) return null
    const turned = Math.abs(media.rotation ?? 0) % 180 === 90
    return {
      width: media.width,
      height: media.height,
      rotated: turned,
    }
  }, [project?.media])

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand-dot" />
        <h1>Text-Based Reel Editor</h1>
        <span className="spacer" />
        {ready && <>
          <button className="ghost tiny" disabled={savingEdit || unsavedEdit || !project?.can_undo} onClick={undo}>Geri al</button>
          <button className="ghost tiny" disabled={savingEdit || unsavedEdit || !project?.can_redo} onClick={redo}>Təkrar et</button>
        </>}
        {savingEdit && (
          <span className="small muted row">
            <span className="spinner" /> recalculating timeline
          </span>
        )}
        {!savingEdit && unsavedEdit && (
          <span className="chip warn" title="This edit has not reached the server yet.">
            unsaved changes
          </span>
        )}
        {capabilities && (
          <span className="chip" title={capabilities.ffmpeg_version}>
            subs: {capabilities.subtitle_engine}
          </span>
        )}
        {project && (
          <div className="view-tabs">
            <button
              className={`tab-btn ${currentView === 'edit' ? 'active' : ''}`}
              onClick={() => setCurrentView('edit')}
            >
              Video Edit
            </button>
            <button
              className={`tab-btn ${currentView === 'share' ? 'active' : ''}`}
              onClick={() => setCurrentView('share')}
            >
              Share & Schedule
            </button>
          </div>
        )}
        {project && (
          <button className="ghost tiny" onClick={closeProject}>
            ← All projects
          </button>
        )}
      </header>

      <div className="layout">
        {currentView === 'edit' && (
          <main>
          {recoveryDraft && project && (
            <div className="banner warn">
              <strong>Saxlanmamış yerli qaralama mövcuddur.</strong>
              <button onClick={() => { commit(new Set(recoveryDraft.removed), recoveryDraft.trimSilences); setRecoveryDraft(null) }}>Qaralamanı bərpa et</button>
              <button onClick={() => { localStorage.removeItem(draftKey(project.id)); setRecoveryDraft(null) }}>Qaralamanı sil</button>
            </div>
          )}
          {notice && (
            <div className={`banner ${notice.kind}`}>
              <strong>{notice.message}</strong>
              {notice.detail && <pre>{notice.detail}</pre>}
              <button
                className="ghost tiny"
                style={{ marginTop: '0.5rem' }}
                onClick={() => setNotice(null)}
              >
                Dismiss
              </button>
            </div>
          )}

          {configWarnings.length > 0 && !project && (
            <div className="banner warn">
              <strong>Environment notes</strong>
              <ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.1rem' }}>
                {configWarnings.map((warning) => (
                  <li key={warning} className="small">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!project && (
            <UploadPanel
              recent={projects}
              onUploaded={(created) => {
                skipNextSaveRef.current = true
                pendingEditRef.current = null
                setUnsavedEdit(false)
                currentProjectIdRef.current = created.id
                ++navigationRef.current
                setProject(created)
                setRemoved(new Set())
                setTimeline(null)
                setRenderJob(null)
                setIngestJob(null)
                setNotice(null)

              }}
              onError={fail}
              onOpen={(id) => void openProject(id)}
              onDelete={(id) => void deleteProject(id)}
            />
          )}

          {project && !ready && (
            <ProcessingPanel project={project} job={ingestJob} onStartOver={closeProject}
              onCancel={() => { void api.cancel(project.id) }}
              onRetry={() => { void api.retryIngest(project.id).then(async job => {
                const fresh = await api.getProject(project.id)
                if (currentProjectIdRef.current !== project.id) return
                setProject(fresh); setIngestJob(job)
              }).catch(e => handleApiError(e, 'Emal yenidən başlamadı')) }} />
          )}

          {project && project.warnings.length > 0 && (
            <div className="banner warn">
              <strong>Pipeline warnings</strong>
              <ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.1rem' }}>
                {project.warnings.map((warning) => (
                  <li key={warning} className="small">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {ready && project.transcript && (
            <TranscriptEditor
              transcript={project.transcript}
              removed={removed}
              activeWordId={activeWordId}
              broll={project.broll}
              onToggleWord={toggleWord}
              onToggleSentence={toggleSentence}
              onSeek={(time) => setSeekTo({ time, nonce: Date.now() })}
              trimSilences={trimSilences}
              onTrimSilencesChange={(value) => commit(new Set(removed), value)}
              silenceCount={Math.max(0, project.speech_segments.length - 1)}
              onUpdateWordText={updateWordText}
              onMergeWords={handleMergeWords}
              onAcceptSuggestion={handleAcceptSuggestion}
              onDismissSuggestion={handleDismissSuggestion}
              onAcceptAllSuggestions={handleAcceptAllSuggestions}
              onDismissAllSuggestions={handleDismissAllSuggestions}
              onProofread={handleProofread}
              isProofreading={isProofreading}
            />
          )}
        </main>
        )}

        {currentView === 'edit' && (
        <aside className="sidebar">
          {project && (
            <>
              <PreviewPlayer
                src={api.sourceUrl(project.id)}
                timeline={timeline}
                seekTo={seekTo}
                onActiveWord={setActiveWordId}
              />
              {ready && (
                <>
                  <RenderPanel
                    projectId={project.id}
                    timeline={timeline}
                    capabilities={capabilities}
                    job={renderJob}
                    outputFilename={project.output_filename}
                    onRender={(options) => void startRender(options)}
                    busy={savingEdit || unsavedEdit}
                    stale={project.output_revision != null && project.output_revision !== project.revision}
                    onCancel={() => { void api.cancel(project.id) }}
                  />
                  <BrollPanel
                    suggestions={project.broll}
                    busyId={brollBusyId}
                    pexelsConfigured={capabilities?.pexels_configured ?? true}
                    onDecide={(id, status, clipId) => void decideBroll(id, status, clipId)}
                    onSearch={(id, query) => void searchBroll(id, query)}
                  />
                </>
              )}
              {project.media && displayGeometry && (
                <div className="panel">
                  <h2>Source</h2>
                  <div className="small muted mono">
                    {displayGeometry.width}×{displayGeometry.height}
                    {displayGeometry.rotated && ' (rotated)'} · {project.media.fps} fps ·{' '}
                    {project.media.duration.toFixed(1)}s
                    <br />
                    {project.media.video_codec}/{project.media.audio_codec}
                    {project.media.audio_channels
                      ? ` · ${project.media.audio_channels === 1 ? 'mono' : `${project.media.audio_channels}ch`}`
                      : ''}
                    {project.media.audio_sample_rate
                      ? ` ${(project.media.audio_sample_rate / 1000).toFixed(1)} kHz`
                      : ''}{' '}
                    · {(project.media.size_bytes / 1e6).toFixed(1)} MB
                    <br />
                    speech: {project.speech_duration.toFixed(1)}s in{' '}
                    {project.speech_segments.length} segments
                  </div>
                </div>
              )}
            </>
          )}
        </aside>
        )}

        {currentView === 'share' && project && (
          <div className="share-view">
            <SocialPanel
              project={project}
              onProjectUpdate={setProject}
              onError={(e) => handleApiError(e, 'Xəta baş verdi')}
            />
          </div>
        )}
      </div>
    </div>
  )
}
