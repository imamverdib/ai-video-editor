import type {
  BrollStatus,
  BrollSuggestion,
  Capabilities,
  Job,
  Project,
  RenderOptions,
  Timeline,
  Word,
} from './types'

const queues = new Map<string, Promise<unknown>>()
const revisions = new Map<string, number>()
const clientId = sessionStorage.getItem('editor-client') || crypto.randomUUID()
sessionStorage.setItem('editor-client', clientId)
let sequence = Date.now() * 1000
export const draftKey = (id: string) => `editor-draft:${id}`
export function keepDraft(id: string, removed: number[], trimSilences: boolean) {
  localStorage.setItem(draftKey(id), JSON.stringify({ removed, trimSilences, revision: revisions.get(id) }))
}
const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiRequestError extends Error {
  code: string
  detail?: string
  status: number

  constructor(status: number, code: string, message: string, detail?: string) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
    this.code = code
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { ...init, signal: init?.signal ?? AbortSignal.timeout(300_000) })
  if (!response.ok) {
    let code = 'http_error'
    let message = `${response.status} ${response.statusText}`
    let detail: string | undefined
    try {
      const payload = await response.json()
      if (payload?.error) {
        code = payload.error.code ?? code
        message = payload.error.message ?? message
        detail = payload.error.detail
      } else if (payload?.detail) {
        message = typeof payload.detail === 'string' ? payload.detail : message
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiRequestError(response.status, code, message, detail)
  }
  if (response.status === 204) return undefined as T
  const value = await response.json()
  const project = value?.project ?? value
  if (project?.id && typeof project.revision === 'number') {
    revisions.set(project.id, Math.max(project.revision, revisions.get(project.id) ?? -1))
  }
  return value as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  capabilities: () => request<Capabilities>('/api/capabilities'),

  listProjects: () => request<Project[]>('/api/projects'),

  getProject: (id: string) => request<Project>(`/api/projects/${id}`),

  deleteProject: (id: string) =>
    request<void>(`/api/projects/${id}`, { method: 'DELETE' }),

  ingestJob: (id: string) => request<Job>(`/api/projects/${id}/job`),

  getJob: (jobId: string, projectId?: string) =>
    request<Job>(
      `/api/jobs/${jobId}${projectId ? `?project_id=${projectId}` : ''}`,
    ),

  uploadSource: (file: File, name: string) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('name', name)
    return request<Project>('/api/projects', { method: 'POST', body: formData })
  },

  generateSocialText: (id: string) =>
    request<{ social_posts: Record<string, import('./types').SocialPost> }>(
      `/api/projects/${id}/social/generate_text`,
      { method: 'POST' }
    ),

  generateSocialCover: (id: string, coverTitle?: string) =>
    request<{ cover_image: string }>(
      `/api/projects/${id}/social/generate_cover`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cover_title: coverTitle }),
      }
    ),

  publishSocialPost: (id: string, platform: string, schedule_time?: number) =>
    request<{ status: string }>(
      `/api/projects/${id}/social/publish`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ platform, schedule_time }),
      }
    ),

  upload: (file: File, onProgress?: (fraction: number) => void) =>
    new Promise<Project>((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/api/projects`)
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as Project)
          return
        }
        try {
          const payload = JSON.parse(xhr.responseText)
          reject(
            new ApiRequestError(
              xhr.status,
              payload?.error?.code ?? 'http_error',
              payload?.error?.message ?? 'Upload failed',
              payload?.error?.detail,
            ),
          )
        } catch {
          reject(new ApiRequestError(xhr.status, 'http_error', 'Upload failed'))
        }
      }
      xhr.onerror = () =>
        reject(new ApiRequestError(0, 'network_error', 'Network error during upload'))
      xhr.send(form)
    }),

  getTimeline: (id: string) => request<Timeline>(`/api/projects/${id}/timeline`),

  saveEdit: (id: string, removedWordIds: number[], trimSilences: boolean) => {
    const seq = ++sequence
    keepDraft(id, removedWordIds, trimSilences)
    const send = () => request<{ project: Project; timeline: Timeline }>(`/api/projects/${id}/edit`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({removed_word_ids: removedWordIds, trim_silences: trimSilences,
        client_id: clientId, sequence: seq, expected_revision: revisions.get(id)}),
    })
    const pending = (queues.get(id) ?? Promise.resolve()).catch(() => {}).then(send)
    queues.set(id, pending)
    return pending
  },

  history: (id: string, direction: 'undo' | 'redo') =>
    request<{project: Project; timeline: Timeline}>(`/api/projects/${id}/history/${direction}`, {method:'POST'}),
  retryIngest: (id: string) => request<Job>(`/api/projects/${id}/retry`, {method:'POST'}),
  cancel: (id: string) => request(`/api/projects/${id}/cancel`, {method:'POST'}),

  updateWord: (id: string, wordId: number, text: string) =>
    request<{ project: Project; timeline: Timeline; word: Word }>(
      `/api/projects/${id}/words/${wordId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      },
    ),

  mergeWords: (id: string, wordIds: number[], text?: string) =>
    request<{ project: Project; timeline: Timeline; merged_word: Word }>(
      `/api/projects/${id}/words/merge`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word_ids: wordIds, text }),
      },
    ),

  proofreadTranscript: (id: string) =>
    request<{ project: Project; suggestions_count: number; warnings: string[]; ai_status: string }>(`/api/projects/${id}/proofread`, {
      method: 'POST',
    }),

  acceptSuggestion: (id: string, wordId: number) =>
    request<{ project: Project; timeline: Timeline; word: Word }>(
      `/api/projects/${id}/words/${wordId}/suggestion/accept`,
      { method: 'POST' },
    ),

  dismissSuggestion: (id: string, wordId: number) =>
    request<{ project: Project; word: Word }>(
      `/api/projects/${id}/words/${wordId}/suggestion/dismiss`,
      { method: 'POST' },
    ),

  acceptAllSuggestions: (id: string) =>
    request<{ project: Project; timeline: Timeline; accepted_count: number }>(
      `/api/projects/${id}/suggestions/accept-all`,
      { method: 'POST' },
    ),

  dismissAllSuggestions: (id: string) =>
    request<{ project: Project }>(`/api/projects/${id}/suggestions/dismiss-all`, {
      method: 'POST',
    }),

  saveEditBeacon: (id: string, removedWordIds: number[], trimSilences: boolean) => {
    keepDraft(id, removedWordIds, trimSilences)
    // JSON content type matches FastAPI. The durable local draft survives a failed unload request.
    void fetch(`${BASE}/api/projects/${id}/edit`, {
      method:'POST', headers:{'Content-Type':'application/json'}, keepalive:true,
      body:JSON.stringify({removed_word_ids:removedWordIds, trim_silences:trimSilences,
        client_id:clientId, sequence:++sequence, expected_revision:revisions.get(id)}),
    }).catch(() => {})
  },

  searchBroll: (id: string, suggestionId: string, query: string) =>
    request<BrollSuggestion>(
      `/api/projects/${id}/broll/${suggestionId}/search`,
      json({ query }),
    ),

  decideBroll: (
    id: string,
    suggestionId: string,
    status: BrollStatus,
    clipId?: string,
  ) =>
    request<BrollSuggestion>(
      `/api/projects/${id}/broll/${suggestionId}/decision`,
      json({ status, clip_id: clipId ?? null }),
    ),

  render: (id: string, removedWordIds: number[], options: RenderOptions) =>
    request<Job>(
      `/api/projects/${id}/render`,
      json({ removed_word_ids: removedWordIds, options }),
    ),

  sourceUrl: (id: string) => `${BASE}/api/projects/${id}/source`,
  outputUrl: (id: string, name?: string | null) =>
    `${BASE}/api/projects/${id}/output${name ? `?name=${encodeURIComponent(name)}` : ''}`,
  downloadUrl: (id: string, name?: string | null) =>
    `${BASE}/api/projects/${id}/download${name ? `?name=${encodeURIComponent(name)}` : ''}`,
}

export const formatTime = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const whole = Math.floor(seconds)
  const mins = Math.floor(whole / 60)
  const secs = whole % 60
  return `${mins}:${String(secs).padStart(2, '0')}`
}

export const formatPrecise = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0'
  return `${formatTime(seconds)}.${String(Math.floor((seconds % 1) * 10))}`
}
