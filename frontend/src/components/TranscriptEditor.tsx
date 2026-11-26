import { useMemo, useState } from 'react'
import { formatPrecise } from '../api'
import type { BrollSuggestion, Transcript, Word } from '../types'

interface Props {
  transcript: Transcript
  removed: Set<number>
  activeWordId: number | null
  broll: BrollSuggestion[]
  onToggleWord: (wordId: number, extend: boolean) => void
  onToggleSentence: (sentenceId: number, remove: boolean) => void
  onSeek: (seconds: number) => void
  trimSilences: boolean
  onTrimSilencesChange: (value: boolean) => void
  silenceCount: number
  onUpdateWordText?: (wordId: number, newText: string) => Promise<void> | void
  onMergeWords?: (wordIds: number[], text?: string) => Promise<void> | void
  onAcceptSuggestion?: (wordId: number) => Promise<void> | void
  onDismissSuggestion?: (wordId: number) => Promise<void> | void
  onAcceptAllSuggestions?: () => Promise<void> | void
  onDismissAllSuggestions?: () => Promise<void> | void
  onProofread?: () => Promise<void> | void
  isProofreading?: boolean
}

export default function TranscriptEditor({
  transcript,
  removed,
  activeWordId,
  broll,
  onToggleWord,
  onToggleSentence,
  onSeek,
  trimSilences,
  onTrimSilencesChange,
  silenceCount,
  onUpdateWordText,
  onMergeWords,
  onAcceptSuggestion,
  onDismissSuggestion,
  onAcceptAllSuggestions,
  onDismissAllSuggestions,
  onProofread,
  isProofreading,
}: Props) {
  const [editingWordId, setEditingWordId] = useState<number | null>(null)
  const [editText, setEditText] = useState<string>('')
  const [hoveredSuggestionWordId, setHoveredSuggestionWordId] = useState<number | null>(null)
  const [mergingState, setMergingState] = useState<{
    currentWord: Word
    nextWord: Word
    text: string
  } | null>(null)

  const wordsById = useMemo(() => {
    const map = new Map<number, Word>()
    transcript.words.forEach((word) => map.set(word.id, word))
    return map
  }, [transcript.words])

  const brollBySentence = useMemo(() => {
    const map = new Map<number, BrollSuggestion[]>()
    broll.forEach((item) => {
      const list = map.get(item.sentence_id) ?? []
      list.push(item)
      map.set(item.sentence_id, list)
    })
    return map
  }, [broll])

  const pendingSuggestions = useMemo(
    () => transcript.words.filter((w) => Boolean(w.suggestion)),
    [transcript.words],
  )

  const startEditing = (word: Word, e?: React.MouseEvent | React.KeyboardEvent) => {
    if (e) e.stopPropagation()
    setHoveredSuggestionWordId(null)
    setMergingState(null)
    setEditingWordId(word.id)
    setEditText(word.text)
  }

  const saveEditing = async (word: Word) => {
    const trimmed = editText.trim()
    if (!trimmed || trimmed === word.text) {
      setEditingWordId(null)
      return
    }
    if (onUpdateWordText) {
      await onUpdateWordText(word.id, trimmed)
    }
    setEditingWordId(null)
  }

  const cancelEditing = () => {
    setEditingWordId(null)
  }

  const startMerging = (currentWord: Word, nextWord: Word, e?: React.MouseEvent) => {
    if (e) e.stopPropagation()
    setEditingWordId(null)
    setHoveredSuggestionWordId(null)
    const clean1 = currentWord.text.trim()
    const clean2 = nextWord.text.trim()
    const combined = `${clean1}${clean2}`
    setMergingState({ currentWord, nextWord, text: combined })
  }

  const confirmMerge = async () => {
    if (!mergingState || !onMergeWords) return
    const { currentWord, nextWord, text } = mergingState
    const trimmed = text.trim() || `${currentWord.text}${nextWord.text}`
    await onMergeWords([currentWord.id, nextWord.id], trimmed)
    setMergingState(null)
  }

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>
          Transcript · {transcript.words.length - removed.size}/{transcript.words.length} words kept
        </h2>
        {onProofread && (
          <button
            className="ghost tiny"
            disabled={isProofreading}
            onClick={() => void onProofread()}
            title="AI ilə orfoqrafik xətaları yoxla"
            style={{ color: '#c084fc', borderColor: 'rgba(168, 85, 247, 0.4)' }}
          >
            {isProofreading ? (
              <>
                <span className="spinner" /> Yoxlanılır…
              </>
            ) : (
              <>✨ AI ilə Yoxla</>
            )}
          </button>
        )}
      </div>

      <label className="switch">
        <input
          type="checkbox"
          checked={trimSilences}
          onChange={(event) => onTrimSilencesChange(event.target.checked)}
        />
        Tighten pauses — also cut the dead air Silero VAD found ({silenceCount} silent
        gaps detected)
      </label>
      <p className="small muted" style={{ marginTop: '-0.35rem' }}>
        Click a word to strike it out · Double-click or click ✎ to edit text ·{' '}
        <span className="kbd">Shift</span>+click to strike a range.
      </p>

      {pendingSuggestions.length > 0 && (
        <div className="suggestions-banner">
          <div className="row" style={{ gap: '0.4rem' }}>
            <span>✨</span>
            <span>
              AI <strong>{pendingSuggestions.length} söz</strong> üçün orfoqrafik düzəliş təklif edir
            </span>
          </div>
          <div className="suggestions-banner-actions">
            {onAcceptAllSuggestions && (
              <button
                className="tiny suggestion-btn accept"
                onClick={() => void onAcceptAllSuggestions()}
                title="Bütün AI düzəlişlərini tətbiq et"
              >
                ✓ Hamısını qəbul et
              </button>
            )}
            {onDismissAllSuggestions && (
              <button
                className="tiny suggestion-btn dismiss"
                onClick={() => void onDismissAllSuggestions()}
                title="Bütün təklifləri ləğv et"
              >
                ✕ İmtina et
              </button>
            )}
          </div>
        </div>
      )}

      <div className="transcript">
        {transcript.sentences.map((sentence) => {
          const sentenceWords = sentence.word_ids
            .map((id) => wordsById.get(id))
            .filter((word): word is Word => Boolean(word))
          const allRemoved =
            sentenceWords.length > 0 && sentenceWords.every((word) => removed.has(word.id))
          const suggestions = brollBySentence.get(sentence.id) ?? []

          return (
            <div key={sentence.id}>
              {suggestions.map((suggestion) => (
                <div
                  key={suggestion.id}
                  className={`broll-inline ${suggestion.status}`}
                  title={suggestion.reason}
                >
                  <span aria-hidden>🎬</span>
                  <span>
                    <strong>B-roll</strong> “{suggestion.query}” ·{' '}
                    {(suggestion.source_end - suggestion.source_start).toFixed(1)}s at{' '}
                    {formatPrecise(suggestion.source_start)}
                  </span>
                  <span className={`chip ${suggestion.status}`}>{suggestion.status}</span>
                </div>
              ))}

              <div className={`sentence${allRemoved ? ' removed' : ''}`}>
                <div className="sentence-tools">
                  <button
                    className="tiny ghost"
                    title={allRemoved ? 'Restore sentence' : 'Delete sentence'}
                    onClick={() => onToggleSentence(sentence.id, !allRemoved)}
                  >
                    {allRemoved ? '↺' : '✕'}
                  </button>
                </div>
                {sentenceWords.map((word, wordIdx) => {
                  const nextWord = sentenceWords[wordIdx + 1] ?? null
                  if (editingWordId === word.id) {
                    return (
                      <span
                        key={word.id}
                        className="token-inline-editor"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          className="token-inline-input"
                          value={editText}
                          autoFocus
                          onChange={(e) => setEditText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault()
                              void saveEditing(word)
                            } else if (e.key === 'Escape') {
                              e.preventDefault()
                              cancelEditing()
                            }
                          }}
                          onBlur={() => {
                            void saveEditing(word)
                          }}
                        />
                        <button
                          type="button"
                          className="token-inline-btn save"
                          title="Save text (Enter)"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => void saveEditing(word)}
                        >
                          ✓
                        </button>
                        <button
                          type="button"
                          className="token-inline-btn cancel"
                          title="Cancel (Esc)"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={cancelEditing}
                        >
                          ✕
                        </button>
                      </span>
                    )
                  }

                  const hasSuggestion = Boolean(word.suggestion)
                  const isEdited = Boolean(word.raw_text && word.raw_text !== word.text)
                  const showPopover = hoveredSuggestionWordId === word.id && hasSuggestion
                  const showMergePopover = mergingState !== null && mergingState.currentWord.id === word.id

                  const tooltip = isEdited
                    ? `Original: “${word.raw_text}” · Corrected: “${word.text}” (${word.start.toFixed(2)}s)`
                    : word.raw_text
                      ? `ASR heard “${word.raw_text}” · ${word.start.toFixed(2)}s`
                      : `${word.start.toFixed(2)}s → ${word.end.toFixed(2)}s`

                  return (
                    <span
                      key={word.id}
                      className={[
                        'token',
                        removed.has(word.id) ? 'removed' : '',
                        activeWordId === word.id ? 'active' : '',
                        hasSuggestion ? 'has-suggestion' : isEdited ? 'edited' : word.raw_text ? 'normalized' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      title={hasSuggestion || showMergePopover ? undefined : tooltip}
                      onMouseEnter={() => {
                        if (hasSuggestion) setHoveredSuggestionWordId(word.id)
                      }}
                      onMouseLeave={() => {
                        if (hoveredSuggestionWordId === word.id) setHoveredSuggestionWordId(null)
                      }}
                      onClick={(event) => {
                        if (event.altKey) {
                          onSeek(word.start)
                          return
                        }
                        if (hasSuggestion && !removed.has(word.id)) {
                          setHoveredSuggestionWordId((prev) => (prev === word.id ? null : word.id))
                          return
                        }
                        onToggleWord(word.id, event.shiftKey)
                      }}
                      onDoubleClick={(event) => {
                        event.stopPropagation()
                        startEditing(word, event)
                      }}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          if (e.altKey) {
                            onSeek(word.start)
                            return
                          }
                          onToggleWord(word.id, e.shiftKey)
                        } else if (e.key === 'e' || e.key === 'E') {
                          e.preventDefault()
                          startEditing(word, e)
                        }
                      }}
                    >
                      <span>{word.text}</span>
                      {hasSuggestion && <span className="token-suggestion-icon">✨</span>}
                      {onUpdateWordText && !removed.has(word.id) && !hasSuggestion && (
                        <button
                          type="button"
                          className="token-edit-btn"
                          title="Sözü redaktə et (Double-click / E)"
                          aria-label={`Edit word ${word.text}`}
                          onClick={(e) => {
                            e.stopPropagation()
                            startEditing(word, e)
                          }}
                        >
                          ✎
                        </button>
                      )}
                      {onMergeWords && nextWord && !removed.has(word.id) && !removed.has(nextWord.id) && !hasSuggestion && (
                        <button
                          type="button"
                          className="token-merge-btn"
                          title={`"${word.text}" və "${nextWord.text}" sözlərini birləşdir`}
                          aria-label={`Merge "${word.text}" with "${nextWord.text}"`}
                          onClick={(e) => {
                            e.stopPropagation()
                            startMerging(word, nextWord, e)
                          }}
                        >
                          🔗
                        </button>
                      )}

                      {showMergePopover && mergingState && (
                        <div className="merge-popover" onClick={(e) => e.stopPropagation()}>
                          <div className="merge-header">
                            <span>🔗 Sözləri Birləşdir (Merge)</span>
                          </div>
                          <div className="merge-preview">
                            <span className="part">“{mergingState.currentWord.text}”</span>
                            <span className="plus">+</span>
                            <span className="part">“{mergingState.nextWord.text}”</span>
                            <span className="arrow">➔</span>
                          </div>
                          <div className="merge-input-row">
                            <input
                              className="token-inline-input"
                              autoFocus
                              value={mergingState.text}
                              onChange={(e) =>
                                setMergingState({ ...mergingState, text: e.target.value })
                              }
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  e.preventDefault()
                                  void confirmMerge()
                                } else if (e.key === 'Escape') {
                                  e.preventDefault()
                                  setMergingState(null)
                                }
                              }}
                            />
                            <button
                              type="button"
                              className="suggestion-btn accept"
                              title="Birləşdir (Enter)"
                              onClick={() => void confirmMerge()}
                            >
                              ✓
                            </button>
                            <button
                              type="button"
                              className="suggestion-btn dismiss"
                              title="İmtina (Esc)"
                              onClick={() => setMergingState(null)}
                            >
                              ✕
                            </button>
                          </div>
                        </div>
                      )}

                      {showPopover && (
                        <div
                          className="suggestion-popover"
                          onClick={(e) => e.stopPropagation()}
                          onMouseEnter={() => setHoveredSuggestionWordId(word.id)}
                          onMouseLeave={() => setHoveredSuggestionWordId(null)}
                        >
                          <div className="suggestion-header">
                            <span>✨ AI Düzəliş Təklifi</span>
                          </div>
                          <div className="suggestion-diff">
                            <span className="original">{word.text}</span>
                            <span className="arrow">➔</span>
                            <span className="suggested">{word.suggestion}</span>
                          </div>
                          {word.suggestion_reason && (
                            <div className="suggestion-reason">{word.suggestion_reason}</div>
                          )}
                          <div className="suggestion-actions">
                            {onAcceptSuggestion && (
                              <button
                                type="button"
                                className="suggestion-btn accept"
                                title="Təklifi qəbul et"
                                onClick={async (e) => {
                                  e.stopPropagation()
                                  setHoveredSuggestionWordId(null)
                                  await onAcceptSuggestion(word.id)
                                }}
                              >
                                ✓ Qəbul et
                              </button>
                            )}
                            {onDismissSuggestion && (
                              <button
                                type="button"
                                className="suggestion-btn dismiss"
                                title="Təklifdən imtina et"
                                onClick={async (e) => {
                                  e.stopPropagation()
                                  setHoveredSuggestionWordId(null)
                                  await onDismissSuggestion(word.id)
                                }}
                              >
                                ✕ İmtina
                              </button>
                            )}
                            <button
                              type="button"
                              className="suggestion-btn dismiss"
                              title="Əl ilə redaktə et"
                              onClick={(e) => {
                                e.stopPropagation()
                                startEditing(word, e)
                              }}
                            >
                              ✎ Redaktə
                            </button>
                          </div>
                        </div>
                      )}
                    </span>
                  )
                })}
                <span className="sentence-time mono">{formatPrecise(sentence.start)}</span>
              </div>
            </div>
          )
        })}
      </div>
      <p className="small dim" style={{ marginTop: '1rem' }}>
        <span className="kbd">Alt</span>+click a word to scrub preview ·
        Double-click or click ✎ to edit text ·
        <span style={{ color: '#c084fc' }}> ✨ bənövşəyi = AI təklifi</span> ·
        solid underline = user edited.
      </p>
    </div>
  )
}
