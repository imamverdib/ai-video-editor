import { useState } from 'react'
import type { BrollStatus, BrollSuggestion } from '../types'

interface Props {
  suggestions: BrollSuggestion[]
  busyId: string | null
  pexelsConfigured: boolean
  onDecide: (id: string, status: BrollStatus, clipId?: string) => void
  onSearch: (id: string, query: string) => void
}

export default function BrollPanel({
  suggestions,
  busyId,
  pexelsConfigured,
  onDecide,
  onSearch,
}: Props) {
  const [queries, setQueries] = useState<Record<string, string>>({})

  return (
    <div className="panel">
      <h2>B-roll · {suggestions.filter((s) => s.status === 'accepted').length} accepted</h2>
      {!pexelsConfigured && (
        <div className="banner warn small">
          PEXELS_API_KEY is not configured, so stock search is unavailable.
        </div>
      )}
      {suggestions.length === 0 && (
        <p className="small muted">No cutaway moments were suggested for this Reel.</p>
      )}

      {suggestions.map((suggestion) => {
        const busy = busyId === suggestion.id
        const selected = suggestion.selected_clip_id
        const query = queries[suggestion.id] ?? suggestion.query
        return (
          <div key={suggestion.id} className={`broll-card ${suggestion.status}`}>
            <div className="row">
              <h4 style={{ flex: 1 }}>“{suggestion.query}”</h4>
              <span className={`chip ${suggestion.status}`}>{suggestion.status}</span>
            </div>
            {suggestion.reason && <p className="small muted" style={{ margin: '0 0 0.4rem' }}>{suggestion.reason}</p>}
            <p className="small dim mono" style={{ margin: '0 0 0.4rem' }}>
              @{suggestion.source_start.toFixed(1)}s ·{' '}
              {(suggestion.source_end - suggestion.source_start).toFixed(1)}s · sentence #
              {suggestion.sentence_id}
            </p>

            {suggestion.search_error && (
              <div className="banner warn small" style={{ marginBottom: '0.5rem' }}>
                {suggestion.search_error}
              </div>
            )}

            {suggestion.candidates.length > 0 && (
              <div className="clip-strip">
                {suggestion.candidates.map((clip) => (
                  <button
                    key={clip.id}
                    className={`clip-thumb${selected === clip.id ? ' selected' : ''}`}
                    title={`${clip.width}×${clip.height} · ${clip.duration}s${
                      clip.author ? ` · ${clip.author}` : ''
                    }`}
                    disabled={busy}
                    onClick={() => onDecide(suggestion.id, suggestion.status, clip.id)}
                  >
                    {clip.preview_image ? (
                      <img src={clip.preview_image} alt={clip.query} loading="lazy" />
                    ) : (
                      <span style={{ position: 'static' }}>{clip.provider_id}</span>
                    )}
                    <span>{Math.round(clip.duration)}s</span>
                  </button>
                ))}
              </div>
            )}

            <div className="row" style={{ marginTop: '0.5rem' }}>
              <input
                type="text"
                value={query}
                placeholder="New search query…"
                disabled={busy || !pexelsConfigured}
                onChange={(event) =>
                  setQueries((prev) => ({ ...prev, [suggestion.id]: event.target.value }))
                }
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && query.trim()) onSearch(suggestion.id, query.trim())
                }}
              />
              <button
                className="tiny"
                disabled={busy || !query.trim() || !pexelsConfigured}
                onClick={() => onSearch(suggestion.id, query.trim())}
              >
                Swap
              </button>
            </div>

            <div className="row" style={{ marginTop: '0.5rem' }}>
              <button
                className="primary tiny"
                disabled={busy || suggestion.candidates.length === 0}
                onClick={() => onDecide(suggestion.id, 'accepted')}
              >
                Accept
              </button>
              <button
                className="tiny"
                disabled={busy}
                onClick={() => onDecide(suggestion.id, 'pending')}
              >
                Reset
              </button>
              <button
                className="tiny danger"
                disabled={busy}
                onClick={() => onDecide(suggestion.id, 'dismissed')}
              >
                Dismiss
              </button>
              {busy && <span className="spinner" />}
            </div>
          </div>
        )
      })}
    </div>
  )
}
