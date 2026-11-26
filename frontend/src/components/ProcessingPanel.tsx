import type { Job, Project } from '../types'

const STAGES: Array<{ key: string; label: string }> = [
  { key: 'probing', label: 'Inspecting streams' },
  { key: 'extracting_audio', label: 'Extracting 16 kHz mono audio' },
  { key: 'detecting_speech', label: 'Silero VAD · speech boundaries' },
  { key: 'transcribing', label: 'Groq whisper-large-v3 · word timestamps' },
  { key: 'normalizing', label: 'Terminology normalisation' },
  { key: 'suggesting_broll', label: 'Contextual B-roll suggestions' },
  { key: 'done', label: 'Ready to edit' },
]

interface Props {
  project: Project
  job: Job | null
  onStartOver?: () => void
  onRetry?: () => void
  onCancel?: () => void
}

export default function ProcessingPanel({ project, job, onStartOver, onRetry, onCancel }: Props) {
  const progress = job?.progress ?? project.progress
  const currentIndex = STAGES.findIndex((stage) => stage.key === project.stage)
  const failed = project.status === 'failed' || job?.status === 'failed' || job?.status === 'cancelled'

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2>Processing “{project.original_filename}”</h2>
        {onStartOver && (
          <button className="ghost tiny" onClick={onStartOver}>
            ← Start over
          </button>
        )}
      </div>
      {!failed && <button className="ghost tiny" onClick={onCancel}>Emalı dayandır</button>}
      <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <i style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <div className="row small muted" style={{ marginTop: '0.5rem' }}>
        {!failed && <span className="spinner" />}
        <span>{job?.message || project.message || 'Working…'}</span>
        <span className="spacer" style={{ flex: 1 }} />
        <span className="mono">{Math.round(progress * 100)}%</span>
      </div>

      {failed && (
        <div className="banner error" style={{ marginTop: '0.8rem' }}>
          <strong>Processing failed</strong>
          {onRetry && <button className="primary tiny" onClick={onRetry}>Yenidən emal et</button>}
          <p className="small" style={{ margin: '0.2rem 0' }}>
            {project.error?.message || job?.error?.message || 'An error occurred during ingest.'}
          </p>
          {onStartOver && (
            <button
              className="primary tiny"
              style={{ marginTop: '0.4rem' }}
              onClick={onStartOver}
            >
              Choose another file
            </button>
          )}
        </div>
      )}

      <ul style={{ listStyle: 'none', padding: 0, margin: '1rem 0 0' }}>
        {STAGES.map((stage, index) => {
          const done = currentIndex > index
          const active = currentIndex === index
          return (
            <li
              key={stage.key}
              className="small"
              style={{
                display: 'flex',
                gap: '0.55rem',
                padding: '0.22rem 0',
                color: done ? 'var(--ok)' : active ? 'var(--text)' : 'var(--dim)',
              }}
            >
              <span style={{ width: '1rem' }}>{done ? '✓' : active ? '›' : '·'}</span>
              <span>{stage.label}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
