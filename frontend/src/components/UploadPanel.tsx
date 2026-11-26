import { useCallback, useRef, useState } from 'react'
import { api } from '../api'
import type { Project } from '../types'

interface Props {
  onUploaded: (project: Project) => void
  onError: (message: string, detail?: string) => void
  recent: Project[]
  onOpen: (projectId: string) => void
  onDelete: (projectId: string) => void
  disabled?: boolean
}

const ACCEPT = '.mp4,.mov,.m4v,.mkv,.webm,.avi'

export default function UploadPanel({
  onUploaded,
  onError,
  recent,
  onOpen,
  onDelete,
  disabled,
}: Props) {
  const [over, setOver] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const upload = useCallback(
    async (file: File) => {
      setProgress(0)
      try {
        const project = await api.upload(file, setProgress)
        onUploaded(project)
      } catch (error) {
        const err = error as { message?: string; detail?: string }
        onError(err.message ?? 'Upload failed', err.detail)
      } finally {
        setProgress(null)
      }
    },
    [onError, onUploaded],
  )

  return (
    <div className="panel">
      <h2>New Reel</h2>
      <div
        className={`dropzone${over ? ' over' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault()
          setOver(false)
          const file = event.dataTransfer.files?.[0]
          if (file) void upload(file)
        }}
      >
        <h3>Drop a vertical video here</h3>
        <p className="small">
          9:16 MP4 / MOV with an audio track · up to the server upload limit
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) void upload(file)
            event.target.value = ''
          }}
        />
        <button
          className="primary"
          disabled={disabled || progress !== null}
          onClick={() => inputRef.current?.click()}
        >
          {progress !== null ? 'Uploading…' : 'Choose file'}
        </button>
        {progress !== null && (
          <div style={{ marginTop: '1rem' }}>
            <div className="progress">
              <i style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <div className="small muted" style={{ marginTop: '0.35rem' }}>
              {Math.round(progress * 100)}% uploaded
            </div>
          </div>
        )}
      </div>

      {recent.length > 0 && (
        <>
          <h2 style={{ marginTop: '1.2rem' }}>Recent projects</h2>
          <div className="project-list">
            {recent.map((project) => (
              <div key={project.id} className="row">
                <button className="project-row" onClick={() => onOpen(project.id)}>
                  <span className="name">{project.original_filename}</span>
                  <span className={`chip ${project.status === 'ready' ? 'accepted' : 'pending'}`}>
                    {project.status}
                  </span>
                </button>
                <button
                  className="ghost tiny danger"
                  title="Delete project"
                  onClick={() => onDelete(project.id)}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
