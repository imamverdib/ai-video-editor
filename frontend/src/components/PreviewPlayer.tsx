import { useCallback, useEffect, useRef, useState } from 'react'
import { formatTime } from '../api'
import type { Timeline } from '../types'

interface Props {
  src: string
  timeline: Timeline | null
  seekTo: { time: number; nonce: number } | null
  onActiveWord: (wordId: number | null) => void
}

/**
 * Plays the *edited* cut without rendering: whenever playback reaches the end
 * of a surviving clip it jumps straight to the start of the next one, so what
 * you hear is what the exported Reel will contain.
 */
export default function PreviewPlayer({ src, timeline, seekTo, onActiveWord }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [position, setPosition] = useState(0)
  const [followCut, setFollowCut] = useState(true)

  useEffect(() => {
    if (!seekTo || !videoRef.current) return
    videoRef.current.currentTime = seekTo.time
  }, [seekTo])

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const time = video.currentTime
    setPosition(time)

    if (timeline) {
      const word = timeline.words.find(
        (item) => time >= item.source_start - 0.02 && time <= item.source_end + 0.02,
      )
      onActiveWord(word ? word.id : null)
    }

    if (!followCut || !timeline || timeline.clips.length === 0) return
    const inside = timeline.clips.find(
      (clip) => time >= clip.source_start - 0.01 && time <= clip.source_end,
    )
    if (inside) return
    const next = timeline.clips.find((clip) => clip.source_start > time)
    if (next) {
      video.currentTime = next.source_start
    } else if (!video.paused) {
      video.pause()
    }
  }, [followCut, onActiveWord, timeline])

  const duration = timeline?.source_duration || videoRef.current?.duration || 0
  const percent = (value: number) => (duration > 0 ? (value / duration) * 100 : 0)

  return (
    <div className="panel">
      <h2>Kəsimlərin ilkin baxışı</h2>
      <p className="small muted">Orijinal görüntü göstərilir. B-roll, subtitr, crop və səs keçidlərini hazır exportda yoxlayın.</p>
      <div className="player-wrap">
        <video
          ref={videoRef}
          src={src}
          controls
          playsInline
          preload="metadata"
          onTimeUpdate={handleTimeUpdate}
          onSeeked={handleTimeUpdate}
        />
      </div>

      <div className="timeline-bar" title="Green = kept, red = removed, teal = B-roll">
        {timeline?.clips.map((clip) => (
          <div
            key={clip.index}
            className="keep"
            style={{
              left: `${percent(clip.source_start)}%`,
              width: `${Math.max(0.3, percent(clip.source_end - clip.source_start))}%`,
            }}
          />
        ))}
        {timeline?.broll.map((overlay) => {
          const clip = timeline.clips.find(
            (item) =>
              overlay.output_start >= item.output_start && overlay.output_start <= item.output_end,
          )
          const sourceStart = clip
            ? clip.source_start + (overlay.output_start - clip.output_start)
            : 0
          return (
            <div
              key={overlay.suggestion_id}
              className="broll"
              style={{
                left: `${percent(sourceStart)}%`,
                width: `${Math.max(0.3, percent(overlay.output_end - overlay.output_start))}%`,
              }}
            />
          )
        })}
        <div className="cursor" style={{ left: `${percent(position)}%` }} />
      </div>

      <div className="legend">
        <span><i style={{ background: '#2f4a3c' }} />kept</span>
        <span><i style={{ background: '#2a1c1f' }} />removed</span>
        <span><i style={{ background: 'var(--accent-2)' }} />B-roll</span>
        <span className="mono" style={{ marginLeft: 'auto' }}>
          {formatTime(position)} / {formatTime(duration)}
        </span>
      </div>

      <label className="switch" style={{ marginTop: '0.6rem' }}>
        <input
          type="checkbox"
          checked={followCut}
          onChange={(event) => setFollowCut(event.target.checked)}
        />
        Skip removed regions during playback
      </label>
    </div>
  )
}
