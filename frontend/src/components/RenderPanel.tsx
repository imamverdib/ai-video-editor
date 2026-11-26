import { useState } from 'react'
import { api, formatTime } from '../api'
import type { Capabilities, Job, RenderOptions, Timeline } from '../types'

interface Props {
  projectId: string
  timeline: Timeline | null
  capabilities: Capabilities | null
  job: Job | null
  outputFilename: string | null
  onRender: (options: RenderOptions) => void
  busy: boolean
  stale: boolean
  onCancel: () => void
}

const DEFAULTS: RenderOptions = {
  profile: 'quality', framing: 'crop', fps: 'source', crop_x: .5, hdr_policy: 'tonemap',
  burn_subtitles: true,
  include_broll: true,
  subtitle_position: 'lower_third',
  subtitle_highlight_color: '#FFE94D',
  subtitle_font_size: 112,
  subtitle_uppercase: false,
  subtitle_engine: null,
  cinematic_effect: false,
}

export default function RenderPanel({
  projectId,
  timeline,
  capabilities,
  job,
  outputFilename,
  onRender,
  busy,
  stale,
  onCancel,
}: Props) {
  const [options, setOptions] = useState<RenderOptions>(DEFAULTS)
  const set = <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) =>
    setOptions((prev) => ({ ...prev, [key]: value }))

  const running = job?.status === 'queued' || job?.status === 'running'
  const finished = job?.status === 'succeeded'
  const resultName = job?.output_filename ?? outputFilename

  return (
    <div className="panel">
      <h2>Render</h2>
      <div className="field">
        <label htmlFor="export-profile">Export keyfiyyəti</label>
        <select id="export-profile" value={options.profile} onChange={e => set('profile', e.target.value as RenderOptions['profile'])}>
          <option value="quality">Keyfiyyətli · 1080p · CRF 18 / medium</option>
          <option value="draft">Sürətli baxış · 1080p · CRF 20 / veryfast</option>
          <option value="master">Master · mənbə ölçüsü · CRF 16 / slow · SDR</option>
          <option value="4k">4K (2160p) · Yüksək keyfiyyət</option>
        </select>
      </div>
      <div className="field">
        <label htmlFor="framing">Kadrlaşdırma</label>
        <select id="framing" value={options.framing} disabled={options.profile === 'master'} onChange={e => set('framing', e.target.value as RenderOptions['framing'])}>
          <option value="crop">9:16 doldur — kənarları kəs</option>
          <option value="fit">Bütün görüntünü saxla — boş kənarlar əlavə et</option>
          <option value="source">Mənbənin ölçüsünü saxla</option>
        </select>
        {options.framing === 'crop' && options.profile !== 'master' && <>
          <label htmlFor="crop-x">Üfüqi kəsim mövqeyi</label>
          <input id="crop-x" type="range" min="0" max="1" step=".05" value={options.crop_x} onChange={e => set('crop_x', Number(e.target.value))} />
        </>}
      </div>
      <div className="field">
        <label htmlFor="export-fps">Kadr tezliyi</label>
        <select id="export-fps" value={options.fps} onChange={e => set('fps', e.target.value as RenderOptions['fps'])}>
          <option value="source">Mənbəyə uyğun (VFR üçün orta FPS)</option>
          <option value="30">30 FPS</option><option value="60">60 FPS</option>
        </select>
      </div>
      <div className="field">
        <label htmlFor="hdr-policy">HDR mənbə</label>
        <select id="hdr-policy" value={options.hdr_policy} onChange={e => set('hdr_policy', e.target.value as RenderOptions['hdr_policy'])}>
          <option value="tonemap">Rəngləri SDR-ə uyğunlaşdır (tone mapping)</option>
          <option value="reject">HDR olduqda dayandır</option>
        </select>
      </div>
      <p className="small muted">Subtitr və kəsimlər üçün video yenidən kodlanır. Orijinal fayl saxlanılır.</p>
      {stale && <div className="banner warn small">Son export əvvəlki redaktəyə aiddir. Cari dəyişikliklər üçün yenidən render edin.</div>}

      <div className="stats" style={{ marginBottom: '0.8rem' }}>
        <div className="stat">
          <b className="mono">{formatTime(timeline?.output_duration ?? 0)}</b>
          <span>final length</span>
        </div>
        <div className="stat">
          <b className="mono">−{formatTime(timeline?.removed_duration ?? 0)}</b>
          <span>removed</span>
        </div>
        <div className="stat">
          <b className="mono">{timeline?.clips.length ?? 0}</b>
          <span>cuts</span>
        </div>
        <div className="stat">
          <b className="mono">{timeline?.broll.length ?? 0}</b>
          <span>B-roll</span>
        </div>
      </div>

      <label className="switch">
        <input
          type="checkbox"
          checked={options.burn_subtitles}
          onChange={(event) => set('burn_subtitles', event.target.checked)}
        />
        Burn kinetic subtitles
      </label>
      <label className="switch">
        <input
          type="checkbox"
          checked={options.include_broll}
          onChange={(event) => set('include_broll', event.target.checked)}
        />
        Include accepted B-roll
      </label>
      <label className="switch">
        <input
          type="checkbox"
          checked={options.subtitle_uppercase === true}
          onChange={(event) => set('subtitle_uppercase', event.target.checked)}
        />
        UPPERCASE captions
      </label>
      <label className="switch">
        <input
          type="checkbox"
          checked={options.cinematic_effect}
          onChange={(event) => set('cinematic_effect', event.target.checked)}
        />
        Cinematik effekt (Lumetri Color)
      </label>

      <div className="field">
        <label htmlFor="subtitle_position">Caption placement</label>
        <select
          id="subtitle_position"
          value={options.subtitle_position ?? 'lower_third'}
          onChange={(event) =>
            set('subtitle_position', event.target.value as 'center' | 'lower_third')
          }
        >
          <option value="lower_third">Lower third (safe for Reels UI)</option>
          <option value="center">Dead centre</option>
        </select>
      </div>

      {options.burn_subtitles && (
        <div className="field">
          <label htmlFor="subtitle_font_size">
            Subtitle size: <span className="mono">{options.subtitle_font_size ?? 112}px</span>
          </label>
          <div className="row" style={{ alignItems: 'center', gap: '0.6rem' }}>
            <input
              id="subtitle_font_size"
              type="range"
              min="72"
              max="144"
              step="2"
              value={options.subtitle_font_size ?? 112}
              onChange={(event) => set('subtitle_font_size', Number(event.target.value))}
              style={{ flex: 1 }}
            />
            <button
              type="button"
              className="ghost tiny"
              title="Reset to 112px default"
              onClick={() => set('subtitle_font_size', 112)}
            >
              Reset
            </button>
          </div>
        </div>
      )}

      <div className="field">
        <label id="highlight-label">Active-word highlight</label>
        <div className="row" role="group" aria-labelledby="highlight-label">
          {['#FFE94D', '#5EEAD4', '#FF6B6B', '#A78BFA', '#FFFFFF'].map((color) => (
            <button
              key={color}
              className="tiny"
              title={color}
              style={{
                background: color,
                borderColor:
                  options.subtitle_highlight_color === color ? 'var(--text)' : 'transparent',
                width: 26,
                height: 26,
              }}
              onClick={() => set('subtitle_highlight_color', color)}
            />
          ))}
        </div>
      </div>

      {capabilities && !capabilities.has_libass && (
        <div className="banner info small">
          FFmpeg has no libass, so captions are composited with the built-in Pillow
          rasteriser (engine: <b>{capabilities.subtitle_engine}</b>).
        </div>
      )}

      <button
        className="primary"
        style={{ width: '100%', marginTop: '0.4rem' }}
        disabled={busy || running || !timeline}
        onClick={() => onRender(options)}
      >
        {running ? 'Rendering…' : 'Render Reel'}
      </button>
      {running && <button className="ghost" onClick={onCancel}>Renderi dayandır</button>}

      {job && (
        <div style={{ marginTop: '0.8rem' }}>
          <div className="progress">
            <i style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="row small muted" style={{ marginTop: '0.4rem' }}>
            {running && <span className="spinner" />}
            <span>{job.message || job.stage}</span>
            <span style={{ flex: 1 }} />
            <span className="mono">{Math.round(job.progress * 100)}%</span>
          </div>
          {job.error && (
            <div className="banner error small" style={{ marginTop: '0.6rem' }}>
              <strong>{job.error.message}</strong>
              {job.error.detail && <pre>{job.error.detail}</pre>}
            </div>
          )}
        </div>
      )}

      {(finished || !running) && resultName && (
        <div style={{ marginTop: '0.9rem' }}>
          <div className="player-wrap" style={{ maxHeight: '34vh' }}>
            <video src={api.outputUrl(projectId, resultName)} controls playsInline />
          </div>
          <a
            className="row small"
            style={{ marginTop: '0.6rem', justifyContent: 'center' }}
            href={api.downloadUrl(projectId, resultName)}
          >
            ⬇ Download {resultName}
          </a>
        </div>
      )}
    </div>
  )
}
