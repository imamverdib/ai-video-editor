/** Mirrors the Pydantic models in app/schemas.py. */

export interface MediaInfo {
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  video_codec: string | null
  audio_codec: string | null
  audio_channels?: number | null
  audio_sample_rate?: number | null
  size_bytes: number
  rotation?: number
  color_transfer?: string | null
}

export interface SpeechSegment { start: number; end: number }

export interface Word {
  id: number
  text: string
  start: number
  end: number
  sentence_id: number
  raw_text: string | null
  suggestion?: string | null
  suggestion_reason?: string | null
}

export interface Sentence {
  id: number
  word_ids: number[]
  text: string
  start: number
  end: number
}

export interface Transcript {
  language: string
  text: string
  words: Word[]
  sentences: Sentence[]
  normalization_applied: boolean
  normalized_word_count: number
  asr_chunks: number
  suggestions_count?: number
  ai_warnings?: string[]
  ai_model?: string | null
}

export type BrollStatus = 'pending' | 'accepted' | 'dismissed'

export interface BrollClip {
  id: string
  provider: string
  provider_id: string
  width: number
  height: number
  duration: number
  download_url: string
  preview_image: string | null
  author: string | null
  author_url: string | null
  page_url: string | null
  query: string
}

export interface BrollSuggestion {
  id: string
  sentence_id: number
  anchor_word_id: number
  source_start: number
  source_end: number
  reason: string
  query: string
  keywords: string[]
  status: BrollStatus
  candidates: BrollClip[]
  selected_clip_id: string | null
  search_error: string | null
}

export interface TimelineClip {
  index: number
  source_start: number
  source_end: number
  output_start: number
  output_end: number
}

export interface TimelineWord {
  id: number
  text: string
  sentence_id: number
  source_start: number
  source_end: number
  output_start: number
  output_end: number
}

export interface BrollOverlay {
  suggestion_id: string
  clip: BrollClip
  output_start: number
  output_end: number
}

export interface Timeline {
  clips: TimelineClip[]
  words: TimelineWord[]
  broll: BrollOverlay[]
  output_duration: number
  source_duration: number
  removed_duration: number
  kept_word_count: number
  removed_word_count: number
}

export type ProjectStatus = 'uploaded' | 'processing' | 'ready' | 'failed'

export interface ApiError {
  code: string
  message: string
  detail?: string
  provider?: string
  retry_after?: number
}

export interface SocialPost {
  platform: 'instagram' | 'youtube' | 'linkedin'
  text: string
  status: 'draft' | 'scheduled' | 'published' | 'failed'
  schedule_time: number | null
  publish_job_id: string | null
  error: string | null
}

export interface Project {
  revision: number
  output_revision: number | null
  can_undo: boolean
  can_redo: boolean
  id: string
  original_filename: string
  source_file: string
  created_at: number
  updated_at: number
  status: ProjectStatus
  stage: string
  progress: number
  message: string
  error: ApiError | null
  warnings: string[]
  media: MediaInfo | null
  speech_segments: SpeechSegment[]
  speech_duration: number
  transcript: Transcript | null
  broll: BrollSuggestion[]
  edit: { removed_word_ids: number[]; trim_silences: boolean }
  social_posts?: Record<string, SocialPost>
  social_cover_image?: string | null
  last_render_job_id: string | null
  output_filename: string | null
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface Job {
  id: string
  project_id: string
  kind: string
  status: JobStatus
  progress: number
  stage: string
  message: string
  error: ApiError | null
  created_at: number
  updated_at: number
  output_filename: string | null
  log_tail: string[]
}

export interface RenderOptions {
  profile: 'quality' | 'draft' | 'master' | '4k'
  framing: 'crop' | 'fit' | 'source'
  fps: 'source' | '30' | '60'
  crop_x: number
  hdr_policy: 'tonemap' | 'reject' 
  burn_subtitles: boolean
  include_broll: boolean
  subtitle_position: 'center' | 'lower_third' | null
  subtitle_highlight_color: string | null
  subtitle_font_size: number | null
  subtitle_uppercase: boolean | null
  subtitle_engine: string | null
  cinematic_effect: boolean
}

export interface Capabilities {
  ffmpeg_version: string
  has_libass: boolean
  has_drawtext: boolean
  subtitle_engine: string
  groq_configured: boolean
  pexels_configured: boolean
  normalizer_provider: string
  normalizer_model?: string | null
  agy_available?: boolean
  font_file: string | null
  font_covers_language: boolean
  warnings: string[]
}
