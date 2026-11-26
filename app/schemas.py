"""Pydantic models shared by the pipeline, the persistence layer and the API.

Three distinct time bases are used throughout the app and every field name
states which one it belongs to:

* ``source_*``  - seconds on the original uploaded video's timeline.
* ``compact_*`` - seconds on the VAD-compacted (silence-stripped) audio that is
  sent to Groq Whisper. Only used inside the transcription step.
* ``output_*``  - seconds on the final rendered timeline.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# --------------------------------------------------------------------- media
class MediaInfo(_Model):
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
    size_bytes: int = 0
    rotation: int = 0
    pixel_format: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    video_start: float = 0
    audio_start: float = 0
    frame_rate: str | None = None
    audio_duration: float | None = None

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        return self.height >= self.width


# ----------------------------------------------------------------------- VAD
class SpeechSegment(_Model):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# --------------------------------------------------------------- transcript
class Word(_Model):
    id: int
    text: str
    start: float = Field(description="Source-timeline start, seconds")
    end: float = Field(description="Source-timeline end, seconds")
    sentence_id: int
    #: Original ASR spelling, kept when the normaliser rewrote the token or user edited.
    raw_text: str | None = None
    #: AI suggested spelling or grammar correction.
    suggestion: str | None = None
    suggestion_reason: str | None = None


class Sentence(_Model):
    id: int
    word_ids: list[int]
    text: str
    start: float
    end: float


class Transcript(_Model):
    language: str
    text: str
    words: list[Word]
    sentences: list[Sentence]
    normalization_applied: bool = False
    normalized_word_count: int = 0
    asr_chunks: int = 0
    suggestions_count: int = 0
    ai_warnings: list[str] = Field(default_factory=list)
    ai_provider: str | None = None
    ai_model: str | None = None


# -------------------------------------------------------------------- b-roll
class BrollStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    dismissed = "dismissed"


class BrollClip(_Model):
    id: str
    provider: str = "pexels"
    provider_id: str
    width: int
    height: int
    duration: float
    download_url: str
    preview_image: str | None = None
    author: str | None = None
    author_url: str | None = None
    page_url: str | None = None
    query: str = ""

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


class BrollSuggestion(_Model):
    id: str
    sentence_id: int
    anchor_word_id: int
    source_start: float
    source_end: float
    reason: str = ""
    query: str = ""
    keywords: list[str] = Field(default_factory=list)
    status: BrollStatus = BrollStatus.pending
    candidates: list[BrollClip] = Field(default_factory=list)
    selected_clip_id: str | None = None
    search_error: str | None = None

    def selected_clip(self) -> BrollClip | None:
        if not self.selected_clip_id:
            return self.candidates[0] if self.candidates else None
        for clip in self.candidates:
            if clip.id == self.selected_clip_id:
                return clip
        return None


# ---------------------------------------------------------------- edit state
class EditState(_Model):
    removed_word_ids: list[int] = Field(default_factory=list)
    #: Also cut the long non-speech gaps that Silero VAD detected.
    trim_silences: bool = True


# ------------------------------------------------------------------ timeline
class TimelineClip(_Model):
    index: int
    source_start: float
    source_end: float
    output_start: float
    output_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)


class TimelineWord(_Model):
    id: int
    text: str
    sentence_id: int
    source_start: float
    source_end: float
    output_start: float
    output_end: float


class BrollOverlay(_Model):
    suggestion_id: str
    clip: BrollClip
    output_start: float
    output_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.output_end - self.output_start)


class Timeline(_Model):
    clips: list[TimelineClip]
    words: list[TimelineWord]
    broll: list[BrollOverlay] = Field(default_factory=list)
    output_duration: float = 0.0
    source_duration: float = 0.0
    removed_duration: float = 0.0
    kept_word_count: int = 0
    removed_word_count: int = 0
    strict_deletions: bool = False
    source_fps: float | None = None


# -------------------------------------------------------------------- render
class RenderOptions(_Model):
    profile: Literal["quality", "draft", "master", "4k"] = "quality"
    framing: Literal["crop", "fit", "source"] = "crop"
    fps: Literal["source", "30", "60"] = "source"
    crop_x: float = Field(default=0.5, ge=0, le=1)
    hdr_policy: Literal["tonemap", "reject"] = "tonemap"
    burn_subtitles: bool = True
    include_broll: bool = True
    subtitle_position: Literal["center", "lower_third"] | None = None
    subtitle_highlight_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    subtitle_font_size: int | None = Field(default=None, ge=24, le=200)
    subtitle_uppercase: bool | None = None
    #: Force a specific subtitle engine ("ass" | "pillow"); None = auto.
    subtitle_engine: Literal["ass", "pillow", "auto"] | None = None
    cinematic_effect: bool = False


# ---------------------------------------------------------------- social
class SocialPlatform(str, Enum):
    instagram = "instagram"
    youtube = "youtube"
    linkedin = "linkedin"


class SocialPostStatus(str, Enum):
    draft = "draft"
    scheduled = "scheduled"
    published = "published"
    failed = "failed"


class SocialPost(_Model):
    platform: SocialPlatform
    text: str = ""
    status: SocialPostStatus = SocialPostStatus.draft
    schedule_time: float | None = None
    publish_job_id: str | None = None
    error: str | None = None


# ----------------------------------------------------------------- project
class ProjectStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class PipelineStage(str, Enum):
    queued = "queued"
    probing = "probing"
    extracting_audio = "extracting_audio"
    detecting_speech = "detecting_speech"
    transcribing = "transcribing"
    normalizing = "normalizing"
    suggesting_broll = "suggesting_broll"
    done = "done"
    failed = "failed"


class Project(_Model):
    id: str
    original_filename: str
    source_file: str = "source.mp4"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    status: ProjectStatus = ProjectStatus.uploaded
    stage: PipelineStage = PipelineStage.queued
    progress: float = 0.0
    message: str = ""
    error: dict[str, object] | None = None
    warnings: list[str] = Field(default_factory=list)

    media: MediaInfo | None = None
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    speech_duration: float = 0.0
    transcript: Transcript | None = None
    broll: list[BrollSuggestion] = Field(default_factory=list)
    edit: EditState = Field(default_factory=EditState)
    social_posts: dict[SocialPlatform, SocialPost] = Field(default_factory=dict)
    social_cover_image: str | None = None

    last_render_job_id: str | None = None
    output_filename: str | None = None
    revision: int = 0
    output_revision: int | None = None
    edit_clients: dict[str, int] = Field(default_factory=dict, exclude=True)
    undo_history: list[dict] = Field(default_factory=list, exclude=True)
    redo_history: list[dict] = Field(default_factory=list, exclude=True)

    @computed_field
    @property
    def can_undo(self) -> bool:
        return bool(self.undo_history)

    @computed_field
    @property
    def can_redo(self) -> bool:
        return bool(self.redo_history)


# --------------------------------------------------------------------- jobs
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class Job(_Model):
    id: str
    project_id: str
    kind: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    stage: str = "queued"
    message: str = ""
    error: dict[str, object] | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    output_filename: str | None = None
    log_tail: list[str] = Field(default_factory=list)


# ----------------------------------------------------------- API request DTOs
class WordUpdateRequest(_Model):
    text: str = Field(min_length=1, max_length=200, description="The corrected word text")


class MergeWordsRequest(_Model):
    word_ids: list[int] = Field(min_length=2, max_length=15, description="Word IDs to merge")
    text: str | None = Field(default=None, max_length=200, description="Custom merged text")


class EditRequest(_Model):
    removed_word_ids: list[int] = Field(default_factory=list)
    trim_silences: bool | None = None
    word_edits: dict[int, str] | None = None
    client_id: str | None = Field(default=None, max_length=80)
    sequence: int | None = Field(default=None, ge=0)
    expected_revision: int | None = Field(default=None, ge=0)


class BrollSearchRequest(_Model):
    query: str = Field(min_length=1, max_length=200)


class BrollDecisionRequest(_Model):
    status: BrollStatus
    clip_id: str | None = None


class RenderRequest(_Model):
    removed_word_ids: list[int] | None = None
    options: RenderOptions = Field(default_factory=RenderOptions)


class CapabilitiesResponse(_Model):
    ffmpeg_version: str
    has_libass: bool
    has_drawtext: bool
    subtitle_engine: str
    groq_configured: bool
    pexels_configured: bool
    normalizer_provider: str
    normalizer_model: str | None = None
    agy_available: bool = False
    font_file: str | None = None
    #: False when the chosen font cannot render every Azerbaijani letter.
    font_covers_language: bool = True
    warnings: list[str] = Field(default_factory=list)
