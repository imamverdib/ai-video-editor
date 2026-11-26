"""Central configuration, loaded from environment / ``.env``.

All tunables live here so no module hardcodes a magic number. Values are
validated once at import time and reused as a cached singleton.
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ASR_PROMPT = (
    "Azərbaycan dilində nitq, texnologiya və marketinq terminləri: "
    "AI, B-roll, hook, storytelling, funnel, retention, framework, workflow, "
    "prompt engineering, SaaS, reel, script."
)

# English terms that must keep their English spelling when the speaker uses them.
DEFAULT_GLOSSARY = [
    "AI",
    "B-roll",
    "hook",
    "storytelling",
    "funnel",
    "retention",
    "framework",
    "workflow",
    "prompt engineering",
    "SaaS",
    "reel",
    "script",
    "content",
    "engagement",
    "Instagram",
    "Reels",
    "TikTok",
    "YouTube",
    "CTA",
    "ROI",
    "KPI",
    "landing page",
    "copywriting",
    "lead",
    "conversion",
    "onboarding",
    "thumbnail",
    "caption",
    "carousel",
]

#: Loanwords whose *Azerbaijani* orthography is already correct. Without this
#: list the normaliser helpfully "fixes" them into English (alqoritm ->
#: algoritm, marketinq -> marketing), which corrupts a correct transcript.
#: Discovered by benchmarking the real prompt on Azerbaijani ASR output.
AZ_PRESERVE = [
    "kontent",
    "marketinq",
    "alqoritm",
    "auditoriya",
    "brend",
    "məzmun",
    "strategiya",
    "platforma",
    "kanal",
    "abunəçi",
    "izləyici",
    "reklam",
    "büdcə",
    "video",
    "montaj",
    "kadr",
    "musiqi",
    "trend",
    "layihə",
    "məhsul",
    "xidmət",
    "müştəri",
    "satış",
    "nəticə",
    "hesab",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- secrets
    groq_api_key: str = ""
    pexels_api_key: str = ""
    anthropic_api_key: str = ""

    # ------------------------------------------------------------------ paths
    data_dir: Path = PROJECT_ROOT / "data"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    # ------------------------------------------------------------------ HTTP
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 512
    #: Root logger level. Must live here, not os.getenv: pydantic-settings
    #: reads .env into this object without exporting it to os.environ.
    log_level: str = "INFO"

    # ------------------------------------------------------------------- VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 200
    vad_min_silence_ms: int = 180
    #: Padding added around every detected speech boundary (spec: 50-80 ms).
    vad_pad_ms: int = 65
    #: Segments closer than this after padding are merged into one.
    vad_merge_gap_ms: int = 120

    # ------------------------------------------------------------------- ASR
    groq_asr_model: str = "whisper-large-v3"
    asr_language: str = "az"
    asr_prompt: str = DEFAULT_ASR_PROMPT
    #: Max compacted-speech duration per Groq upload (Groq caps file size/length).
    asr_chunk_seconds: float = 420.0
    asr_max_retries: int = 5
    asr_temperature: float = 0.0
    #: Hard ceiling on one uploaded ASR file. Groq rejects anything over 25 MB.
    asr_max_upload_bytes: int = 24 * 1024 * 1024
    #: Wall-clock budget for all retries of a single ASR upload, in seconds.
    asr_retry_budget_s: float = 900.0
    #: Wall-clock budget for all retries of a single LLM call, in seconds.
    llm_retry_budget_s: float = 180.0

    # ------------------------------------------------------- LLM (normalise + b-roll)
    #: "gemini" | "groq" | "anthropic" | "off"
    normalizer_provider: str = "agy"
    agy_binary: str = "agy"
    agy_model: str = "gemini-3.8-flash-high"
    agy_timeout_s: float = Field(default=120, ge=5, le=600)
    agy_max_attempts: int = Field(default=2, ge=1, le=3)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"
    gemini_fallbacks: str = "gemini-2.5-flash"
    groq_llm_model: str = "qwen/qwen3.6-27b"
    groq_llm_fallbacks: str = "groq/compound,openai/gpt-oss-120b,openai/gpt-oss-20b"
    anthropic_model: str = "claude-sonnet-5"
    llm_max_retries: int = 4
    #: Words per terminology-normalisation request.
    normalizer_batch_size: int = 110

    # ---------------------------------------------------------------- B-roll
    broll_min_suggestions: int = 2
    broll_max_suggestions: int = 3
    broll_search_results: int = 8
    broll_default_duration: float = 3.0
    broll_max_duration: float = 5.0
    broll_download_max_mb: int = 120

    # -------------------------------------------------------------- timeline
    #: Extra head/tail kept around a run of kept words, in ms.
    cut_pad_ms: int = 40
    #: Gaps shorter than this are not worth a cut - keep them for natural flow.
    min_removable_gap_ms: int = 120
    #: Cuts shorter than this are dropped entirely (avoids 1-frame clips).
    min_clip_ms: int = 120
    #: Default for EditState.trim_silences: also cut the non-speech dead air
    #: that Silero VAD found inside surviving passages.
    trim_silences: bool = True
    #: Only silences at least this long are worth removing; shorter pauses are
    #: part of natural delivery.
    silence_trim_ms: int = 350
    #: Hard ceiling on the number of clips one edit may produce. Beyond this the
    #: filtergraph becomes pathological (one trim + one acrossfade per clip).
    max_clips: int = 400
    #: Above this clip count the serial acrossfade chain costs more than it is
    #: worth, so the plain per-clip-fade concat path is used instead.
    crossfade_max_clips: int = 150

    # ---------------------------------------------------------------- render
    output_width: int = Field(default=1080, ge=64, le=7680, multiple_of=2)
    output_height: int = Field(default=1920, ge=64, le=7680, multiple_of=2)
    output_fps: int = Field(default=30, ge=1, le=120)
    video_crf: int = Field(default=18, ge=0, le=51)
    video_preset: str = "medium"
    render_timeout_s: float = Field(default=1800, ge=10, le=14400)
    max_job_queue: int = Field(default=8, ge=1, le=32)
    max_source_duration_s: float = Field(default=1800, ge=10, le=7200)
    max_source_pixels: int = Field(default=35_000_000, ge=1_000_000)
    audio_bitrate: str = "192k"
    #: Fade applied to the head/tail of the whole audio track.
    audio_fade_ms: int = 12
    #: Audio dissolve length straddling every cut. The extra material is
    #: taken from *after* the cut point, so the timeline length is preserved
    #: exactly (see renderer._build_audio_chain).
    audio_crossfade_ms: int = 24
    #: "crossfade" (default) or "fade" (per-clip micro fades + hard concat).
    audio_join_mode: str = "crossfade"

    # ------------------------------------------------------------- subtitles
    #: "auto" picks `ass` when libass is available, else the Pillow rasteriser.
    subtitle_engine: str = "auto"
    subtitle_font_family: str = "Montserrat ExtraBold"
    #: Explicit .ttf/.otf path. Empty -> auto-discovered from system font dirs.
    subtitle_font_file: str = ""
    subtitle_font_size: int = 112
    subtitle_base_color: str = "#FFFFFF"
    subtitle_highlight_color: str = "#FFE94D"
    subtitle_outline_color: str = "#000000"
    subtitle_outline_width: int = 11
    subtitle_shadow_offset: int = 6
    #: "center" | "lower_third"
    subtitle_position: str = "lower_third"
    subtitle_max_words_per_line: int = 3
    subtitle_max_chars_per_line: int = 18
    #: Break a subtitle line when the pause between words exceeds this.
    subtitle_line_break_gap_ms: int = 550
    subtitle_uppercase: bool = False
    subtitle_highlight_scale: float = 1.12

    # ------------------------------------------------------------ validators
    @field_validator("data_dir", "frontend_dist", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("normalizer_provider", mode="after")
    @classmethod
    def _provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"agy", "gemini", "groq", "anthropic", "off"}:
            raise ValueError("normalizer_provider must be agy, gemini, groq, anthropic or off")
        return value

    @field_validator("audio_join_mode", mode="after")
    @classmethod
    def _join_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"crossfade", "fade"}:
            raise ValueError("audio_join_mode must be crossfade or fade")
        return value

    @field_validator("subtitle_engine", mode="after")
    @classmethod
    def _engine(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "ass", "pillow"}:
            raise ValueError("subtitle_engine must be auto, ass or pillow")
        return value

    @field_validator("subtitle_position", mode="after")
    @classmethod
    def _position(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"center", "lower_third"}:
            raise ValueError("subtitle_position must be center or lower_third")
        return value

    # ------------------------------------------------------------- helpers
    @property
    def groq_llm_chain(self) -> list[str]:
        """Primary model followed by de-duplicated fallbacks."""
        chain = [self.groq_llm_model.strip()]
        chain += [m.strip() for m in self.groq_llm_fallbacks.split(",")]
        seen: set[str] = set()
        result: list[str] = []
        for m in chain:
            if m and m not in seen:
                seen.add(m)
                result.append(m)
        return result

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.projects_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_binary(self, name: str) -> str:
        found = shutil.which(name)
        if not found:
            from .errors import ConfigurationError

            raise ConfigurationError(
                f"Required binary '{name}' was not found on PATH.",
                detail="Install FFmpeg (macOS: `brew install ffmpeg`) and retry.",
            )
        return found

    def glossary(self) -> list[str]:
        return list(DEFAULT_GLOSSARY)

    def az_preserve(self) -> list[str]:
        return list(AZ_PRESERVE)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
