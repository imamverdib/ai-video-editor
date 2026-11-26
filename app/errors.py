"""Domain-level exceptions.

Every exception carries an HTTP status code and a stable machine-readable
``code`` so the frontend can react (e.g. show a retry button on rate limits)
without string-matching on messages.
"""

from __future__ import annotations


class EditorError(Exception):
    """Base class for all recoverable, user-facing application errors."""

    status_code: int = 400
    code: str = "editor_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ConfigurationError(EditorError):
    status_code = 500
    code = "configuration_error"


class UnsupportedMediaError(EditorError):
    status_code = 415
    code = "unsupported_media"


class MissingVideoStreamError(EditorError):
    status_code = 422
    code = "missing_video_stream"


class MissingAudioStreamError(EditorError):
    status_code = 422
    code = "missing_audio_stream"


class NoSpeechDetectedError(EditorError):
    status_code = 422
    code = "no_speech_detected"


class FFmpegError(EditorError):
    status_code = 500
    code = "ffmpeg_failed"


class FFmpegCapabilityError(EditorError):
    status_code = 500
    code = "ffmpeg_capability_missing"


class UpstreamAPIError(EditorError):
    """A third-party API (Groq / Pexels / Anthropic) returned an error."""

    status_code = 502
    code = "upstream_api_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        detail: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message, detail=detail)
        self.provider = provider
        self.status = status

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["provider"] = self.provider
        if self.status is not None:
            payload["upstream_status"] = self.status
        return payload


class RateLimitedError(UpstreamAPIError):
    status_code = 429
    code = "upstream_rate_limited"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retry_after: float | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, detail=detail, status=429)
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        if self.retry_after is not None:
            payload["retry_after"] = self.retry_after
        return payload


class ProjectNotFoundError(EditorError):
    status_code = 404
    code = "project_not_found"


class BrollNotFoundError(EditorError):
    status_code = 404
    code = "broll_not_found"


class JobNotFoundError(EditorError):
    status_code = 404
    code = "job_not_found"


class ProjectNotReadyError(EditorError):
    status_code = 409
    code = "project_not_ready"


class InvalidEditError(EditorError):
    status_code = 400
    code = "invalid_edit"
