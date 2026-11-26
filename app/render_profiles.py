"""Immutable per-render settings; concurrent renders never mutate globals."""

from dataclasses import dataclass

from .config import settings
from .errors import UnsupportedMediaError
from .schemas import MediaInfo, RenderOptions


@dataclass(frozen=True)
class RenderProfile:
    width: int
    height: int
    fps: float
    crf: int
    preset: str
    framing: str = "crop"
    crop_x: float = 0.5
    tonemap: bool = False

    @classmethod
    def resolve(cls, media: MediaInfo, options: RenderOptions):
        hdr = media.color_transfer in {"smpte2084", "arib-std-b67"}
        if hdr and options.hdr_policy == "reject":
            raise UnsupportedMediaError("HDR mənbə üçün SDR tone mapping seçin.")
        source_size = options.profile == "master" or options.framing == "source"
        if options.profile == "4k":
            width, height = 2160, 3840
        else:
            width, height = (
                (media.width, media.height)
                if source_size
                else (settings.output_width, settings.output_height)
            )
        fps = media.fps if options.fps == "source" else float(options.fps)
        if not 1 <= fps <= 120:
            raise UnsupportedMediaError("Dəstəklənən FPS aralığı 1–120-dir.")
        crf, preset = {
            "quality": (settings.video_crf, settings.video_preset),
            "draft": (20, "veryfast"),
            "master": (16, "slow"),
            "4k": (16, "medium"),
        }[options.profile]
        return cls(
            width - width % 2,
            height - height % 2,
            fps,
            crf,
            preset,
            "source" if source_size else options.framing,
            options.crop_x,
            hdr,
        )

    def scale_chain(self, *, alpha: bool = False) -> str:
        w, h = self.width, self.height
        if self.framing == "fit":
            return (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease:"
                f"force_divisible_by=2:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
            )
        if self.framing == "source" and not alpha:
            return f"scale={w}:{h}:flags=lanczos"
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase:"
            f"flags=lanczos,crop={w}:{h}:x=(iw-ow)*{self.crop_x}:y=(ih-oh)/2"
        )
