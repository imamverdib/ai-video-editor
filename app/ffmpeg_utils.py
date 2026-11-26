"""Thin, dependable wrapper around the ffmpeg/ffprobe CLIs.

Responsibilities:

* probe media and fail loudly (with actionable messages) on missing streams;
* run ffmpeg with machine-readable progress so the UI can show a real bar;
* detect build capabilities up-front (libass / drawtext / encoders), because a
  Homebrew ffmpeg without libass is common and must not surface as a cryptic
  "No such filter" halfway through a render;
* resolve a concrete bold font file for the subtitle renderers.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .config import settings
from .errors import (
    FFmpegError,
    MissingAudioStreamError,
    MissingVideoStreamError,
    UnsupportedMediaError,
)
from .runtime import check_cancelled
from .schemas import MediaInfo

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

FONT_SEARCH_DIRS = (
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path.home() / ".fonts",
    Path.home() / ".local" / "share" / "fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("C:/Windows/Fonts"),
)

#: Ordered fallbacks - the first one that exists *and* covers the target
#: alphabet wins. Heavy weights first: kinetic reel subtitles need a very bold
#: face to stay readable over busy footage.
#:
#: Impact and Arial Black are deliberately last: neither ships a glyph for `ə`
#: (U+0259), so every Azerbaijani schwa would render as a .notdef box.
FONT_FALLBACKS = (
    "Montserrat-ExtraBold",
    "Montserrat-Bold",
    "Montserrat-Black",
    "Anton-Regular",
    "Inter-Bold",
    "Roboto-Bold",
    "NotoSans-Bold",
    "DejaVuSans-Bold",
    "DejaVuLGCSans-Bold",
    "Arial Bold",
    "Arial Unicode",
    "Helvetica",
    "Impact",
    "Arial Black",
)

#: Characters a subtitle font must really have for Azerbaijani.
AZERBAIJANI_TEST_CHARS = "əğışçöü"


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    return settings.resolve_binary(settings.ffmpeg_binary)


@functools.lru_cache(maxsize=1)
def ffprobe_path() -> str:
    return settings.resolve_binary(settings.ffprobe_binary)


# --------------------------------------------------------------- capabilities
@dataclass(frozen=True)
class FFmpegCapabilities:
    version: str
    filters: frozenset[str]
    encoders: frozenset[str]
    warnings: tuple[str, ...] = field(default=())

    @property
    def has_libass(self) -> bool:
        return "ass" in self.filters and "subtitles" in self.filters

    @property
    def has_drawtext(self) -> bool:
        return "drawtext" in self.filters

    @property
    def has_alpha_overlay_encoder(self) -> bool:
        return "libvpx-vp9" in self.encoders

    def require_filter(self, name: str) -> None:
        if name not in self.filters:
            from .errors import FFmpegCapabilityError

            raise FFmpegCapabilityError(
                f"This FFmpeg build has no '{name}' filter.",
                detail="Rebuild/reinstall FFmpeg with the required libraries.",
            )


def _parse_listing(output: str) -> set[str]:
    """Extract names from `ffmpeg -filters` / `-encoders` tabular output."""
    names: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Filters:", "Encoders:", "---", "=")):
            continue
        parts = stripped.split()
        # Format: "<flags> <name> <io> <description>"
        if len(parts) >= 2 and not parts[0].startswith("-"):
            names.add(parts[1])
    return names


@functools.lru_cache(maxsize=1)
def capabilities() -> FFmpegCapabilities:
    binary = ffmpeg_path()
    version_out = _capture([binary, "-hide_banner", "-version"])
    version = version_out.splitlines()[0] if version_out else "unknown"
    filters = _parse_listing(_capture([binary, "-hide_banner", "-filters"]))
    encoders = _parse_listing(_capture([binary, "-hide_banner", "-encoders"]))

    warnings: list[str] = []
    if "ass" not in filters:
        warnings.append(
            "FFmpeg was built without libass - falling back to the Pillow "
            "subtitle rasteriser. For the .ass karaoke path install a build "
            "with libass (macOS: `brew reinstall ffmpeg`)."
        )
    if "libx264" not in encoders:
        warnings.append("libx264 encoder missing; H.264 output will fail.")
    return FFmpegCapabilities(
        version=version,
        filters=frozenset(filters),
        encoders=frozenset(encoders),
        warnings=tuple(warnings),
    )


def active_subtitle_engine(requested: str | None = None) -> str:
    """Resolve the subtitle engine to use: ``"ass"`` or ``"pillow"``."""
    choice = (requested or settings.subtitle_engine or "auto").lower()
    caps = capabilities()
    if choice == "ass":
        caps.require_filter("ass")
        return "ass"
    if choice == "pillow":
        return "pillow"
    return "ass" if caps.has_libass else "pillow"


# ----------------------------------------------------------------------- font
def font_covers(path: Path, chars: str = AZERBAIJANI_TEST_CHARS) -> bool:
    """True when ``path`` has a real glyph for every character in ``chars``.

    Implemented with Pillow (already a dependency) rather than fontTools:
    render each character and compare the bitmap against the ``.notdef`` glyph
    that a private-use codepoint always produces. Cross-checked against
    fontTools' cmap tables - identical verdicts, including the Impact and Arial
    Black families that silently lack `ə`.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover - Pillow is a hard requirement
        return True  # cannot verify, so do not block the font

    try:
        font = ImageFont.truetype(str(path), 48)
    except OSError:
        return False

    def render(char: str) -> bytes:
        canvas = Image.new("L", (90, 90), 0)
        ImageDraw.Draw(canvas).text((6, 6), char, font=font, fill=255)
        return canvas.tobytes()

    try:
        notdef = render("\ue000")  # private use area -> guaranteed .notdef
        blank = render(" ")
        return all(render(char) not in (notdef, blank) for char in chars)
    except Exception:  # noqa: BLE001 - an unrenderable font is simply unusable
        return True


@functools.lru_cache(maxsize=8)
def resolve_font_file(family: str | None = None, explicit: str | None = None) -> Path | None:
    """Find a usable bold font that also covers the subtitle alphabet.

    An explicit override always wins (the user knows what they want), but a
    warning is logged when it cannot render Azerbaijani.
    """
    explicit = explicit if explicit is not None else settings.subtitle_font_file
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            if not font_covers(candidate):
                log.warning(
                    "SUBTITLE_FONT_FILE=%s has no glyph for some Azerbaijani "
                    "letters (%s); captions will show empty boxes.",
                    candidate.name,
                    AZERBAIJANI_TEST_CHARS,
                )
            return candidate
        log.warning("SUBTITLE_FONT_FILE=%s does not exist; auto-discovering.", explicit)

    family = family or settings.subtitle_font_family
    wanted = [family.replace(" ", "-"), family.replace(" ", ""), *FONT_FALLBACKS]

    available: list[Path] = []
    for directory in FONT_SEARCH_DIRS:
        if directory.is_dir():
            for suffix in ("*.ttf", "*.otf", "*.ttc"):
                available.extend(directory.glob(suffix))

    def normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    index = {normalise(path.stem): path for path in available}

    #: Best font that exists but cannot render the alphabet - used only if
    #: nothing better turns up.
    incomplete: Path | None = None

    for name in wanted:
        hit = index.get(normalise(name))
        if hit is None:
            continue
        if font_covers(hit):
            return hit
        if incomplete is None:
            incomplete = hit
            log.info("Font '%s' lacks Azerbaijani glyphs; continuing the search.", hit.name)

    # Nothing on the preferred list fits: take any bold face that covers the
    # alphabet, then any covering face at all.
    heavy = [p for p in available if {"bold", "black"} & set(p.stem.lower().split("-"))]
    for pool in (heavy, available):
        for path in pool:
            if font_covers(path):
                return path

    if incomplete is not None:
        log.warning(
            "No installed font covers %s; falling back to '%s', which will show "
            "empty boxes for the missing letters. Install Montserrat or set "
            "SUBTITLE_FONT_FILE.",
            AZERBAIJANI_TEST_CHARS,
            incomplete.name,
        )
        return incomplete
    return available[0] if available else None


# ---------------------------------------------------------------------- probe
def _capture(command: Sequence[str], *, timeout: float = 60.0) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed binary, argv list
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - guarded by resolve_binary
        raise FFmpegError(f"Cannot execute {command[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{command[0]} timed out after {timeout:.0f}s") from exc
    return (result.stdout or "") + (result.stderr or "")


def _probe_json(path: Path) -> dict:
    command = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedMediaError("ffprobe timed out after 30 seconds.") from exc

    if result.returncode != 0:
        raise UnsupportedMediaError(
            "Could not read the uploaded file as media.",
            detail=(result.stderr or "").strip()[-800:],
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("ffprobe returned malformed JSON.") from exc


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if raw and raw != "0/0":
            try:
                value = float(Fraction(raw))
            except (ZeroDivisionError, ValueError):
                continue
            if value > 0:
                return round(value, 4)
    return float(settings.output_fps)


def _rotation(stream: dict) -> int:
    for entry in stream.get("side_data_list", []) or []:
        if "rotation" in entry:
            try:
                return int(float(entry["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    tag = (stream.get("tags") or {}).get("rotate")
    if tag:
        try:
            return int(float(tag)) % 360
        except (TypeError, ValueError):
            pass
    return 0


def probe_media(path: Path, *, require_audio: bool = True) -> MediaInfo:
    """Inspect a media file, raising precise errors for missing streams."""
    if not path.is_file():
        raise UnsupportedMediaError(f"File not found: {path.name}")

    data = _probe_json(path)
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise MissingVideoStreamError(
            "The uploaded file contains no video stream.",
            detail="Upload a vertical (9:16) MP4/MOV that has both video and audio.",
        )
    if audio is None and require_audio:
        raise MissingAudioStreamError(
            "The uploaded video has no audio stream, so there is nothing to transcribe.",
            detail="Export your clip with an audio track and upload it again.",
        )

    fmt = data.get("format", {}) or {}
    duration = 0.0
    for candidate in (video.get("duration"), fmt.get("duration")):
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break
    if duration <= 0:
        raise UnsupportedMediaError(
            "Could not determine the video duration.",
            detail="The file may be truncated or still uploading.",
        )

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width
    if width <= 0 or height <= 0:
        raise UnsupportedMediaError("Video stream reports invalid dimensions.")

    try:
        size_bytes = int(fmt.get("size") or path.stat().st_size)
    except (TypeError, ValueError):
        size_bytes = path.stat().st_size

    return MediaInfo(
        duration=duration,
        width=width,
        height=height,
        fps=_fps(video),
        has_audio=audio is not None,
        video_codec=video.get("codec_name"),
        audio_codec=(audio or {}).get("codec_name"),
        audio_channels=(audio or {}).get("channels"),
        audio_sample_rate=int((audio or {}).get("sample_rate") or 0) or None,
        size_bytes=size_bytes,
        rotation=rotation,
        pixel_format=video.get("pix_fmt"),
        color_transfer=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        color_space=video.get("color_space"),
        color_range=video.get("color_range"),
        video_start=float(video.get("start_time") or 0),
        audio_start=float((audio or {}).get("start_time") or 0),
        audio_duration=float(audio["duration"]) if audio and audio.get("duration") else None,
        frame_rate=video.get("avg_frame_rate"),
    )


# ------------------------------------------------------------------- run/exec
_PROGRESS_TIME = re.compile(r"^out_time_(?:ms|us)=(\d+)$")


def run_ffmpeg(
    args: Sequence[str],
    *,
    total_duration: float | None = None,
    on_progress: ProgressCallback | None = None,
    log_tail: list[str] | None = None,
    timeout: float | None = None,
) -> None:
    """Run ffmpeg, streaming `-progress` output into ``on_progress`` (0..1)."""
    command = [ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    command += ["-progress", "pipe:1", "-nostats"]
    command += [str(a) for a in args]
    log.debug("ffmpeg %s", " ".join(command[1:]))

    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        lines_queue: queue.Queue = queue.Queue()

        def read_progress():
            try:
                for line in process.stdout:
                    lines_queue.put(line)
            finally:
                lines_queue.put(None)

        reader = threading.Thread(target=read_progress, daemon=True)
        reader.start()
        deadline = time.monotonic() + (timeout or settings.render_timeout_s)
        try:
            while True:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise FFmpegError("FFmpeg işinin vaxt limiti bitdi.")
                try:
                    line = lines_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                match = _PROGRESS_TIME.match(line.strip())
                if match and on_progress and total_duration and total_duration > 0:
                    seconds = int(match.group(1)) / 1_000_000.0
                    on_progress(max(0.0, min(1.0, seconds / total_duration)))
            returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise FFmpegError("FFmpeg işinin vaxt limiti bitdi.") from None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            reader.join(timeout=1)
            process.stdout.close()

        err_file.seek(0)
        stderr = err_file.read()

    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    if log_tail is not None:
        log_tail.extend(lines[-40:])
    if returncode != 0:
        raise FFmpegError(
            "FFmpeg failed while processing the video.",
            detail="\n".join(lines[-25:]) or f"exit code {returncode}",
        )
    if on_progress:
        on_progress(1.0)
