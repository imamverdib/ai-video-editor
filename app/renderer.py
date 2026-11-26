"""FFmpeg rendering engine: cut, concatenate, overlay B-roll, burn subtitles.

Everything happens in a **single** ffmpeg invocation so the output is encoded
exactly once (no generation loss from an intermediate file).

Audio joins
-----------
A hard cut in speech audio produces a click because the waveform jumps
mid-cycle. Two strategies are available:

``crossfade`` (default)
    Every clip's *audio* trim is extended by ``d`` seconds past its video cut
    point, and consecutive clips are joined with ``acrossfade=d``. Since
    ``len(A+d) + len(B) - d == len(A) + len(B)``, the audio stays sample-exactly
    as long as the video - so the subtitle timings computed by
    :mod:`app.timeline_manager` remain valid. The dissolve material is the real
    audio immediately after the cut, i.e. a classic audio dissolve.

``fade``
    A ``d``-millisecond fade in/out on each clip followed by a hard concat. Used
    automatically when a junction has no spare source material to dissolve
    across (e.g. a cut that ends exactly at the end of the file).
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .config import settings
from .errors import EditorError, FFmpegError
from .ffmpeg_utils import active_subtitle_engine, capabilities, probe_media, run_ffmpeg
from .llm_broll import download_clip
from .render_profiles import RenderProfile
from .schemas import RenderOptions, Timeline
from .subtitles import (
    SubtitleStyle,
    build_ass,
    render_raster_overlay,
    write_ass,
)

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class RenderResult:
    path: Path
    duration: float
    subtitle_engine: str | None
    broll_count: int
    audio_join_mode: str
    log_tail: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


#: Times printed into the filtergraph. Six decimals is enough to represent a
#: frame boundary of any realistic frame rate without rounding it *up* onto the
#: next frame (see timeline_manager._floor_time).
_TIME_FMT = ".6f"


# ------------------------------------------------------------------- escaping
def _escape_once(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _escape_filter_value(value: str) -> str:
    """Escape a value used inside a ``-filter_complex`` option (e.g. a path).

    FFmpeg unescapes such a value **twice**: once when the graph description is
    split into filters and options, and again inside the filter's own option
    parser. A single backslash therefore survives only the first pass, so a path
    containing an apostrophe or a colon (both legal on macOS) would reach the
    filter unquoted and break the graph. Escaping twice makes exactly one level
    survive each pass.
    """
    return _escape_once(_escape_once(value))


# --------------------------------------------------------------- video chain
def _build_video_chain(
    timeline: Timeline, profile: RenderProfile | None = None
) -> tuple[list[str], str]:
    """Trim + concat the surviving source ranges, then normalise to 9:16."""
    parts: list[str] = []
    labels: list[str] = []
    profile = profile or RenderProfile(
        settings.output_width,
        settings.output_height,
        settings.output_fps,
        settings.video_crf,
        settings.video_preset,
    )
    labels_in = "".join(f"[vs{i}]" for i in range(len(timeline.clips)))
    parts.append(
        f"[0:v]setpts=PTS-STARTPTS,fps={timeline.source_fps or profile.fps}:start_time=0,"
        f"split={len(timeline.clips)}{labels_in}"
    )
    for clip in timeline.clips:
        label = f"v{clip.index}"
        parts.append(
            f"[vs{clip.index}]trim=start={clip.source_start:{_TIME_FMT}}:"
            f"end={clip.source_end:{_TIME_FMT}},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")

    if len(labels) == 1:
        concat_out = labels[0].strip("[]")
    else:
        concat_out = "vcat"
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[{concat_out}]")

    tone = (
        "zscale=transfer=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,"
        "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=limited,"
        if profile.tonemap
        else ""
    )
    parts.append(
        f"[{concat_out}]{tone}{profile.scale_chain()},"
        f"setsar=1,fps={profile.fps},format=yuv420p[vbase]"
    )
    return parts, "vbase"


# --------------------------------------------------------------- audio chain
def _crossfade_durations(timeline: Timeline, source_duration: float) -> list[float] | None:
    """Per-junction dissolve length, or ``None`` if a plain concat is required."""
    if timeline.strict_deletions:
        return None
    requested = max(0.0, settings.audio_crossfade_ms / 1000.0)
    if requested <= 0 or len(timeline.clips) < 2:
        return None
    if len(timeline.clips) > settings.crossfade_max_clips:
        # One acrossfade per junction, chained serially: past a few hundred
        # junctions the graph costs far more than the dissolves are worth.
        log.info(
            "%d clips exceed CROSSFADE_MAX_CLIPS=%d; using per-clip fades.",
            len(timeline.clips),
            settings.crossfade_max_clips,
        )
        return None

    durations: list[float] = []
    for index in range(len(timeline.clips) - 1):
        current = timeline.clips[index]
        following = timeline.clips[index + 1]
        available = source_duration - current.source_end
        limit = min(
            requested,
            available,
            current.duration / 2.0,
            following.duration / 2.0,
        )
        if limit < 0.004:
            log.info(
                "Audio crossfade unavailable at junction %d (limit %.4fs); "
                "falling back to per-clip fades.",
                index,
                limit,
            )
            return None
        durations.append(round(limit, 4))
    return durations


def _build_audio_chain(
    timeline: Timeline,
    source_duration: float,
    source_offset: float = 0,
) -> tuple[list[str], str, str]:
    parts: list[str] = []
    clips = timeline.clips
    align = "asetpts=PTS-STARTPTS,"
    if source_offset > 0:
        align += f"adelay={source_offset * 1000:.6f}:all=1,"
    elif source_offset < 0:
        align += f"atrim=start={-source_offset:.6f},asetpts=PTS-STARTPTS,"
    align += f"apad=whole_dur={source_duration:.6f},"
    crossfades = _crossfade_durations(timeline, source_duration)
    mode = "crossfade" if (crossfades and settings.audio_join_mode == "crossfade") else "fade"
    if settings.audio_join_mode == "fade":
        crossfades = None

    if mode == "crossfade":
        assert crossfades is not None
        for index, clip in enumerate(clips):
            extend = crossfades[index] if index < len(crossfades) else 0.0
            end = min(source_duration, clip.source_end + extend)
            parts.append(
                f"[0:a]{align}atrim=start={clip.source_start:{_TIME_FMT}}:"
                f"end={end:{_TIME_FMT}},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        current = "a0"
        for index in range(1, len(clips)):
            nxt = f"ax{index}"
            parts.append(
                f"[{current}][a{index}]acrossfade=d={crossfades[index - 1]:.4f}"
                f":c1=tri:c2=tri[{nxt}]"
            )
            current = nxt
    else:
        fade = max(0.002, settings.audio_fade_ms / 1000.0)
        labels: list[str] = []
        for index, clip in enumerate(clips):
            span = clip.duration
            local = min(fade, max(0.002, span / 3.0))
            out_at = max(0.0, span - local)
            chain = [
                f"[0:a]{align}atrim=start={clip.source_start:{_TIME_FMT}}:"
                f"end={clip.source_end:{_TIME_FMT}}",
                "asetpts=PTS-STARTPTS",
            ]
            # The track-level pass below already fades the very start and the
            # very end of the concatenated result. Applying a per-clip envelope
            # there as well multiplies the two, squaring the intended linear
            # ramp and over-attenuating the first and last syllable.
            if index > 0:
                chain.append(f"afade=t=in:st=0:d={local:{_TIME_FMT}}")
            if index < len(clips) - 1:
                chain.append(f"afade=t=out:st={out_at:{_TIME_FMT}}:d={local:{_TIME_FMT}}")
            parts.append(",".join(chain) + f"[a{index}]")
            labels.append(f"[a{index}]")
        if len(labels) == 1:
            current = "a0"
        else:
            current = "acat"
            parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[{current}]")

    # Track-level head/tail fade + fixed output format.
    track_fade = max(0.002, settings.audio_fade_ms / 1000.0)
    total = max(track_fade * 2 + 0.01, timeline.output_duration)
    parts.append(
        f"[{current}]afade=t=in:st=0:d={track_fade:{_TIME_FMT}},"
        f"afade=t=out:st={total - track_fade:{_TIME_FMT}}:d={track_fade:{_TIME_FMT}},"
        f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
    )
    return parts, "aout", mode


# --------------------------------------------------------------- b-roll chain
def _build_broll_chain(
    timeline: Timeline,
    first_input_index: int,
    video_label: str,
    profile: RenderProfile | None = None,
) -> tuple[list[str], str]:
    parts: list[str] = []
    current = video_label
    profile = profile or RenderProfile(
        settings.output_width,
        settings.output_height,
        settings.output_fps,
        settings.video_crf,
        settings.video_preset,
    )
    fps = profile.fps

    for order, overlay in enumerate(timeline.broll):
        input_index = first_input_index + order
        duration = overlay.duration
        start = overlay.output_start
        prepared = f"br{order}"
        parts.append(
            f"[{input_index}:v]trim=start=0:end={duration:{_TIME_FMT}},setpts=PTS-STARTPTS,"
            f"{profile.scale_chain(alpha=True)},setsar=1,fps={fps},format=yuva420p,"
            f"tpad=start_duration={start:{_TIME_FMT}}:start_mode=add:color=black@0[{prepared}]"
        )
        nxt = f"vb{order}"
        parts.append(
            f"[{current}][{prepared}]overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
            f"format=auto:enable='between(t\\,{start:{_TIME_FMT}}\\,"
            f"{start + duration:{_TIME_FMT}})'[{nxt}]"
        )
        current = nxt
    return parts, current


_VERSION_NUMBER = re.compile(r"version n?(\d+)\.(\d+)")


def _supports_option_file(version: str) -> bool:
    """Whether this build accepts ``-/optname file`` (ffmpeg 6.1 and newer)."""
    match = _VERSION_NUMBER.search(version or "")
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= (6, 1)


def _probe_duration(path: Path) -> float | None:
    """Real duration of a downloaded asset, or ``None`` if it cannot be read."""
    try:
        return probe_media(path, require_audio=False).duration
    except EditorError as exc:
        log.warning("Could not probe B-roll file %s: %s", path.name, exc)
        return None


# ------------------------------------------------------------------- rendering
def render_timeline(
    *,
    source_path: Path,
    timeline: Timeline,
    work_dir: Path,
    output_path: Path,
    options: RenderOptions | None = None,
    on_progress: ProgressFn | None = None,
) -> RenderResult:
    """Render the final vertical Reel described by ``timeline``."""
    options = options or RenderOptions()
    source_media = probe_media(source_path)
    profile = RenderProfile.resolve(source_media, options)
    caps = capabilities()
    if profile.tonemap and not {"zscale", "tonemap"}.issubset(caps.filters):
        raise FFmpegError("Bu FFmpeg HDR tone mapping üçün zscale/tonemap dəstəkləmir.")
    if "libx264" not in caps.encoders:
        raise FFmpegError(
            "This FFmpeg build has no libx264 encoder.",
            detail="Reinstall FFmpeg with H.264 support.",
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(fraction: float, message: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)), message)

    # ---------------------------------------------------------- 1. B-roll files
    requested_overlays = list(timeline.broll) if options.include_broll else []
    overlays: list = []
    broll_files: list[Path] = []
    for order, overlay in enumerate(requested_overlays):
        progress(
            0.02 + 0.13 * (order / max(1, len(requested_overlays))),
            f"Downloading B-roll {order + 1}/{len(requested_overlays)}",
        )
        path = download_clip(overlay.clip, settings.cache_dir / "broll")
        # The Pexels *metadata* duration (whole seconds) is not the duration of
        # the file we actually got. Trimming past the real end silently lets the
        # base video through for the rest of the enable window, so the B-roll
        # would cut out mid-shot with no warning. Clamp to what we downloaded.
        real = _probe_duration(path)
        kept = overlay
        if real is not None and real + 1e-3 < overlay.duration:
            trimmed = min(overlay.output_end, overlay.output_start + real)
            if trimmed - overlay.output_start < 0.5:
                log.warning(
                    "B-roll %s dropped: the downloaded file is only %.2fs long.",
                    overlay.suggestion_id,
                    real,
                )
                continue
            log.warning(
                "B-roll %s shortened from %.2fs to %.2fs: the downloaded file is "
                "shorter than the Pexels metadata claimed.",
                overlay.suggestion_id,
                overlay.duration,
                trimmed - overlay.output_start,
            )
            kept = overlay.model_copy(update={"output_end": round(trimmed, 6)})
        overlays.append(kept)
        broll_files.append(path)

    active_timeline = timeline
    if overlays != list(timeline.broll):
        active_timeline = timeline.model_copy(update={"broll": overlays})

    # -------------------------------------------------------- 2. subtitle asset
    engine: str | None = None
    ass_path: Path | None = None
    raster = None
    style = SubtitleStyle.from_settings(options)
    ratio = profile.width / settings.output_width
    style = replace(
        style,
        width=profile.width,
        height=profile.height,
        fps=profile.fps,
        font_size=max(8, round(style.font_size * ratio)),
        margin_x=max(8, round(style.margin_x * ratio)),
        outline_width=max(1, round(style.outline_width * ratio)),
        shadow_offset=max(1, round(style.shadow_offset * ratio)),
    )
    if options.burn_subtitles and active_timeline.words:
        engine = active_subtitle_engine(options.subtitle_engine)
        progress(0.17, f"Generating kinetic subtitles ({engine})")
        if engine == "ass":
            ass_path = write_ass(
                work_dir / "subtitles.ass",
                build_ass(active_timeline.words, style, active_timeline.output_duration),
            )
        else:
            raster_dir = work_dir / "subtitle_frames"
            shutil.rmtree(raster_dir, ignore_errors=True)
            raster = render_raster_overlay(
                active_timeline.words,
                style,
                active_timeline.output_duration,
                raster_dir,
            )

    # ----------------------------------------------------------- 3. filtergraph
    source_duration = active_timeline.source_duration
    video_parts, video_label = _build_video_chain(active_timeline, profile)
    audio_parts, audio_label, join_mode = _build_audio_chain(
        active_timeline,
        source_duration,
        source_media.audio_start - source_media.video_start,
    )

    inputs: list[str] = ["-i", str(source_path)]
    next_input = 1
    for path in broll_files:
        inputs += ["-i", str(path)]
    broll_parts, video_label = _build_broll_chain(active_timeline, next_input, video_label, profile)
    next_input += len(broll_files)

    if options.cinematic_effect:
        nxt = "vcinematic"
        broll_parts.append(
            f"[{video_label}]eq=saturation=0.835:brightness=-0.03:gamma=1.05,"
            f"vignette=a=PI/4[{nxt}]"
        )
        video_label = nxt

    subtitle_parts: list[str] = []
    if engine == "ass" and ass_path is not None:
        fonts_dir = str(style.font_file.parent) if style.font_file else ""
        options_str = f"filename={_escape_filter_value(str(ass_path))}"
        if fonts_dir:
            options_str += f":fontsdir={_escape_filter_value(fonts_dir)}"
        subtitle_parts.append(f"[{video_label}]ass={options_str}[vsub]")
        video_label = "vsub"
    elif engine == "pillow" and raster is not None:
        # Without an explicit rate the concat demuxer stamps the image stream at
        # 25/1 with time_base 1/25, so every per-image duration snaps to a 40 ms
        # grid before `fps=` resamples it and half the caption states land a
        # frame late. Pin it to the output rate instead.
        inputs += [
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(profile.fps),
            "-i",
            str(raster.concat_file),
        ]
        subtitle_index = next_input
        next_input += 1
        subtitle_parts.append(
            f"[{subtitle_index}:v]format=rgba,fps={profile.fps},setpts=PTS-STARTPTS[subs]"
        )
        subtitle_parts.append(
            f"[{video_label}][subs]overlay=x=0:y=0:eof_action=pass:repeatlast=0:format=auto[vsub]"
        )
        video_label = "vsub"

    graph = ";".join([*video_parts, *audio_parts, *broll_parts, *subtitle_parts])
    log.debug("filter_complex: %s", graph)
    graph_path = work_dir / "filter_complex.txt"
    graph_path.write_text(graph, encoding="utf-8")

    # The graph grows linearly with the clip count (one trim + one join per
    # clip). Passing a few hundred kB of graph on argv risks E2BIG (a bare
    # OSError, not an EditorError), so it is read from the file whenever this
    # build supports it. `-/opt file` replaced `-filter_complex_script`, which
    # newer builds have removed outright.
    graph_args = (
        ["-/filter_complex", str(graph_path)]
        if _supports_option_file(caps.version)
        else ["-filter_complex", graph]
    )

    args: list[str] = [
        *inputs,
        *graph_args,
        "-map",
        f"[{video_label}]",
        "-map",
        f"[{audio_label}]",
        "-c:v",
        "libx264",
        "-preset",
        profile.preset,
        "-crf",
        str(profile.crf),
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(round(profile.fps * 2)),
        "-r",
        str(profile.fps),
        "-c:a",
        "aac",
        "-b:a",
        settings.audio_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-max_muxing_queue_size",
        "2048",
        *(
            ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
            if profile.tonemap
            else []
        ),
        str(output_path),
    ]

    log_tail: list[str] = []
    progress(0.25, "Encoding final Reel")

    def encode_progress(value: float) -> None:
        progress(0.25 + 0.75 * value, "Encoding final Reel")

    run_ffmpeg(
        args,
        total_duration=active_timeline.output_duration,
        on_progress=encode_progress,
        log_tail=log_tail,
    )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FFmpegError(
            "The render finished but produced no output file.",
            detail="\n".join(log_tail[-15:]),
        )

    progress(1.0, "Render complete")
    actual = probe_media(output_path)
    tolerance = max(0.08, 2 / profile.fps)
    if abs(actual.duration - active_timeline.output_duration) > tolerance:
        output_path.unlink(missing_ok=True)
        raise FFmpegError("Export müddəti timeline ilə uyğun gəlmir.")
    if (
        actual.audio_duration is not None
        and abs(actual.audio_duration - actual.duration) > tolerance
    ):
        output_path.unlink(missing_ok=True)
        raise FFmpegError("Export səsi və görüntüsü sinxron uzunluqda deyil.")
    if (actual.width, actual.height) != (profile.width, profile.height):
        output_path.unlink(missing_ok=True)
        raise FFmpegError("Export ölçüsü seçilmiş profilə uyğun gəlmir.")
    return RenderResult(
        path=output_path,
        duration=actual.duration,
        subtitle_engine=engine,
        broll_count=len(overlays),
        audio_join_mode=join_mode,
        log_tail=log_tail,
        metadata={"profile": asdict(profile), "media": actual.model_dump()},
    )
