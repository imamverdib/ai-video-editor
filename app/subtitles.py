"""Kinetic word-by-word subtitles for 9:16 output.

Two interchangeable engines share one layout model:

``ass``
    Emits an Advanced SubStation Alpha script that FFmpeg burns in with the
    ``ass`` filter (libass). Highest quality: real font shaping, animated
    scale-up on the active word, sub-pixel outlines.

``pillow``
    Rasterises one RGBA PNG per highlight state and feeds them to FFmpeg as a
    variable-duration image sequence (concat demuxer) that is composited with a
    single ``overlay``. This is the automatic fallback for FFmpeg builds without
    libass/libfreetype - still a full, real renderer, not a degraded stub.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from .config import settings
from .errors import ConfigurationError
from .ffmpeg_utils import resolve_font_file
from .schemas import RenderOptions

log = logging.getLogger(__name__)

#: How long the last word of a line lingers after it finishes being spoken.
_LINE_TAIL = 0.18
#: Minimum on-screen time for a single highlight state.
_MIN_STATE = 0.09


def _frame_period() -> float:
    """One output frame, in seconds - the finest timing the overlay can carry."""
    return 1.0 / max(1, settings.output_fps)


class SubtitleWord(Protocol):
    text: str
    sentence_id: int
    output_start: float
    output_end: float


# ------------------------------------------------------------------- styling
def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ConfigurationError(f"Invalid colour '{value}'; expected #RRGGBB.")
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid colour '{value}'; expected #RRGGBB.") from exc


def _ass_colour(value: str, alpha: int = 0) -> str:
    red, green, blue = _hex_to_rgb(value)
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


@dataclass
class SubtitleStyle:
    width: int
    height: int
    font_family: str
    font_file: Path | None
    font_size: int
    base_color: str
    highlight_color: str
    outline_color: str
    outline_width: int
    shadow_offset: int
    position: str
    max_words_per_line: int
    max_chars_per_line: int
    line_break_gap: float
    uppercase: bool
    highlight_scale: float
    margin_x: int = 72
    fps: float = 30

    @classmethod
    def from_settings(cls, options: RenderOptions | None = None) -> SubtitleStyle:
        options = options or RenderOptions()
        font_file = resolve_font_file(settings.subtitle_font_family, settings.subtitle_font_file)
        font_size = options.subtitle_font_size or settings.subtitle_font_size
        outline_width = (
            int(round(font_size * 0.10))
            if options.subtitle_font_size
            else settings.subtitle_outline_width
        )
        shadow_offset = (
            int(round(font_size * 0.05))
            if options.subtitle_font_size
            else settings.subtitle_shadow_offset
        )
        return cls(
            fps=settings.output_fps,
            width=settings.output_width,
            height=settings.output_height,
            font_family=settings.subtitle_font_family,
            font_file=font_file,
            font_size=font_size,
            base_color=settings.subtitle_base_color,
            highlight_color=(options.subtitle_highlight_color or settings.subtitle_highlight_color),
            outline_color=settings.subtitle_outline_color,
            outline_width=outline_width,
            shadow_offset=shadow_offset,
            position=(options.subtitle_position or settings.subtitle_position),
            max_words_per_line=settings.subtitle_max_words_per_line,
            max_chars_per_line=settings.subtitle_max_chars_per_line,
            line_break_gap=settings.subtitle_line_break_gap_ms / 1000.0,
            uppercase=(
                settings.subtitle_uppercase
                if options.subtitle_uppercase is None
                else options.subtitle_uppercase
            ),
            highlight_scale=settings.subtitle_highlight_scale,
        )

    @property
    def block_center_y(self) -> int:
        return int(self.height * (0.5 if self.position == "center" else 0.72))


# --------------------------------------------------------------- line grouping
@dataclass
class SubtitleLine:
    words: list[str]
    starts: list[float]
    ends: list[float]

    @property
    def start(self) -> float:
        return self.starts[0]

    @property
    def end(self) -> float:
        return self.ends[-1]


def build_lines(
    words: Sequence[SubtitleWord],
    style: SubtitleStyle,
    measure: Callable[[str], float] | None = None,
    max_pixel_width: float | None = None,
) -> list[SubtitleLine]:
    """Group re-timed words into short, punchy caption lines.

    ``measure``/``max_pixel_width`` make the break decision pixel-accurate
    (the Pillow engine knows its exact glyph metrics); the character limit alone
    would let a wide line spill past the safe margins.
    """
    lines: list[SubtitleLine] = []
    current: SubtitleLine | None = None

    for index, word in enumerate(words):
        text = word.text.strip()
        if not text:
            continue
        if style.uppercase:
            text = text.replace("i", "İ").replace("ı", "I").upper()

        previous = words[index - 1] if index else None
        must_break = False
        if current is None:
            must_break = True
        else:
            gap = word.output_start - (previous.output_end if previous else word.output_start)
            candidate = " ".join([*current.words, text])
            too_wide = len(candidate) > style.max_chars_per_line
            if measure is not None and max_pixel_width is not None:
                too_wide = measure(candidate) > max_pixel_width
            must_break = (
                len(current.words) >= style.max_words_per_line
                or too_wide
                or gap >= style.line_break_gap
                or (previous is not None and previous.sentence_id != word.sentence_id)
                or (previous is not None and previous.text.rstrip().endswith((".", "!", "?", "…")))
            )

        if must_break:
            current = SubtitleLine(words=[], starts=[], ends=[])
            lines.append(current)

        assert current is not None
        current.words.append(text)
        current.starts.append(float(word.output_start))
        current.ends.append(float(word.output_end))

    return [line for line in lines if line.words]


@dataclass
class HighlightState:
    """One rendered frame-state: a line with exactly one word highlighted."""

    line_index: int
    word_index: int
    words: list[str]
    start: float
    end: float


def build_states(
    lines: Sequence[SubtitleLine], total_duration: float, fps: float | None = None
) -> list[HighlightState]:
    states: list[HighlightState] = []
    for line_index, line in enumerate(lines):
        next_line_start = (
            lines[line_index + 1].start if line_index + 1 < len(lines) else total_duration
        )
        for word_index, _ in enumerate(line.words):
            start = line.starts[word_index]
            if word_index + 1 < len(line.words):
                end = max(line.starts[word_index + 1], start + _MIN_STATE)
            else:
                end = min(line.ends[word_index] + _LINE_TAIL, next_line_start)
                end = max(end, start + _MIN_STATE)
            states.append(
                HighlightState(
                    line_index=line_index,
                    word_index=word_index,
                    words=list(line.words),
                    start=round(max(0.0, start), 4),
                    end=round(min(total_duration, end), 4),
                )
            )

    # Guarantee strict monotonicity so no two states ever overlap, and make
    # every state at least one output frame long: a shorter state cannot survive
    # the image-sequence resample at all, so the word would be spoken with no
    # highlight ever appearing.
    frame = 1 / fps if fps else _frame_period()
    cleaned: list[HighlightState] = []
    for state in states:
        if cleaned and state.start < cleaned[-1].end:
            state.start = round(cleaned[-1].end, 4)
        if state.start >= total_duration:
            continue
        if state.end - state.start < frame:
            state.end = round(min(total_duration, state.start + frame), 4)
        if state.end - state.start < frame / 2.0:
            continue
        cleaned.append(state)

    # Close any hole shorter than one output frame. Such a hole would otherwise
    # be rendered as a fully transparent frame, blinking the caption off at
    # every line change - the exact artefact kinetic captions exist to avoid.
    for previous, current in zip(cleaned, cleaned[1:], strict=False):
        if 0.0 < current.start - previous.end < frame:
            previous.end = current.start
    return cleaned


# ------------------------------------------------------------------ ASS engine
def _ass_time(seconds: float) -> str:
    total = max(0, round(seconds * 100))
    hours, remainder = divmod(total, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "∖").replace("{", "(").replace("}", ")").replace("\n", " ")


def fitted_style(words: Sequence[SubtitleWord], style: SubtitleStyle) -> SubtitleStyle:
    """Fit an indivisible long token inside safe margins in both engines."""
    if not words or style.font_file is None:
        return style
    font = _load_font(style, max(8, round(style.font_size * style.highlight_scale)))
    available = max(1, style.width - 2 * style.margin_x - 4 * style.outline_width)
    texts = [
        w.text.replace("i", "İ").replace("ı", "I").upper() if style.uppercase else w.text
        for w in words
    ]
    widest = max(font.getlength(t) for t in texts)
    size = max(8, int(style.font_size * min(1, available / max(1, widest))))
    return replace(style, font_size=size, font_family=font.getname()[0])


def build_ass(
    words: Sequence[SubtitleWord],
    style: SubtitleStyle,
    total_duration: float,
) -> str:
    """Render a complete .ass script with per-word karaoke highlighting."""
    style = fitted_style(words, style)
    max_width = float(style.width - 2 * style.margin_x)
    measure: Callable[[str], float] | None = None
    if style.font_file:
        try:
            from PIL import ImageFont

            hfont = ImageFont.truetype(
                str(style.font_file),
                max(8, int(style.font_size * style.highlight_scale)),
            )
            sp_width = hfont.getlength(" ")

            def _measure(text: str) -> float:
                parts = text.split(" ")
                return sum(hfont.getlength(p) for p in parts) + sp_width * (len(parts) - 1)

            measure = _measure
        except Exception:
            measure = None

    lines = build_lines(words, style, measure, max_width)
    states = build_states(lines, total_duration, style.fps)

    alignment = 5 if style.position == "center" else 2
    margin_v = 0 if alignment == 5 else int(style.height - style.block_center_y)
    base = _ass_colour(style.base_color)
    highlight = _ass_colour(style.highlight_color)
    outline = _ass_colour(style.outline_color)
    scale = int(round(style.highlight_scale * 100))

    header = [
        "[Script Info]",
        "; Generated by the Text-Based Video Editor",
        "ScriptType: v4.00+",
        f"PlayResX: {style.width}",
        f"PlayResY: {style.height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Kinetic,{style.font_family},{style.font_size},{base},{highlight},"
            f"{outline},&H96000000,-1,0,0,0,100,100,1,0,1,"
            f"{style.outline_width},{style.shadow_offset},{alignment},"
            f"{style.margin_x},{style.margin_x},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    for state in states:
        pieces: list[str] = []
        for index, word in enumerate(state.words):
            escaped = _ass_escape(word)
            if index == state.word_index:
                pieces.append(
                    f"{{\\c{highlight}\\fscx100\\fscy100"
                    f"\\t(0,90,\\fscx{scale}\\fscy{scale})}}{escaped}"
                )
            else:
                pieces.append(f"{{\\c{base}\\fscx100\\fscy100}}{escaped}")
        text = "{\\q2}" + " ".join(pieces)
        events.append(
            f"Dialogue: 0,{_ass_time(state.start)},{_ass_time(state.end)},Kinetic,,0,0,0,,{text}"
        )

    return "\n".join([*header, *events, ""])


def write_ass(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------- Pillow engine
@dataclass
class RasterOverlay:
    """Result of the Pillow engine: a concat playlist of RGBA PNG states."""

    concat_file: Path
    frame_count: int
    directory: Path
    files: list[Path] = field(default_factory=list)


def _load_font(style: SubtitleStyle, size: int):
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "Pillow is required for the fallback subtitle renderer.",
            detail="Run: pip install -r requirements.txt",
        ) from exc

    if style.font_file is None:
        raise ConfigurationError(
            "No usable font file was found for subtitle rendering.",
            detail=(
                "Install a bold font (e.g. Montserrat ExtraBold) or set "
                "SUBTITLE_FONT_FILE=/path/to/Font.ttf in your .env."
            ),
        )
    try:
        return ImageFont.truetype(str(style.font_file), size)
    except OSError as exc:
        raise ConfigurationError(
            f"Could not load font '{style.font_file}'.",
            detail=str(exc),
        ) from exc


def _wrap_words(
    words: Sequence[str],
    measure: Callable[[str], float],
    max_width: float,
    space_width: float,
) -> list[list[int]]:
    """Split word indices into rows that fit ``max_width``."""
    rows: list[list[int]] = []
    row: list[int] = []
    row_width = 0.0
    for index, word in enumerate(words):
        width = measure(word)
        extra = width if not row else space_width + width
        if row and row_width + extra > max_width:
            rows.append(row)
            row, row_width = [index], width
        else:
            row.append(index)
            row_width += extra
    if row:
        rows.append(row)
    return rows


def render_raster_overlay(
    words: Sequence[SubtitleWord],
    style: SubtitleStyle,
    total_duration: float,
    out_dir: Path,
) -> RasterOverlay:
    """Rasterise every highlight state to a transparent PNG + concat playlist."""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)

    style = fitted_style(words, style)
    font = _load_font(style, style.font_size)
    highlight_font = _load_font(style, max(8, int(style.font_size * style.highlight_scale)))
    space_width = highlight_font.getlength(" ")
    max_width = style.width - 2 * style.margin_x
    line_height = int(style.font_size * 1.18)

    # Every word occupies a slot as wide as its *highlighted* rendition, so the
    # layout is identical for all states of a line: no jitter as the highlight
    # advances, and no overlap when the active word scales up.
    def slot_width(word: str) -> float:
        return highlight_font.getlength(word)

    def measure_line(text: str) -> float:
        parts = text.split(" ")
        return sum(slot_width(part) for part in parts) + space_width * (len(parts) - 1)

    lines = build_lines(words, style, measure_line, max_width)
    states = build_states(lines, total_duration, style.fps)

    base_rgb = _hex_to_rgb(style.base_color)
    highlight_rgb = _hex_to_rgb(style.highlight_color)
    outline_rgb = _hex_to_rgb(style.outline_color)

    blank_path = out_dir / "state_blank.png"
    Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0)).save(blank_path)

    frame = 1 / style.fps
    # (file, intended end time). Durations are derived from these boundaries at
    # write time so they are whole output frames and never accumulate error.
    timed: list[tuple[Path, float]] = []
    generated: list[Path] = []
    cursor = 0.0

    for order, state in enumerate(states):
        # A hole shorter than one output frame is not representable: emitting a
        # transparent frame for it blinks the caption off, so absorb it by
        # letting the previous entry run on instead.
        if state.start - cursor >= frame:
            timed.append((blank_path, state.start))
        cursor = state.end

        canvas = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        rows = _wrap_words(state.words, slot_width, max_width, space_width)
        total_height = len(rows) * line_height
        top = style.block_center_y - total_height // 2

        for row_index, row in enumerate(rows):
            row_width = sum(slot_width(state.words[i]) for i in row)
            row_width += space_width * (len(row) - 1)
            x = (style.width - row_width) / 2
            y = top + row_index * line_height
            for word_index in row:
                word = state.words[word_index]
                slot = slot_width(word)
                active = word_index == state.word_index
                chosen = highlight_font if active else font
                # Centre each rendition inside its fixed slot.
                offset = (slot - chosen.getlength(word)) / 2
                draw.text(
                    (x + offset, y - (int(line_height * 0.035) if active else 0)),
                    word,
                    font=chosen,
                    fill=(*(highlight_rgb if active else base_rgb), 255),
                    stroke_width=style.outline_width,
                    stroke_fill=(*outline_rgb, 235),
                )
                x += slot + space_width

        path = out_dir / f"state_{order:05d}.png"
        canvas.save(path, optimize=True)
        generated.append(path)
        timed.append((path, state.end))

    if total_duration - cursor >= frame:
        timed.append((blank_path, total_duration))
    if not timed:
        timed.append((blank_path, max(0.1, total_duration)))

    # Emit a *constant-rate* playlist: one entry per output frame, each state's
    # PNG repeated for as many frames as it occupies.
    #
    # The concat demuxer gives an image stream time_base 1/25 by default, so a
    # variable-duration playlist snaps every boundary to a 40 ms grid - captions
    # then land up to a frame late against sample-exact audio, and a sub-frame
    # entry can collapse to zero and never be shown at all. Forcing the input
    # rate (`-r`) fixes the grid but also forces CFR, which is only correct if
    # every entry really is one frame long - which is exactly what this builds.
    fps = max(1, style.fps)
    frame_duration = 1.0 / fps
    entries: list[Path] = []
    previous_frame = 0
    for path, end in timed:
        boundary = max(previous_frame + 1, int(round(end * fps)))
        entries.extend([path] * (boundary - previous_frame))
        previous_frame = boundary

    playlist = out_dir / "subtitles_concat.txt"
    with playlist.open("w", encoding="utf-8") as handle:
        handle.write("ffconcat version 1.0\n")
        for path in entries:
            handle.write(f"file '{path.name}'\n")
            handle.write(f"duration {frame_duration:.9f}\n")
        # The concat demuxer ignores the final `duration` unless the last file
        # is repeated without one.
        handle.write(f"file '{entries[-1].name}'\n")

    log.info(
        "Rasterised %d subtitle states (%d PNGs, %d overlay frames at %d fps)",
        len(states),
        len(generated),
        len(entries),
        fps,
    )
    return RasterOverlay(
        concat_file=playlist,
        frame_count=len(entries),
        directory=out_dir,
        files=generated,
    )
