"""Text-edit -> cut-list arithmetic.

Given the transcript and the set of words the user struck out, this module
produces the authoritative :class:`~app.schemas.Timeline`:

* the ordered source ranges (clips) that survive the edit,
* every kept word re-timed onto the final output timeline,
* accepted B-roll overlays re-timed onto the output timeline.

Semantics that matter for the feel of the cut
---------------------------------------------
* Cuts happen **only where words were actually removed**. Natural pauses inside
  a surviving run of words are preserved, so the delivery keeps its rhythm.
* Each clip is padded by ``CUT_PAD_MS`` so consonant onsets/offsets are never
  clipped.
* A removal that only saves less than ``MIN_REMOVABLE_GAP_MS`` is not worth a
  visible jump cut, so the two neighbouring clips are merged instead.
* Clips are never shorter than ``MIN_CLIP_MS`` (a 1-frame clip reads as a glitch
  and confuses the encoder).
* When ``trim_silences`` is on, the dead air that Silero VAD found *inside* a
  surviving passage is also removed, but only where the silence is at least
  ``SILENCE_TRIM_MS`` long - shorter pauses are part of natural delivery.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Iterable, Sequence

from .config import settings
from .errors import InvalidEditError
from .schemas import (
    BrollOverlay,
    BrollStatus,
    BrollSuggestion,
    Sentence,
    SpeechSegment,
    Timeline,
    TimelineClip,
    TimelineWord,
    Word,
)

log = logging.getLogger(__name__)


def validate_removed_ids(removed: Iterable[int], words: Sequence[Word]) -> list[int]:
    """Keep only ids that exist, de-duplicated and sorted."""
    valid = {word.id for word in words}
    return sorted({int(i) for i in removed if int(i) in valid})


def _intersect_speech(
    start: float,
    end: float,
    speech: Sequence[SpeechSegment],
    min_gap: float,
) -> list[list[float]]:
    """Split ``[start, end)`` at every silence of at least ``min_gap`` seconds.

    ``speech`` is the padded/merged VAD output, so the gaps between its segments
    are exactly the non-speech regions. Gaps shorter than ``min_gap`` are kept so
    the delivery does not sound machine-gunned.
    """
    pieces: list[list[float]] = []
    for segment in speech:
        if segment.end <= start:
            continue
        if segment.start >= end:
            break
        pieces.append([max(start, segment.start), min(end, segment.end)])
    if not pieces:
        # No VAD coverage for this range (e.g. the fallback VAD missed it):
        # keep the range intact rather than silently deleting content.
        return [[start, end]]

    merged: list[list[float]] = [pieces[0]]
    for piece_start, piece_end in pieces[1:]:
        if piece_start - merged[-1][1] < min_gap:
            merged[-1][1] = piece_end
        else:
            merged.append([piece_start, piece_end])
    return merged


#: Source/output times are emitted with this many decimals (see
#: renderer._TIME_FMT). Frame boundaries are *truncated* rather than rounded so
#: the printed value can never sit above the exact frame instant - ffmpeg's
#: ``trim`` keeps frames whose PTS is >= start, so a value rounded up by half a
#: microsecond would silently drop a whole frame.
_TIME_DECIMALS = 6
_TIME_SCALE = 10**_TIME_DECIMALS


def _floor_time(value: float) -> float:
    """Truncate ``value`` to the emitted precision, never rounding upwards."""
    return math.floor(max(0.0, value) * _TIME_SCALE) / _TIME_SCALE


def _snap_ranges_to_frames(
    ranges: Sequence[Sequence[float]],
    source_fps: float,
    source_duration: float,
) -> list[list[float]]:
    """Snap every cut to the source frame grid.

    ``[0:v]trim=start=S:end=E`` can only keep whole decoded frames: it keeps the
    frames whose PTS lies in ``[S, E)``, i.e. ``ceil(E*fps) - ceil(S*fps)``
    frames. The audio chain's ``atrim`` is sample-exact, so unless both lanes cut
    at the same instant the picture and the sound separate by up to one frame per
    cut and the error accumulates across the concat. Snapping the *timeline* to
    the frame grid makes the two lanes agree by construction and keeps
    ``output_duration`` equal to the real picture length.
    """
    fps = float(source_fps)
    eps = 1e-9
    last_frame = int(math.floor(source_duration * fps + 1e-6))
    if last_frame < 1:
        return [list(r) for r in ranges]

    snapped: list[list[float]] = []
    for start, end in ranges:
        first = int(math.ceil(start * fps - eps))
        stop = int(math.ceil(end * fps - eps))
        first = max(0, min(first, last_frame - 1))
        stop = min(max(stop, first + 1), last_frame)
        if stop <= first:
            continue
        snapped.append([_floor_time(first / fps), _floor_time(stop / fps)])

    # Snapping can make two neighbours touch or overlap; a zero-length gap is
    # not a cut, so merge them rather than emitting a redundant junction.
    merged: list[list[float]] = []
    for start, end in snapped:
        if merged and start <= merged[-1][1] + 1.0 / _TIME_SCALE:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _build_clips(
    kept: Sequence[Word],
    source_duration: float,
    speech: Sequence[SpeechSegment] = (),
    trim_silences: bool = False,
    source_fps: float | None = None,
    all_words: Sequence[Word] | None = None,
    removed_words: Sequence[Word] = (),
) -> list[TimelineClip]:
    pad = settings.cut_pad_ms / 1000.0
    merge_gap = settings.min_removable_gap_ms / 1000.0
    min_clip = settings.min_clip_ms / 1000.0

    # 1) Group kept words into runs of consecutive original indices.
    positions = {w.id: i for i, w in enumerate(all_words or kept)}
    runs: list[list[Word]] = []
    for word in kept:
        if runs and positions[word.id] == positions[runs[-1][-1].id] + 1:
            runs[-1].append(word)
        else:
            runs.append([word])

    # 2) Turn each run into a padded source range, optionally splitting out the
    #    long VAD silences that sit inside it.
    silence_gap = settings.silence_trim_ms / 1000.0
    ranges: list[list[float]] = []
    for run in runs:
        start = max(0.0, run[0].start - pad)
        end = min(source_duration, run[-1].end + pad)
        if end <= start:
            end = min(source_duration, start + min_clip)
        if trim_silences and speech:
            ranges.extend(_intersect_speech(start, end, speech, silence_gap))
        else:
            ranges.append([start, end])
    ranges.sort()

    # 3) Merge ranges whose separation is too small to be worth a jump cut.
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # 4) Enforce the minimum clip length, then re-merge if the growth caused
    #    an overlap.
    grown: list[list[float]] = []
    for start, end in merged:
        if end - start < min_clip:
            deficit = min_clip - (end - start)
            start = max(0.0, start - deficit / 2)
            end = min(source_duration, start + min_clip)
        if grown and start <= grown[-1][1]:
            grown[-1][1] = max(grown[-1][1], end)
        else:
            grown.append([start, end])

    # 5) Snap to the source frame grid so the video trim and the audio atrim cut
    #    at exactly the same instant (see _snap_ranges_to_frames).
    snapped = bool(source_fps and source_fps > 0)
    if snapped:
        grown = _snap_ranges_to_frames(grown, float(source_fps), source_duration)
    # User deletions are hard exclusions, even when shorter than merge_gap.
    # Apply AFTER padding/minimum-duration growth so neither can restore speech.
    for word in removed_words:
        start, end = word.start, word.end
        if snapped:
            start = math.floor(start * source_fps + 1e-8) / source_fps
            end = math.ceil(end * source_fps - 1e-8) / source_fps
        pieces = []
        for left, right in grown:
            if end <= left or start >= right:
                pieces.append([left, right])
            else:
                if start > left:
                    pieces.append([left, min(start, right)])
                if end < right:
                    pieces.append([max(end, left), right])
        grown = pieces
    decimals = _TIME_DECIMALS if snapped else 4

    def quantise(value: float) -> float:
        return _floor_time(value) if snapped else round(value, 4)

    clips: list[TimelineClip] = []
    output_cursor = 0.0
    for index, (start, end) in enumerate(grown):
        clip_start = quantise(start)
        clip_end = quantise(end)
        duration = max(0.0, clip_end - clip_start)
        if duration <= 0:
            continue
        clips.append(
            TimelineClip(
                index=index,
                source_start=clip_start,
                source_end=clip_end,
                output_start=round(output_cursor, decimals),
                output_end=round(output_cursor + duration, decimals),
            )
        )
        output_cursor += duration
    # Re-index after any drops so `index` always matches list position.
    for position, clip in enumerate(clips):
        clip.index = position
    return clips


def _retime_words(
    kept: Sequence[Word],
    clips: Sequence[TimelineClip],
) -> list[TimelineWord]:
    starts = [clip.source_start for clip in clips]
    retimed: list[TimelineWord] = []
    for word in kept:
        index = bisect.bisect_right(starts, word.start) - 1
        index = max(0, min(index, len(clips) - 1))
        clip = clips[index]
        # Clamp into the owning clip: padding may push a word marginally out.
        source_start = min(max(word.start, clip.source_start), clip.source_end)
        source_end = min(max(word.end, source_start), clip.source_end)
        output_start = clip.output_start + (source_start - clip.source_start)
        output_end = clip.output_start + (source_end - clip.source_start)
        if output_end <= output_start:
            output_end = min(clip.output_end, output_start + 0.04)
        retimed.append(
            TimelineWord(
                id=word.id,
                text=word.text,
                sentence_id=word.sentence_id,
                source_start=round(source_start, 4),
                source_end=round(source_end, 4),
                output_start=round(output_start, 4),
                output_end=round(output_end, 4),
            )
        )
    retimed.sort(key=lambda item: item.output_start)
    return retimed


def _retime_broll(
    suggestions: Sequence[BrollSuggestion],
    sentences: Sequence[Sentence],
    retimed_by_id: dict[int, TimelineWord],
    output_duration: float,
) -> list[BrollOverlay]:
    sentence_by_id = {sentence.id: sentence for sentence in sentences}
    overlays: list[BrollOverlay] = []

    for suggestion in suggestions:
        if suggestion.status is not BrollStatus.accepted:
            continue
        clip = suggestion.selected_clip()
        if clip is None:
            log.info("B-roll %s accepted without a clip; skipped.", suggestion.id)
            continue

        # Anchor on the suggestion's word if it survived, otherwise on the first
        # surviving word of the same sentence.
        anchor = retimed_by_id.get(suggestion.anchor_word_id)
        if anchor is None:
            sentence = sentence_by_id.get(suggestion.sentence_id)
            candidates = [
                retimed_by_id[wid]
                for wid in (sentence.word_ids if sentence else [])
                if wid in retimed_by_id
            ]
            anchor = candidates[0] if candidates else None
        if anchor is None:
            log.info(
                "B-roll %s dropped: every word of its sentence was removed.",
                suggestion.id,
            )
            continue

        requested = max(0.0, suggestion.source_end - suggestion.source_start)
        if requested <= 0:
            requested = settings.broll_default_duration
        requested = min(requested, settings.broll_max_duration)
        if clip.duration > 0:
            requested = min(requested, clip.duration)

        output_start = anchor.output_start
        output_end = min(output_duration, output_start + requested)
        if output_end - output_start < 0.5:
            log.info("B-roll %s dropped: less than 0.5s of runway left.", suggestion.id)
            continue

        overlays.append(
            BrollOverlay(
                suggestion_id=suggestion.id,
                clip=clip,
                output_start=round(output_start, 4),
                output_end=round(output_end, 4),
            )
        )

    # Resolve overlaps deterministically: earlier overlay wins, later one is
    # pushed to start where the previous ends (dropped if nothing remains).
    overlays.sort(key=lambda item: item.output_start)
    resolved: list[BrollOverlay] = []
    for overlay in overlays:
        if resolved and overlay.output_start < resolved[-1].output_end:
            overlay.output_start = resolved[-1].output_end
            if overlay.output_end - overlay.output_start < 0.5:
                continue
        resolved.append(overlay)
    return resolved


def compute_timeline(
    *,
    words: Sequence[Word],
    sentences: Sequence[Sentence],
    removed_word_ids: Iterable[int],
    source_duration: float,
    broll: Sequence[BrollSuggestion] = (),
    speech_segments: Sequence[SpeechSegment] = (),
    trim_silences: bool = False,
    source_fps: float | None = None,
) -> Timeline:
    """Recalculate the full cut list from the current text edit."""
    if not words:
        raise InvalidEditError("This project has no transcript to edit yet.")

    removed = set(validate_removed_ids(removed_word_ids, words))
    kept = [word for word in words if word.id not in removed]
    if not kept:
        raise InvalidEditError(
            "Every word is struck out - nothing would be left to render.",
            detail="Restore at least one word before rendering.",
        )

    clips = _build_clips(
        kept,
        source_duration,
        speech_segments,
        trim_silences,
        source_fps,
        all_words=words,
        removed_words=[w for w in words if w.id in removed],
    )
    if not clips:  # pragma: no cover - _build_clips always yields >=1 for kept words
        raise InvalidEditError("The current edit produces an empty timeline.")
    if len(clips) > settings.max_clips:
        raise InvalidEditError(
            f"This edit needs {len(clips)} separate cuts, more than the "
            f"{settings.max_clips} this renderer can handle in one pass.",
            detail=(
                "Remove whole phrases or sentences instead of alternating single "
                "words: every isolated removal costs one more cut."
            ),
        )

    retimed = _retime_words(kept, clips)
    output_duration = clips[-1].output_end

    overlays = _retime_broll(
        suggestions=broll,
        sentences=sentences,
        retimed_by_id={item.id: item for item in retimed},
        output_duration=output_duration,
    )

    return Timeline(
        clips=clips,
        words=retimed,
        broll=overlays,
        # Already quantised by _build_clips: re-rounding here would undo the
        # frame alignment that keeps the picture and the sound the same length.
        output_duration=output_duration,
        source_duration=round(source_duration, 4),
        removed_duration=round(max(0.0, source_duration - output_duration), 4),
        kept_word_count=len(kept),
        removed_word_count=len(removed),
        strict_deletions=bool(removed),
        source_fps=source_fps,
    )
