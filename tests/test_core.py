"""Network-free tests for the pure logic of the editor.

Run with:  pytest tests -q
Requires FFmpeg on PATH (fixtures are synthesised, not committed).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.config import settings
from app.errors import (
    FFmpegCapabilityError,
    InvalidEditError,
    MissingAudioStreamError,
    MissingVideoStreamError,
    UpstreamAPIError,
)
from app.ffmpeg_utils import active_subtitle_engine, capabilities, probe_media
from app.renderer import _build_audio_chain, _build_video_chain, _escape_filter_value
from app.schemas import (
    BrollClip,
    BrollStatus,
    BrollSuggestion,
    Sentence,
    SpeechSegment,
    Word,
)
from app.subtitles import SubtitleStyle, build_ass, build_lines, build_states
from app.timeline_manager import compute_timeline, validate_removed_ids


# --------------------------------------------------------------------- helpers
def make_words(texts: list[str], step: float = 0.5, gap: float = 0.08) -> list[Word]:
    words: list[Word] = []
    cursor = 0.0
    for index, text in enumerate(texts):
        words.append(
            Word(id=index, text=text, start=round(cursor, 4),
                 end=round(cursor + step - gap, 4), sentence_id=index // 4)
        )
        cursor += step
    return words


def make_sentences(words: list[Word], size: int = 4) -> list[Sentence]:
    sentences: list[Sentence] = []
    for start in range(0, len(words), size):
        chunk = words[start : start + size]
        sid = start // size
        for word in chunk:
            word.sentence_id = sid
        sentences.append(
            Sentence(id=sid, word_ids=[w.id for w in chunk],
                     text=" ".join(w.text for w in chunk),
                     start=chunk[0].start, end=chunk[-1].end)
        )
    return sentences


@pytest.fixture(scope="module")
def words() -> list[Word]:
    return make_words(["bir", "iki", "üç", "dörd", "beş", "altı", "yeddi", "səkkiz"])


@pytest.fixture(scope="module")
def sentences(words: list[Word]) -> list[Sentence]:
    return make_sentences(words)


# ------------------------------------------------------------------- timeline
def test_no_removals_is_one_clip(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[], source_duration=4.5)
    assert len(timeline.clips) == 1
    assert timeline.removed_word_count == 0
    assert timeline.kept_word_count == len(words)


def test_clip_durations_sum_to_output_duration(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[2, 3, 6], source_duration=4.5)
    total = sum(clip.duration for clip in timeline.clips)
    assert total == pytest.approx(timeline.output_duration, abs=1e-6)
    assert timeline.output_duration < timeline.source_duration


def test_output_times_are_contiguous_and_monotonic(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[1, 5], source_duration=4.5)
    for previous, current in zip(timeline.clips, timeline.clips[1:], strict=False):
        assert current.output_start == pytest.approx(previous.output_end, abs=1e-6)
    starts = [word.output_start for word in timeline.words]
    assert starts == sorted(starts)
    for word in timeline.words:
        assert word.output_end > word.output_start
        assert word.output_end <= timeline.output_duration + 1e-6


def test_removed_words_are_absent_from_the_timeline(words, sentences):
    removed = [0, 1]
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=removed, source_duration=4.5)
    assert {word.id for word in timeline.words}.isdisjoint(removed)
    assert timeline.removed_word_count == len(removed)


def test_removing_everything_is_rejected(words, sentences):
    with pytest.raises(InvalidEditError):
        compute_timeline(words=words, sentences=sentences,
                         removed_word_ids=[w.id for w in words], source_duration=4.5)


def test_validate_removed_ids_filters_and_dedupes(words):
    assert validate_removed_ids([3, 3, 999, -1, 0], words) == [0, 3]


def test_explicit_tiny_deletions_are_always_cut():
    """Explicit deletions must exclude their source audio, even for short words."""
    tiny = settings.min_removable_gap_ms / 1000.0 / 3
    words = [
        Word(id=0, text="a", start=0.0, end=1.0, sentence_id=0),
        Word(id=1, text="b", start=1.0, end=1.0 + tiny, sentence_id=0),
        Word(id=2, text="c", start=1.0 + tiny, end=2.0, sentence_id=0),
    ]
    sentences = make_sentences(words, size=3)
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[1], source_duration=2.0)
    assert len(timeline.clips) == 2
    assert all(c.source_end <= 1.0 or c.source_start >= 1.0 + tiny for c in timeline.clips)


def test_clips_respect_the_minimum_length():
    words = [
        Word(id=0, text="a", start=0.0, end=0.01, sentence_id=0),
        Word(id=1, text="b", start=3.0, end=3.5, sentence_id=0),
    ]
    sentences = make_sentences(words, size=2)
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[], source_duration=4.0)
    minimum = settings.min_clip_ms / 1000.0
    assert all(clip.duration >= minimum - 1e-6 for clip in timeline.clips)


# ------------------------------------------------------------ silence trimming
def test_trim_silences_splits_long_pauses():
    words = [
        Word(id=0, text="a", start=0.1, end=0.6, sentence_id=0),
        Word(id=1, text="b", start=2.2, end=2.7, sentence_id=0),
    ]
    sentences = make_sentences(words, size=2)
    speech = [SpeechSegment(start=0.0, end=0.8), SpeechSegment(start=2.0, end=2.9)]
    loose = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                            source_duration=3.0)
    tight = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                            source_duration=3.0, speech_segments=speech,
                            trim_silences=True)
    assert len(loose.clips) == 1
    assert len(tight.clips) == 2
    assert tight.output_duration < loose.output_duration


def test_trim_silences_without_vad_data_keeps_everything():
    words = make_words(["a", "b"])
    sentences = make_sentences(words, size=2)
    common = {
        "words": words,
        "sentences": sentences,
        "removed_word_ids": [],
        "source_duration": 2.0,
    }
    assert (
        compute_timeline(**common, speech_segments=[], trim_silences=True).output_duration
        == compute_timeline(**common).output_duration
    )


# --------------------------------------------------------------------- b-roll
def _suggestion(sentence_id: int, anchor: int, status: BrollStatus) -> BrollSuggestion:
    clip = BrollClip(id="c1", provider_id="1", width=1080, height=1920, duration=6.0,
                     download_url="https://example.invalid/c1.mp4")
    return BrollSuggestion(id="br1", sentence_id=sentence_id, anchor_word_id=anchor,
                           source_start=1.0, source_end=3.0, status=status,
                           candidates=[clip], selected_clip_id="c1")


def test_only_accepted_broll_reaches_the_timeline(words, sentences):
    for status, expected in ((BrollStatus.pending, 0), (BrollStatus.dismissed, 0),
                             (BrollStatus.accepted, 1)):
        timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                                   source_duration=4.5,
                                   broll=[_suggestion(1, 4, status)])
        assert len(timeline.broll) == expected, status


def test_broll_reanchors_when_its_word_is_removed(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[4],
                               source_duration=4.5,
                               broll=[_suggestion(1, 4, BrollStatus.accepted)])
    assert len(timeline.broll) == 1
    assert timeline.broll[0].output_start >= 0


def test_broll_is_dropped_when_its_whole_sentence_is_removed(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[4, 5, 6, 7], source_duration=4.5,
                               broll=[_suggestion(1, 4, BrollStatus.accepted)])
    assert timeline.broll == []


def test_broll_never_exceeds_the_output_duration(words, sentences):
    suggestion = _suggestion(1, 7, BrollStatus.accepted)
    suggestion.source_end = suggestion.source_start + 30.0
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                               source_duration=4.5, broll=[suggestion])
    for overlay in timeline.broll:
        assert overlay.output_end <= timeline.output_duration + 1e-6


# ------------------------------------------------------------------ subtitles
@pytest.fixture(scope="module")
def style() -> SubtitleStyle:
    return SubtitleStyle.from_settings()


def test_lines_respect_the_word_limit(words, sentences, style):
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                               source_duration=4.5)
    lines = build_lines(timeline.words, style)
    assert lines
    assert all(len(line.words) <= style.max_words_per_line for line in lines)


def test_highlight_states_never_overlap(words, sentences, style):
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                               source_duration=4.5)
    states = build_states(build_lines(timeline.words, style), timeline.output_duration)
    assert states
    for previous, current in zip(states, states[1:], strict=False):
        assert previous.end <= current.start + 1e-9
    assert all(state.end <= timeline.output_duration + 1e-6 for state in states)


def test_pixel_measurement_forces_earlier_breaks(style):
    long_words = make_words(["extraordinarily"] * 6)
    timeline = compute_timeline(words=long_words, sentences=make_sentences(long_words, 6),
                               removed_word_ids=[], source_duration=3.5)
    wide = build_lines(timeline.words, style)
    narrow = build_lines(timeline.words, style, lambda text: len(text) * 100.0, 300.0)
    assert sum(len(line.words) for line in narrow) == len(long_words)
    assert len(narrow) >= len(wide)
    assert all(len(line.words) == 1 for line in narrow)


def test_ass_script_structure(words, sentences, style):
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                               source_duration=4.5)
    script = build_ass(timeline.words, style, timeline.output_duration)
    assert "[Script Info]" in script and "[V4+ Styles]" in script and "[Events]" in script
    assert f"PlayResX: {settings.output_width}" in script
    assert f"PlayResY: {settings.output_height}" in script
    assert script.count("Dialogue:") == len(timeline.words)
    assert "üç" in script  # unicode survives


def test_ass_escapes_override_characters(style):
    hostile = [Word(id=0, text="{drop}", start=0.0, end=0.4, sentence_id=0),
               Word(id=1, text="back\\slash", start=0.5, end=0.9, sentence_id=0)]
    sentences = make_sentences(hostile, size=2)
    timeline = compute_timeline(words=hostile, sentences=sentences, removed_word_ids=[],
                               source_duration=1.2)
    script = build_ass(timeline.words, style, timeline.output_duration)
    assert "{drop}" not in script
    assert "(drop)" in script
    assert "back\\slash" not in script


def test_ass_timestamps_are_well_formed(words, sentences, style):
    timeline = compute_timeline(words=words, sentences=sentences, removed_word_ids=[],
                               source_duration=4.5)
    for line in build_ass(timeline.words, style, timeline.output_duration).splitlines():
        if not line.startswith("Dialogue:"):
            continue
        start, end = line.split(",")[1:3]
        for stamp in (start, end):
            hours, minutes, rest = stamp.split(":")
            seconds, centis = rest.split(".")
            assert hours.isdigit() and len(minutes) == 2 and len(seconds) == 2
            assert len(centis) == 2


# ----------------------------------------------------------------- filtergraph
def test_filter_value_escaping():
    escaped = _escape_filter_value("/tmp/a:b,c[d];e'f")
    for char in (":", ",", "[", ";", "'"):
        assert f"\\{char}" in escaped


def test_video_chain_emits_one_trim_per_clip(words, sentences):
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[2, 5], source_duration=4.5)
    parts, label = _build_video_chain(timeline)
    assert label == "vbase"
    assert sum(1 for part in parts if "trim=start=" in part) == len(timeline.clips)
    assert any(f"scale={settings.output_width}:{settings.output_height}" in p for p in parts)


def test_crossfade_chain_preserves_total_length(words, sentences):
    """Each clip's audio is extended by exactly the crossfade it feeds."""
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[2, 5], source_duration=4.5)
    parts, label, mode = _build_audio_chain(timeline, 4.5)
    assert label == "aout"
    if mode == "crossfade":
        joins = [p for p in parts if "acrossfade=" in p]
        assert len(joins) == len(timeline.clips) - 1
    else:
        assert sum(1 for p in parts if "afade=t=in" in p) >= len(timeline.clips)


def test_fade_mode_is_used_when_configured(words, sentences, monkeypatch):
    monkeypatch.setattr(settings, "audio_join_mode", "fade")
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[2], source_duration=4.5)
    _, _, mode = _build_audio_chain(timeline, 4.5)
    assert mode == "fade"


# ---------------------------------------------------------------- ffmpeg probe
def _synth(path: Path, args: list[str]) -> Path:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args, str(path)],
                   check=True)
    return path


def test_probe_reads_geometry_and_fps(tmp_path):
    path = _synth(tmp_path / "v.mp4",
                  ["-f", "lavfi", "-i", "testsrc2=size=540x960:rate=25:duration=1",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
    info = probe_media(path)
    assert (info.width, info.height) == (540, 960)
    assert round(info.fps) == 25
    assert info.has_audio and info.is_vertical


def test_probe_rejects_video_without_audio(tmp_path):
    path = _synth(tmp_path / "mute.mp4",
                  ["-f", "lavfi", "-i", "testsrc2=size=320x568:rate=25:duration=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    with pytest.raises(MissingAudioStreamError):
        probe_media(path)
    assert probe_media(path, require_audio=False).has_audio is False


def test_probe_rejects_audio_only_file(tmp_path):
    path = _synth(tmp_path / "a.m4a",
                  ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"])
    with pytest.raises(MissingVideoStreamError):
        probe_media(path)


def test_probe_detects_landscape(tmp_path):
    path = _synth(tmp_path / "wide.mp4",
                  ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=1",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
    assert probe_media(path).is_vertical is False


# --------------------------------------------------------------- capabilities
def test_capability_probe_finds_core_filters():
    caps = capabilities()
    assert {"trim", "atrim", "concat", "overlay", "afade"} <= caps.filters
    assert caps.version


def test_subtitle_engine_selection_matches_the_build():
    caps = capabilities()
    assert active_subtitle_engine() == ("ass" if caps.has_libass else "pillow")
    assert active_subtitle_engine("pillow") == "pillow"
    if caps.has_libass:
        assert active_subtitle_engine("ass") == "ass"
    else:
        with pytest.raises(FFmpegCapabilityError):
            active_subtitle_engine("ass")


def test_error_payloads_carry_stable_codes():
    from app.errors import RateLimitedError

    payload = RateLimitedError("slow down", provider="Groq", retry_after=12.0).to_dict()
    assert payload["code"] == "upstream_rate_limited"
    assert payload["provider"] == "Groq"
    assert payload["retry_after"] == 12.0
    assert json.dumps(payload)


# ------------------------------------------------------- font glyph coverage
def test_font_coverage_detects_missing_azerbaijani_glyphs():
    """Impact/Arial Black ship without `ə`; the resolver must notice."""
    from app.ffmpeg_utils import AZERBAIJANI_TEST_CHARS, font_covers

    good = Path.home() / "Library" / "Fonts" / "Montserrat-ExtraBold.ttf"
    bad = Path("/System/Library/Fonts/Supplemental/Arial Black.ttf")
    if not good.is_file() or not bad.is_file():
        pytest.skip("reference fonts are not installed on this machine")
    assert font_covers(good, AZERBAIJANI_TEST_CHARS) is True
    assert font_covers(bad, AZERBAIJANI_TEST_CHARS) is False
    # ASCII-only text must pass on both.
    assert font_covers(bad, "abc") is True


def test_resolved_font_can_render_azerbaijani():
    from app.ffmpeg_utils import AZERBAIJANI_TEST_CHARS, font_covers, resolve_font_file

    font = resolve_font_file()
    if font is None:
        pytest.skip("no fonts installed")
    assert font_covers(font, AZERBAIJANI_TEST_CHARS), (
        f"resolver picked {font.name}, which cannot render {AZERBAIJANI_TEST_CHARS}"
    )


def test_missing_explicit_font_falls_back(tmp_path):
    from app.ffmpeg_utils import resolve_font_file

    resolve_font_file.cache_clear()
    try:
        assert resolve_font_file(None, str(tmp_path / "nope.ttf")) is not None
    finally:
        resolve_font_file.cache_clear()


# --------------------------------------------- frame-accurate cut quantisation
def _fps_offsets(value: float, fps: float) -> float:
    """Distance from ``value`` to the nearest source frame boundary."""
    scaled = value * fps
    return abs(scaled - round(scaled))


def test_clips_snap_to_the_source_frame_grid():
    """`[0:v]trim` keeps whole frames only; the audio atrim is sample-exact.

    Unless both lanes cut at the same instant the picture runs ahead of the sound
    by up to one frame per cut, and concat accumulates it. With source_fps given,
    every clip boundary must therefore land on a source frame.
    """
    fps = 30.0
    words = make_words([f"w{i}" for i in range(24)], step=0.4123, gap=0.05)
    sentences = make_sentences(words, 4)
    timeline = compute_timeline(
        words=words, sentences=sentences,
        removed_word_ids=[1, 4, 7, 10, 13, 16, 19, 22],
        source_duration=words[-1].end + 1.0,
        source_fps=fps,
    )
    assert len(timeline.clips) >= 8
    for clip in timeline.clips:
        assert _fps_offsets(clip.source_start, fps) < 1e-3
        assert _fps_offsets(clip.source_end, fps) < 1e-3
        # A whole number of frames, so the picture length equals the audio length.
        assert _fps_offsets(clip.duration, fps) < 1e-3
    total = sum(clip.duration for clip in timeline.clips)
    assert total == pytest.approx(timeline.output_duration, abs=1e-6)
    assert _fps_offsets(timeline.output_duration, fps) < 1e-3


def test_unquantised_timeline_is_unchanged_without_source_fps(words, sentences):
    common = {"words": words, "sentences": sentences, "removed_word_ids": [2, 5],
              "source_duration": 4.5}
    assert (
        compute_timeline(**common).output_duration
        == compute_timeline(**common, source_fps=None).output_duration
    )


def test_absurd_clip_count_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "max_clips", 5)
    words = make_words([f"w{i}" for i in range(40)], step=0.9, gap=0.1)
    sentences = make_sentences(words, 4)
    with pytest.raises(InvalidEditError):
        compute_timeline(words=words, sentences=sentences,
                         removed_word_ids=list(range(1, 40, 2)),
                         source_duration=words[-1].end + 0.5)


def test_head_and_tail_are_faded_exactly_once(words, sentences, monkeypatch):
    """The track-level envelope already covers both ends of the concat."""
    monkeypatch.setattr(settings, "audio_join_mode", "fade")
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[2, 5], source_duration=4.5)
    parts, _, mode = _build_audio_chain(timeline, 4.5)
    assert mode == "fade"
    clip_parts = [p for p in parts if p.startswith("[0:a]") and "atrim=start=" in p]
    assert len(clip_parts) == len(timeline.clips)
    assert "afade=t=in" not in clip_parts[0]
    assert "afade=t=out" not in clip_parts[-1]
    track = parts[-1]
    assert track.count("afade=t=in") == 1
    assert track.count("afade=t=out") == 1


def test_filter_value_escaping_survives_two_unescape_passes():
    """FFmpeg unescapes a -filter_complex option value twice."""
    escaped = _escape_filter_value("/Users/x/Bob's Reels/a:b,c[d];e")

    def unescape(text: str) -> str:
        out, index = [], 0
        while index < len(text):
            if text[index] == "\\" and index + 1 < len(text):
                out.append(text[index + 1])
                index += 2
            else:
                out.append(text[index])
                index += 1
        return "".join(out)

    assert unescape(unescape(escaped)) == "/Users/x/Bob's Reels/a:b,c[d];e"


# ------------------------------------------------------- subtitle state timing
def test_states_are_at_least_one_output_frame_and_never_blink(words, sentences, style):
    frame = 1.0 / settings.output_fps
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[], source_duration=4.5)
    states = build_states(build_lines(timeline.words, style), timeline.output_duration)
    assert states
    for state in states:
        assert state.end - state.start >= frame - 1e-9
    for previous, current in zip(states, states[1:], strict=False):
        gap = current.start - previous.end
        # A sub-frame hole would be rendered as a transparent frame: the caption
        # would blink off for one or two frames at every line change.
        assert gap <= 0.0 or gap >= frame - 1e-9


def test_raster_playlist_is_constant_rate_and_frame_exact(tmp_path, words, sentences, style):
    """The concat demuxer stamps images at 1/25 unless the rate is forced.

    Forcing `-r` also forces CFR, so the playlist must carry exactly one entry
    per output frame; then every caption boundary is an exact frame instant.
    """
    from app.subtitles import render_raster_overlay

    fps = settings.output_fps
    timeline = compute_timeline(words=words, sentences=sentences,
                               removed_word_ids=[], source_duration=4.5)
    states = build_states(build_lines(timeline.words, style), timeline.output_duration)
    overlay = render_raster_overlay(
        timeline.words, style, timeline.output_duration, tmp_path / "frames"
    )
    lines = overlay.concat_file.read_text(encoding="utf-8").splitlines()
    files = [line.split("'")[1] for line in lines if line.startswith("file ")]
    durations = [float(line.split()[1]) for line in lines if line.startswith("duration ")]

    assert durations, "playlist carries no entries"
    for duration in durations:
        assert duration == pytest.approx(1.0 / fps, abs=1e-9)
    # One entry per frame, plus the trailing repeat the demuxer needs.
    assert len(files) == len(durations) + 1
    assert len(durations) == overlay.frame_count

    # Every state occupies exactly the frames its timing calls for.
    expected = sum(
        round(state.end * fps) - round(state.start * fps) for state in states
    )
    assert len(durations) >= expected


# --------------------------------------------------------- transcription logic
def test_retry_hint_parses_every_format_groq_emits():
    from app.transcription import parse_retry_hint

    assert parse_retry_hint("3600") == pytest.approx(3600.0)
    assert parse_retry_hint("2.5s") == pytest.approx(2.5)
    assert parse_retry_hint("1m30s") == pytest.approx(90.0)
    assert parse_retry_hint("7m59.56s") == pytest.approx(479.56)
    assert parse_retry_hint("2m59.56s") == pytest.approx(179.56)
    assert parse_retry_hint("") is None
    assert parse_retry_hint("not-a-duration") is None
    # An HTTP-date resolves to a non-negative number of seconds from now.
    assert parse_retry_hint("Wed, 21 Oct 2026 07:28:00 GMT") >= 0.0


def test_server_retry_hint_is_capped(monkeypatch):
    """A per-day quota reset must not put a job worker to sleep for hours."""
    import app.transcription as tx

    slept: list[float] = []
    monkeypatch.setattr(tx, "interruptible_sleep", lambda seconds: slept.append(seconds))

    class Boom(Exception):
        status_code = 429
        response = type("R", (), {"headers": {"retry-after": "7200"}})()

    def call():
        raise Boom()

    with pytest.raises(UpstreamAPIError):
        tx.with_retries(call, provider="Groq", what="testing", max_attempts=3)
    assert slept
    assert max(slept) <= 61.0


def test_retry_budget_stops_the_loop(monkeypatch):
    import app.transcription as tx

    slept: list[float] = []
    monkeypatch.setattr(tx, "interruptible_sleep", lambda seconds: slept.append(seconds))

    class Boom(Exception):
        status_code = 500

    def call():
        raise Boom()

    with pytest.raises(UpstreamAPIError):
        tx.with_retries(call, provider="Groq", what="testing", max_attempts=8, budget_s=2.0)
    assert sum(slept) <= 2.0


def test_413_is_not_retried_but_429_and_500_are():
    from app.transcription import is_transient

    def err(status):
        return type("E", (Exception,), {"status_code": status})()

    assert not is_transient(err(413))
    assert not is_transient(err(400))
    assert not is_transient(err(422))
    assert is_transient(err(429))
    assert is_transient(err(500))
    assert is_transient(err(503))


def test_parse_fixes_survives_two_json_objects():
    from app.transcription import _parse_fixes

    fixes = _parse_fixes('{"fixes":[{"i":0,"t":"hook"}]} {"fixes":[]}')
    assert fixes == [{"i": 0, "t": "hook"}]


def test_duplicate_fix_index_keeps_the_original_asr_token(monkeypatch):
    import app.transcription as tx

    monkeypatch.setattr(
        tx, "_call_normalizer",
        lambda _prompt: '{"fixes":[{"i":0,"t":"hook"},{"i":0,"t":"HOOK"}]}',
    )
    monkeypatch.setattr(settings, "normalizer_provider", "groq")
    words = [Word(id=0, text="huk", start=0.0, end=0.3, sentence_id=0)]
    result, changed = tx.normalize_terminology(words)
    assert changed == 1
    assert result[0].text == "hook"
    assert result[0].raw_text == "huk"


@pytest.mark.parametrize(
    "replacement",
    ["ho​ok", "‮kcoh", "ho ok", "hook﻿", "ho‎ok"],
)
def test_invisible_characters_are_never_written_into_a_caption(replacement):
    from app.transcription import _acceptable_replacement

    assert not _acceptable_replacement("huk", replacement)


def test_plain_phonetic_fix_is_still_accepted():
    from app.transcription import _acceptable_replacement

    assert _acceptable_replacement("huk", "hook")
    assert not _acceptable_replacement("huk", "prompt engineering")


def test_word_straddling_a_compact_seam_is_not_stretched():
    """The compacted audio glues two segments together with a 4 ms fade."""
    from app.audio_processing import AsrChunk, CompactTimeline
    from app.transcription import transcribe_chunks

    segments = [SpeechSegment(start=0.0, end=1.0), SpeechSegment(start=30.0, end=31.0)]
    timeline = CompactTimeline(segments)
    chunk = AsrChunk(index=0, path=Path("unused.flac"), compact_start=0.0, compact_end=2.0)

    payload = [
        {"word": "bir", "start": 0.1, "end": 0.5},
        {"word": "SEAM", "start": 0.9, "end": 1.1},
        {"word": "iki", "start": 1.3, "end": 1.6},
    ]
    import app.transcription as tx

    tx_words = None
    original = tx._transcribe_chunk_raw
    try:
        tx._transcribe_chunk_raw = lambda _chunk: (payload, "az")
        tx_words = transcribe_chunks([chunk], timeline)
    finally:
        tx._transcribe_chunk_raw = original

    seam = next(word for word in tx_words if word.text == "SEAM")
    assert seam.end - seam.start <= 0.25, f"seam word lasted {seam.end - seam.start:.2f}s"
    assert seam.start < 1.0


# ------------------------------------------------------------------ asr chunks
def test_one_long_segment_is_split_and_offsets_match_the_written_audio(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    from app.audio_processing import SAMPLE_RATE, build_asr_chunks

    monkeypatch.setattr(settings, "asr_chunk_seconds", 30.0)
    seconds = 95
    tone = (0.2 * np.sin(
        2 * np.pi * 220 * np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    )).astype("float32")
    wav = tmp_path / "a.wav"
    sf.write(str(wav), tone, SAMPLE_RATE, subtype="PCM_16")

    # One unbroken VAD segment, and its end runs past the decoded audio (VAD
    # padding is clamped to the ffprobe duration, which can exceed the samples).
    segments = [SpeechSegment(start=0.0, end=float(seconds) + 5.0)]
    timeline, chunks = build_asr_chunks(wav, segments, tmp_path / "asr")

    assert len(chunks) >= 3, "a single long segment must still be split"
    assert timeline.duration == pytest.approx(float(seconds), abs=0.05)
    for chunk in chunks:
        assert chunk.path.is_file()
        info = sf.info(str(chunk.path))
        assert info.frames / info.samplerate == pytest.approx(chunk.duration, abs=0.01)
    assert chunks[-1].compact_end == pytest.approx(timeline.duration, abs=0.05)
    assert not list((tmp_path / "asr").glob("asr_chunk_pending.flac"))


# ---------------------------------------------------------------- store security
def test_store_rejects_path_traversal(tmp_path):
    from app.errors import ProjectNotFoundError
    from app.store import ProjectStore

    test_store = ProjectStore(tmp_path)
    for bad_id in ("../evil", "..", "a/b", "evil\\path", "test..id"):
        with pytest.raises(ProjectNotFoundError):
            test_store.project_dir(bad_id)
        with pytest.raises(ProjectNotFoundError):
            test_store.get(bad_id)
        with pytest.raises(ProjectNotFoundError):
            test_store.get_job("valid_id", bad_id)


# ---------------------------------------------------------------- job recovery & guards
def test_job_manager_latest_for_recovers_from_disk(tmp_path):
    from app.jobs import JobManager
    from app.schemas import Job, JobStatus
    from app.store import ProjectStore

    test_store = ProjectStore(tmp_path)
    job1 = Job(
        id="job_old",
        project_id="proj_1",
        kind="ingest",
        created_at=100.0,
        status=JobStatus.succeeded,
    )
    job2 = Job(
        id="job_new",
        project_id="proj_1",
        kind="ingest",
        created_at=200.0,
        status=JobStatus.succeeded,
    )
    job_other = Job(
        id="job_other",
        project_id="proj_1",
        kind="render",
        created_at=300.0,
        status=JobStatus.succeeded,
    )
    test_store.save_job(job1)
    test_store.save_job(job2)
    test_store.save_job(job_other)

    fresh_manager = JobManager(store=test_store)
    # Memory is empty
    assert len(fresh_manager._jobs) == 0

    latest_ingest = fresh_manager.latest_for("proj_1", kind="ingest")
    assert latest_ingest is not None
    assert latest_ingest.id == "job_new"

    latest_any = fresh_manager.latest_for("proj_1")
    assert latest_any is not None
    assert latest_any.id == "job_other"

    assert fresh_manager.latest_for("proj_nonexistent") is None


def test_compact_timeline_segment_index_empty():
    from app.audio_processing import CompactTimeline

    timeline = CompactTimeline(segments=[])
    assert timeline.segment_index(0.0) == -1
    assert timeline.segment_index(10.5) == -1


def test_pipeline_set_stage_deleted_project_does_not_resurrect(tmp_path):
    from app.pipeline import _set_stage
    from app.schemas import PipelineStage, Project
    from app.store import ProjectStore

    test_store = ProjectStore(tmp_path)
    # Project is not saved to store (store.exists returns False)
    proj = Project(id="proj_deleted", original_filename="test.mp4", source_file="source.mp4")
    assert not test_store.exists(proj.id)

    def dummy_updater(*args, **kwargs):
        pass

    _set_stage(test_store, proj, PipelineStage.probing, 0.1, "Probing", dummy_updater)
    # Ensure store still does not have the project
    assert not test_store.exists(proj.id)


def test_download_clip_partial_unique_name(tmp_path, monkeypatch):
    import httpx

    from app.llm_broll import download_clip

    clip = BrollClip(
        id="clip_12345",
        provider_id="12345",
        download_url="http://example.com/video.mp4",
        duration=5.0,
        width=1080,
        height=1920,
    )
    cache_dir = tmp_path / "broll_cache"

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=65536):  # noqa: ARG002
            yield b"dummy video bytes"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream(self, method, url):  # noqa: ARG002
            return DummyResponse()

    monkeypatch.setattr(httpx, "Client", DummyClient)
    dest = download_clip(clip, cache_dir)
    assert dest.is_file()
    assert dest.name == "clip_12345.mp4"
    assert dest.read_bytes() == b"dummy video bytes"


# ---------------------------------------------------------------- word text editing
def test_update_word_text_syncs_sentence_and_timeline():
    from app.main import _apply_word_edits
    from app.pipeline import build_timeline_for
    from app.schemas import EditState, MediaInfo, Project, ProjectStatus, Sentence, Transcript, Word

    words = [
        Word(id=0, text="inammıram", start=0.0, end=0.8, sentence_id=0),
        Word(id=1, text="bu", start=0.9, end=1.2, sentence_id=0),
        Word(id=2, text="sözə", start=1.3, end=1.8, sentence_id=0),
    ]
    sentence = Sentence(id=0, word_ids=[0, 1, 2], text="inammıram bu sözə", start=0.0, end=1.8)
    transcript = Transcript(
        language="az",
        text="inammıram bu sözə",
        words=words,
        sentences=[sentence],
    )
    project = Project(
        id="proj_edit_test",
        original_filename="sample.mp4",
        status=ProjectStatus.ready,
        media=MediaInfo(duration=2.0, width=1080, height=1920, fps=30.0, has_audio=True),
        transcript=transcript,
        edit=EditState(),
    )

    # 1) Apply word edit
    _apply_word_edits(project, {0: "inanmıram"})

    assert project.transcript.words[0].text == "inanmıram"
    assert project.transcript.words[0].raw_text == "inammıram"
    assert project.transcript.sentences[0].text == "inanmıram bu sözə"
    assert project.transcript.text == "inanmıram bu sözə"

    # 2) Build timeline and check TimelineWord
    timeline = build_timeline_for(project)
    assert timeline.words[0].text == "inanmıram"
    assert timeline.words[0].source_start == 0.0


# ----------------------------------------------------------- spelling suggestions
def test_parse_and_generate_spelling_suggestions(monkeypatch):
    import app.transcription as tx
    from app.schemas import Word

    fake_response = (
        '{"suggestions": ['
        '{"i": 0, "t": "inanmıram", "r": "Orfoqrafiya düzəlişi"}, '
        '{"i": 1, "t": "məndən də", "r": "Ayrı yazılış"}'
        "]}"
    )
    monkeypatch.setattr(tx, "_call_proofreader", lambda _: fake_response)

    words = [
        Word(id=0, text="inammıram", start=0.0, end=0.8, sentence_id=0),
        Word(id=1, text="məndəndə", start=0.9, end=1.5, sentence_id=0),
        Word(id=2, text="düzdür", start=1.6, end=2.0, sentence_id=0),
    ]

    out_words, count = tx.generate_spelling_suggestions(words)
    assert count == 2
    assert out_words[0].suggestion == "inanmıram"
    reason = out_words[0].suggestion_reason or ""
    assert "inammıram" in reason or "Orfoqrafiya" in reason
    assert out_words[1].suggestion == "məndən də"
    assert out_words[2].suggestion is None


def test_merge_words_updates_transcript_and_timeline(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.schemas import EditState, MediaInfo, Project, ProjectStatus, Sentence, Transcript, Word
    from app.store import ProjectStore

    test_store = ProjectStore(tmp_path / "projects")
    monkeypatch.setattr("app.main.store", test_store)

    words = [
        Word(id=0, text="şuur", start=0.0, end=0.4, sentence_id=0),
        Word(id=1, text="altımızdır", start=0.5, end=1.2, sentence_id=0),
        Word(id=2, text="bizim", start=1.3, end=1.8, sentence_id=0),
    ]
    sentence = Sentence(id=0, word_ids=[0, 1, 2], text="şuur altımızdır bizim", start=0.0, end=1.8)
    transcript = Transcript(
        language="az",
        text="şuur altımızdır bizim",
        words=words,
        sentences=[sentence],
    )
    project = Project(
        id="proj_merge_test",
        original_filename="sample.mp4",
        status=ProjectStatus.ready,
        media=MediaInfo(duration=2.0, width=1080, height=1920, fps=30.0, has_audio=True),
        transcript=transcript,
        edit=EditState(),
    )
    test_store.project_dir(project.id).mkdir(parents=True, exist_ok=True)
    test_store.save(project)

    client = TestClient(app)
    res = client.post(
        f"/api/projects/{project.id}/words/merge",
        json={"word_ids": [0, 1], "text": "şüuraltımızdır"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    p = data["project"]
    timeline = data["timeline"]

    assert len(p["transcript"]["words"]) == 2
    merged = p["transcript"]["words"][0]
    assert merged["id"] == 0
    assert merged["text"] == "şüuraltımızdır"
    assert merged["start"] == 0.0
    assert merged["end"] == 1.2

    assert len(timeline["words"]) == 2
    assert timeline["words"][0]["text"] == "şüuraltımızdır"
    assert timeline["words"][0]["source_start"] == 0.0
    assert timeline["words"][0]["source_end"] == 1.2





