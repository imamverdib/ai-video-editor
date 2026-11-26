"""Audio extraction, Silero VAD and construction of the compacted ASR audio.

Pipeline
--------
1. ``extract_audio``          - 16 kHz mono PCM WAV out of the uploaded video.
2. ``detect_speech_segments`` - Silero VAD -> padded, merged speech regions.
3. ``build_asr_chunks``       - concatenate *only* the speech into FLAC chunks
   small enough for the Groq upload limit, plus a :class:`CompactTimeline` that
   maps every compacted second back onto the original video timeline.

Concatenating speech-only audio is what makes the transcription both cheaper
and markedly more accurate: Whisper never sees the breaths and dead air that
otherwise produce hallucinated filler tokens.
"""

from __future__ import annotations

import bisect
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import settings
from .errors import NoSpeechDetectedError
from .ffmpeg_utils import run_ffmpeg
from .schemas import SpeechSegment

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
#: Linear fade applied at every concatenation boundary of the compacted audio.
_EDGE_FADE_MS = 4.0


# ------------------------------------------------------------------ extraction
def extract_audio(video_path: Path, out_path: Path, *, offset: float = 0) -> Path:
    """Decode the video's audio track to 16 kHz mono signed-16-bit WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            *(
                ["-af", f"adelay={offset * 1000:.6f}:all=1"]
                if offset > 0
                else ["-af", f"atrim=start={-offset:.6f},asetpts=PTS-STARTPTS"]
                if offset < 0
                else []
            ),
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(out_path),
        ]
    )
    return out_path


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as float32 mono in [-1, 1]."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


# ------------------------------------------------------------------------ VAD
_silero_cache: dict[str, object] = {}
_silero_lock = threading.Lock()


def _load_silero():
    """Load (and memoise) the Silero VAD model.

    Tries the ``silero-vad`` pip package first, then ``torch.hub``. Returns
    ``None`` when torch is unavailable so the caller can degrade gracefully.
    """
    with _silero_lock:
        if "model" in _silero_cache:
            return _silero_cache["model"], _silero_cache["get_timestamps"]

        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            model = load_silero_vad()
        except Exception as pkg_exc:  # noqa: BLE001 - broad on purpose, we fall through
            log.info("silero-vad package unavailable (%s); trying torch.hub", pkg_exc)
            try:
                import torch

                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                    onnx=False,
                )
                get_speech_timestamps = utils[0]
            except Exception as hub_exc:  # noqa: BLE001
                log.warning("Silero VAD unavailable (%s).", hub_exc)
                return None, None

        _silero_cache["model"] = model
        _silero_cache["get_timestamps"] = get_speech_timestamps
        return model, get_speech_timestamps


def _silero_segments(samples: np.ndarray) -> list[tuple[float, float]] | None:
    model, get_speech_timestamps = _load_silero()
    if model is None:
        return None
    import torch

    tensor = torch.from_numpy(samples)
    with _silero_lock:
        raw = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=settings.vad_threshold,
            min_speech_duration_ms=settings.vad_min_speech_ms,
            min_silence_duration_ms=settings.vad_min_silence_ms,
            speech_pad_ms=0,  # padding is applied by us, uniformly, below
            return_seconds=True,
        )
    return [(float(item["start"]), float(item["end"])) for item in raw]


def _energy_segments(samples: np.ndarray) -> list[tuple[float, float]]:
    """Adaptive-threshold energy VAD used only when torch/Silero is missing.

    Frame-wise RMS in dBFS with a noise floor estimated from the 10th
    percentile; a frame counts as speech when it sits ``margin`` dB above that
    floor. Hysteresis (separate enter/exit thresholds) keeps word-internal dips
    from fragmenting a phrase.
    """
    frame = int(0.030 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    if samples.size < frame:
        return []

    count = 1 + (samples.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(count, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)

    floor = float(np.percentile(db, 10.0))
    peak = float(np.percentile(db, 95.0))
    span = max(6.0, peak - floor)
    enter = floor + span * 0.45
    exit_ = floor + span * 0.28

    segments: list[tuple[float, float]] = []
    active = False
    start_idx = 0
    silence_run = 0
    max_silence = max(1, int(settings.vad_min_silence_ms / 10))

    for index, value in enumerate(db):
        if not active:
            if value >= enter:
                active = True
                start_idx = index
                silence_run = 0
        else:
            if value < exit_:
                silence_run += 1
                if silence_run >= max_silence:
                    end_idx = index - silence_run
                    segments.append(
                        (start_idx * hop / SAMPLE_RATE, (end_idx * hop + frame) / SAMPLE_RATE)
                    )
                    active = False
            else:
                silence_run = 0
    if active:
        segments.append((start_idx * hop / SAMPLE_RATE, samples.size / SAMPLE_RATE))

    min_speech = settings.vad_min_speech_ms / 1000.0
    return [(s, e) for s, e in segments if e - s >= min_speech]


def _pad_and_merge(
    raw: list[tuple[float, float]],
    duration: float,
) -> list[SpeechSegment]:
    pad = settings.vad_pad_ms / 1000.0
    merge_gap = settings.vad_merge_gap_ms / 1000.0

    padded = sorted(
        (max(0.0, start - pad), min(duration, end + pad)) for start, end in raw if end > start
    )
    merged: list[list[float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [SpeechSegment(start=round(s, 4), end=round(e, 4)) for s, e in merged if e > s]


@dataclass(frozen=True)
class VadResult:
    segments: list[SpeechSegment]
    backend: str
    speech_duration: float
    warnings: tuple[str, ...] = ()


def detect_speech_segments(wav_path: Path, duration: float) -> VadResult:
    """Detect speech regions, padded by ``VAD_PAD_MS`` and merged."""
    samples, sample_rate = load_mono(wav_path)
    if sample_rate != SAMPLE_RATE:  # pragma: no cover - extract_audio enforces it
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")

    warnings: list[str] = []
    raw = _silero_segments(samples)
    backend = "silero"
    if raw is None:
        backend = "energy-fallback"
        warnings.append(
            "Silero VAD could not be loaded (PyTorch missing or incompatible with "
            "this Python version); used the built-in energy VAD instead. Install "
            "torch + silero-vad on Python 3.11/3.12 for the most precise cuts."
        )
        raw = _energy_segments(samples)

    segments = _pad_and_merge(raw, duration)
    if not segments:
        raise NoSpeechDetectedError(
            "No speech was detected in this video.",
            detail=(
                "Check that the clip actually contains voice, or lower "
                "VAD_THRESHOLD in your .env (current: "
                f"{settings.vad_threshold})."
            ),
        )
    total = sum(seg.duration for seg in segments)
    log.info(
        "VAD(%s): %d segments, %.2fs speech of %.2fs (%.0f%% kept)",
        backend,
        len(segments),
        total,
        duration,
        100 * total / max(duration, 1e-6),
    )
    return VadResult(
        segments=segments,
        backend=backend,
        speech_duration=total,
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------- compact mapping
class CompactTimeline:
    """Maps the silence-stripped ("compact") timeline back to the source.

    The compact timeline is the concatenation of all speech segments in order,
    so ``to_source`` is a piecewise-linear, monotonically increasing function.
    """

    def __init__(self, segments: list[SpeechSegment]) -> None:
        self.segments = segments
        self._starts: list[float] = []
        offset = 0.0
        for segment in segments:
            self._starts.append(offset)
            offset += segment.duration
        self.duration = offset

    def to_source(self, compact_time: float) -> float:
        if not self.segments:
            return max(0.0, compact_time)
        clamped = min(max(compact_time, 0.0), self.duration)
        index = bisect.bisect_right(self._starts, clamped) - 1
        index = max(0, min(index, len(self.segments) - 1))
        segment = self.segments[index]
        local = clamped - self._starts[index]
        return min(segment.end, segment.start + local)

    def segment_index(self, compact_time: float) -> int:
        if not self.segments:
            return -1
        clamped = min(max(compact_time, 0.0), self.duration)
        index = bisect.bisect_right(self._starts, clamped) - 1
        return max(0, min(index, len(self.segments) - 1))


@dataclass(frozen=True)
class AsrChunk:
    """One FLAC file uploaded to Groq, plus its offset on the compact timeline."""

    index: int
    path: Path
    compact_start: float
    compact_end: float

    @property
    def duration(self) -> float:
        return self.compact_end - self.compact_start


def build_asr_chunks(
    wav_path: Path,
    segments: list[SpeechSegment],
    out_dir: Path,
) -> tuple[CompactTimeline, list[AsrChunk]]:
    """Write speech-only FLAC chunks and return them with the time mapping.

    Every returned offset is derived from the samples that were *actually*
    written, so the compact timeline can never claim more audio than Whisper
    hears (VAD padding can push a segment end past the decoded audio length).
    """
    samples, _ = load_mono(wav_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_chunk = max(30.0, settings.asr_chunk_seconds)
    fade = max(1, int(_EDGE_FADE_MS / 1000.0 * SAMPLE_RATE))
    size_limit = max(1 << 20, settings.asr_max_upload_bytes)

    # 1) Clamp every segment to the audio we really decoded, then split anything
    #    longer than one upload's worth. A single unbroken VAD segment (VAD
    #    merges pauses under 120 ms, so continuous delivery yields one very long
    #    segment) was previously never split and could exceed Groq's file limit.
    pieces: list[tuple[int, int]] = []  # (begin_sample, end_sample)
    max_piece_samples = max(1, int(round(max_chunk * SAMPLE_RATE)))
    for segment in segments:
        begin = max(0, int(round(segment.start * SAMPLE_RATE)))
        finish = min(samples.size, int(round(segment.end * SAMPLE_RATE)))
        if finish <= begin:
            continue
        cursor = begin
        while cursor < finish:
            stop = min(finish, cursor + max_piece_samples)
            pieces.append((cursor, stop))
            cursor = stop

    if not pieces:  # pragma: no cover - detect_speech_segments already guards
        raise NoSpeechDetectedError("Speech segments produced no audio to transcribe.")

    # 2) The compact timeline is built from the effective (clamped, split)
    #    bounds, so compact offsets always match the written sample counts.
    effective = [
        SpeechSegment(start=begin / SAMPLE_RATE, end=stop / SAMPLE_RATE) for begin, stop in pieces
    ]
    timeline = CompactTimeline(effective)

    def render(index: int) -> np.ndarray:
        begin, stop = pieces[index]
        piece = samples[begin:stop].copy()
        span = min(fade, piece.size // 2)
        if span > 1:
            ramp = np.linspace(0.0, 1.0, span, dtype=np.float32)
            piece[:span] *= ramp
            piece[-span:] *= ramp[::-1]
        return piece

    groups: list[list[int]] = []
    current: list[int] = []
    current_samples = 0
    for index, (begin, stop) in enumerate(pieces):
        length = stop - begin
        if current and current_samples + length > max_piece_samples:
            groups.append(current)
            current, current_samples = [], 0
        current.append(index)
        current_samples += length
    if current:
        groups.append(current)

    chunks: list[AsrChunk] = []
    pending = deque(groups)
    scratch = out_dir / "asr_chunk_pending.flac"
    while pending:
        group = pending.popleft()
        audio = np.concatenate([render(index) for index in group])
        sf.write(str(scratch), audio, SAMPLE_RATE, subtype="PCM_16", format="FLAC")
        if scratch.stat().st_size > size_limit:
            if len(group) > 1:
                middle = len(group) // 2
                pending.appendleft(group[middle:])
                pending.appendleft(group[:middle])
                continue
            log.warning(
                "ASR chunk %d is %.1f MB, above the %.1f MB upload limit, and "
                "cannot be split further.",
                len(chunks),
                scratch.stat().st_size / 1e6,
                size_limit / 1e6,
            )
        path = out_dir / f"asr_chunk_{len(chunks):03d}.flac"
        scratch.replace(path)

        compact_start = timeline._starts[group[0]]  # noqa: SLF001 - same module
        written = sum(pieces[index][1] - pieces[index][0] for index in group)
        chunks.append(
            AsrChunk(
                index=len(chunks),
                path=path,
                compact_start=compact_start,
                compact_end=compact_start + written / SAMPLE_RATE,
            )
        )
    scratch.unlink(missing_ok=True)
    log.info(
        "Prepared %d ASR chunk(s), %.1fs total compacted speech",
        len(chunks),
        timeline.duration,
    )
    return timeline, chunks


def estimated_chunk_count(speech_duration: float) -> int:
    return max(1, math.ceil(speech_duration / max(30.0, settings.asr_chunk_seconds)))
