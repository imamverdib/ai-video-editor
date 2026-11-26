"""End-to-end orchestration of the two background jobs.

``run_ingest`` : upload -> probe -> audio -> VAD -> Groq Whisper -> terminology
normalisation -> B-roll suggestions. Leaves the project in ``ready`` state with
a fully populated transcript the UI can edit.

``run_render`` : current text edit -> timeline -> ffmpeg -> MP4 in ``output/``.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from . import audio_processing, llm_broll, transcription
from .config import settings
from .errors import EditorError, UnsupportedMediaError
from .ffmpeg_utils import probe_media
from .jobs import Updater
from .renderer import render_timeline
from .runtime import check_cancelled
from .schemas import (
    PipelineStage,
    Project,
    ProjectStatus,
    RenderOptions,
    Timeline,
)
from .store import ProjectStore
from .timeline_manager import compute_timeline, validate_removed_ids

log = logging.getLogger(__name__)


def _set_stage(
    store: ProjectStore,
    project: Project,
    stage: PipelineStage,
    progress: float,
    message: str,
    update: Updater,
) -> None:
    if not store.exists(project.id):
        return
    project.stage = stage
    project.progress = round(progress, 4)
    project.message = message
    store.save(project)
    update(progress, message, stage=stage.value, force=True)


# ------------------------------------------------------------------- ingestion
def run_ingest(store: ProjectStore, project_id: str, update: Updater) -> dict:
    project = store.get(project_id)
    project.status = ProjectStatus.processing
    project.error = None
    project.warnings = []
    store.save(project)

    work_dir = store.work_dir(project_id)
    source = store.source_path(project)

    try:
        # ---------------------------------------------------------- 1. probe
        _set_stage(store, project, PipelineStage.probing, 0.02, "Inspecting video streams", update)
        media = probe_media(source, require_audio=True)
        if media.duration > settings.max_source_duration_s:
            raise UnsupportedMediaError("Video müddəti icazə verilən həddi aşır.")
        if media.width * media.height > settings.max_source_pixels:
            raise UnsupportedMediaError("Video ölçüsü icazə verilən həddi aşır.")
        project.media = media
        if abs(media.aspect - 9 / 16) > 0.02:
            project.warnings.append(
                f"Source is {media.width}x{media.height}. "
                "Review crop/fit/source framing before export."
            )
        store.save(project)

        # -------------------------------------------------- 2. extract audio
        _set_stage(
            store,
            project,
            PipelineStage.extracting_audio,
            0.08,
            "Extracting 16 kHz mono audio",
            update,
        )
        wav_path = audio_processing.extract_audio(
            source, work_dir / "audio_16k.wav", offset=media.audio_start - media.video_start
        )

        # ------------------------------------------------------- 3. Silero VAD
        _set_stage(
            store,
            project,
            PipelineStage.detecting_speech,
            0.18,
            "Detecting speech with Silero VAD",
            update,
        )
        vad = audio_processing.detect_speech_segments(wav_path, media.duration)
        project.speech_segments = vad.segments
        project.speech_duration = round(vad.speech_duration, 3)
        project.warnings.extend(vad.warnings)
        store.save(project)

        timeline_map, chunks = audio_processing.build_asr_chunks(
            wav_path, vad.segments, work_dir / "asr"
        )

        # ------------------------------------------------------ 4. transcribe
        _set_stage(
            store,
            project,
            PipelineStage.transcribing,
            0.30,
            f"Transcribing {len(chunks)} chunk(s) with Groq {settings.groq_asr_model}",
            update,
        )

        def asr_progress(fraction: float, message: str) -> None:
            update(0.30 + 0.45 * fraction, message, stage=PipelineStage.transcribing.value)

        transcript = transcription.build_transcript(
            chunks,
            timeline_map,
            normalize=settings.normalizer_provider != "off",
            on_progress=asr_progress,
        )
        if transcript.normalization_applied:
            _set_stage(
                store,
                project,
                PipelineStage.normalizing,
                0.78,
                f"Normalised {transcript.normalized_word_count} term(s)",
                update,
            )
        project.warnings.extend(transcript.ai_warnings)
        project.transcript = transcript
        store.save(project)

        # ---------------------------------------------------------- 5. B-roll
        _set_stage(
            store,
            project,
            PipelineStage.suggesting_broll,
            0.85,
            "Finding contextual B-roll",
            update,
        )
        suggestions: list = []
        try:
            suggestions, broll_warnings = llm_broll.build_suggestions(transcript)
            project.warnings.extend(broll_warnings)
        except EditorError as exc:
            # B-roll is optional: never fail the whole ingest over it.
            log.warning("B-roll stage failed: %s", exc.message)
            project.warnings.append(f"B-roll suggestions unavailable: {exc.message}")
        project.broll = suggestions

        # ------------------------------------------------------------- done
        project.status = ProjectStatus.ready
        _set_stage(
            store,
            project,
            PipelineStage.done,
            1.0,
            f"Ready - {len(transcript.words)} words, {len(suggestions)} B-roll suggestion(s)",
            update,
        )

        return {
            "message": project.message,
            "words": len(transcript.words),
            "sentences": len(transcript.sentences),
            "broll": len(suggestions),
        }

    except EditorError as exc:
        project.status = ProjectStatus.failed
        project.stage = PipelineStage.failed
        project.message = exc.message
        project.error = exc.to_dict()
        store.save(project)
        raise
    except Exception as exc:
        project.status = ProjectStatus.failed
        project.stage = PipelineStage.failed
        project.message = "Unexpected error during processing"
        project.error = {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        store.save(project)
        raise


# --------------------------------------------------------------------- render
def build_timeline_for(project: Project, removed_word_ids=None) -> Timeline:
    """Compute the current timeline for a ready project."""
    from .errors import ProjectNotReadyError

    if project.transcript is None or project.media is None:
        raise ProjectNotReadyError(
            "This project has no transcript yet.",
            detail="Wait for processing to finish before editing or rendering.",
        )
    removed = project.edit.removed_word_ids if removed_word_ids is None else list(removed_word_ids)
    return compute_timeline(
        words=project.transcript.words,
        sentences=project.transcript.sentences,
        removed_word_ids=removed,
        source_duration=project.media.duration,
        broll=project.broll,
        speech_segments=project.speech_segments,
        trim_silences=project.edit.trim_silences,
        # Snap every cut to the source frame grid: the video trim can only keep
        # whole frames, so without this the picture drifts ahead of the
        # sample-exact audio by up to one frame per cut.
        source_fps=project.media.fps,
    )


def run_render(
    store: ProjectStore,
    project_id: str,
    options: RenderOptions,
    update: Updater,
    removed_word_ids: list[int] | None = None,
    snapshot: Project | None = None,
) -> dict:
    project = snapshot or store.get(project_id)
    if removed_word_ids is not None and project.transcript is not None:
        project.edit.removed_word_ids = validate_removed_ids(
            removed_word_ids, project.transcript.words
        )

    update(0.01, "Calculating the cut list", stage="timeline", force=True)
    timeline = build_timeline_for(project)

    output_name = f"reel_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = store.output_dir(project_id) / output_name

    def progress(fraction: float, message: str) -> None:
        update(0.02 + 0.97 * fraction, message, stage="rendering")

    # Every render gets its own scratch directory. Two renders of the same
    # project would otherwise rasterise captions to the same
    # ``subtitle_frames/state_%05d.png`` sequence and overwrite each other's
    # filter graph, and the first to finish would rmtree the frames the other's
    # ffmpeg is still reading - either a hard failure or, worse, an MP4 with the
    # wrong words burned in and no error at all.
    render_dir = store.work_dir(project_id) / f"render_{uuid.uuid4().hex[:12]}"
    try:
        result = render_timeline(
            source_path=store.source_path(project),
            timeline=timeline,
            work_dir=render_dir,
            output_path=output_path,
            options=options,
            on_progress=progress,
        )
        check_cancelled()
    except BaseException:
        output_path.unlink(missing_ok=True)
        cleanup_render_dir(store, project_id, render_dir)
        raise

    check_cancelled()
    manifest = {
        "project_id": project_id,
        "revision": project.revision,
        "options": options.model_dump(),
        "duration": result.duration,
        "source": project.media.model_dump() if project.media else None,
        "timeline": timeline.model_dump(),
        "render": result.metadata,
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with store.transaction(project_id) as fresh:
        fresh.output_filename = output_name
        fresh.output_revision = project.revision
    # The rasterised caption frames can be hundreds of PNGs; the MP4 is the
    # artefact worth keeping.
    cleanup_render_dir(store, project_id, render_dir)
    log.info(
        "Rendered %s (%.2fs, %d clips, %d B-roll, subtitles=%s, audio=%s)",
        output_name,
        result.duration,
        len(timeline.clips),
        result.broll_count,
        result.subtitle_engine,
        result.audio_join_mode,
    )

    return {
        "message": (
            f"Rendered {result.duration:.1f}s "
            f"({len(timeline.clips)} clips, {result.broll_count} B-roll, "
            f"subtitles: {result.subtitle_engine or 'off'})"
        ),
        "output_filename": output_name,
        "log_tail": result.log_tail,
    }


def cleanup_render_dir(store: ProjectStore, project_id: str, render_dir: Path) -> None:
    """Remove one render's scratch directory, keeping its filter graph.

    ``audio_16k.wav``, the ASR chunks and ``filter_complex.txt`` stay: they are
    small and invaluable when diagnosing a bad cut. Only the directory belonging
    to *this* render is touched, so a concurrent render of the same project keeps
    its own caption frames.
    """
    if not store.exists(project_id) or not render_dir.is_dir():
        return
    graph = render_dir / "filter_complex.txt"
    if graph.is_file():
        try:
            shutil.copyfile(graph, store.work_dir(project_id) / "filter_complex.txt")
        except OSError as exc:  # pragma: no cover - best-effort housekeeping
            log.warning("Could not preserve %s: %s", graph, exc)
    try:
        shutil.rmtree(render_dir)
    except OSError as exc:  # pragma: no cover - best-effort housekeeping
        log.warning("Could not clean %s: %s", render_dir, exc)


def cleanup_work_dir(store: ProjectStore, project_id: str) -> None:
    """Drop any leftover caption frames from an older, non-isolated render."""
    frames = store.work_dir(project_id) / "subtitle_frames"
    if not frames.is_dir():
        return
    try:
        shutil.rmtree(frames)
    except OSError as exc:  # pragma: no cover - best-effort housekeeping
        log.warning("Could not clean %s: %s", frames, exc)
