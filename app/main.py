"""FastAPI application: upload, edit, B-roll curation, render, download.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import mimetypes
import re
import shutil
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import pipeline
from .config import settings
from .errors import (
    BrollNotFoundError,
    ConfigurationError,
    EditorError,
    InvalidEditError,
    ProjectNotFoundError,
    ProjectNotReadyError,
    UnsupportedMediaError,
)
from .ffmpeg_utils import (
    AZERBAIJANI_TEST_CHARS,
    active_subtitle_engine,
    capabilities,
    font_covers,
    resolve_font_file,
)
from .jobs import Updater, jobs
from .llm_broll import refresh_candidates
from .schemas import (
    BrollDecisionRequest,
    BrollSearchRequest,
    BrollStatus,
    BrollSuggestion,
    CapabilitiesResponse,
    EditRequest,
    EditState,
    Job,
    JobStatus,
    MergeWordsRequest,
    PipelineStage,
    Project,
    ProjectStatus,
    RenderRequest,
    Timeline,
    Transcript,
    WordUpdateRequest,
)
from .store import ProjectStore, store
from .timeline_manager import validate_removed_ids
from .transcription import generate_spelling_suggestions
from .ui import mount_frontend

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")

_UPLOAD_CHUNK = 1 << 20  # 1 MiB
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    jobs.reconcile()
    caps = capabilities()
    log.info("Using %s", caps.version)
    log.info("Subtitle engine: %s", active_subtitle_engine())
    for warning in caps.warnings:
        log.warning(warning)
    if not settings.groq_api_key:
        log.warning("GROQ_API_KEY is not set - transcription will fail.")
    if not settings.pexels_api_key:
        log.warning("PEXELS_API_KEY is not set - B-roll search will be disabled.")
    try:
        yield
    finally:
        jobs.shutdown()


app = FastAPI(
    title="Text-Based Video Editor",
    description=(
        "Edit vertical Instagram Reels by editing their transcript: Silero VAD + "
        "Groq whisper-large-v3 word timestamps, LLM B-roll suggestions, kinetic "
        "subtitles and an FFmpeg render pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


# ------------------------------------------------------------ error handling
@app.exception_handler(EditorError)
async def _editor_error_handler(_: Request, exc: EditorError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})


@app.exception_handler(HTTPException)
async def _http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "http_error", "message": str(exc.detail)}},
    )


def get_store() -> ProjectStore:
    return store


@app.post("/api/projects/{project_id}/history/{direction}", tags=["edit"])
async def edit_history(
    project_id: str,
    direction: str,
    db: ProjectStore = Depends(get_store),
) -> dict:
    if direction not in {"undo", "redo"}:
        raise InvalidEditError("Unknown history action.")
    with db.transaction(project_id, record_history=False) as project:
        history = project.undo_history if direction == "undo" else project.redo_history
        opposite = project.redo_history if direction == "undo" else project.undo_history
        if not history:
            raise InvalidEditError("Geri qaytarılacaq redaktə yoxdur.")
        opposite.append(db.snapshot(project))
        previous = history.pop()
        project.edit = EditState.model_validate(previous["edit"])
        project.transcript = Transcript.model_validate(previous["transcript"])
        project.broll = [BrollSuggestion.model_validate(b) for b in previous["broll"]]
        timeline = pipeline.build_timeline_for(project)
    return {"project": db.get(project_id), "timeline": timeline}


@app.post("/api/projects/{project_id}/retry", response_model=Job, tags=["projects"])
async def retry_ingest(project_id: str, db: ProjectStore = Depends(get_store)) -> Job:
    with db.lock_for(project_id):
        project = db.get(project_id)
        if project.status is not ProjectStatus.failed:
            raise ProjectNotReadyError("Yalnız uğursuz emalı yenidən başlatmaq olar.")
        running = jobs.latest_for(project_id, "ingest")
        if running and running.status in (JobStatus.queued, JobStatus.running):
            raise ProjectNotReadyError("Emal artıq davam edir.")
        with db.transaction(project_id, record_history=False) as fresh:
            fresh.status = ProjectStatus.uploaded
            fresh.stage = PipelineStage.queued
            fresh.error = None
            fresh.progress = 0
        return jobs.submit(
            project_id=project_id,
            kind="ingest",
            body=lambda update: pipeline.run_ingest(db, project_id, update),
        )


@app.post("/api/projects/{project_id}/cancel", tags=["projects"])
async def cancel_project_jobs(project_id: str, db: ProjectStore = Depends(get_store)) -> dict:
    db.get(project_id)
    jobs.cancel_project(project_id)
    return {"status": "cancelling"}


# --------------------------------------------------------------------- health
@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities", response_model=CapabilitiesResponse, tags=["system"])
async def get_capabilities() -> CapabilitiesResponse:
    caps = capabilities()
    font = resolve_font_file()
    warnings = list(caps.warnings)
    covers = font_covers(font) if font else False
    if font and not covers:
        warnings.append(
            f"The subtitle font '{font.name}' cannot render every Azerbaijani "
            f"letter ({AZERBAIJANI_TEST_CHARS}); install Montserrat or set "
            "SUBTITLE_FONT_FILE."
        )
    elif not font:
        warnings.append("No subtitle font file was found on this machine.")
    return CapabilitiesResponse(
        ffmpeg_version=caps.version,
        has_libass=caps.has_libass,
        has_drawtext=caps.has_drawtext,
        subtitle_engine=active_subtitle_engine(),
        groq_configured=bool(settings.groq_api_key),
        pexels_configured=bool(settings.pexels_api_key),
        normalizer_provider=settings.normalizer_provider,
        normalizer_model=settings.agy_model if settings.normalizer_provider == "agy" else None,
        agy_available=bool(shutil.which(settings.agy_binary)),
        font_file=str(font) if font else None,
        font_covers_language=covers,
        warnings=warnings,
    )


# ------------------------------------------------------------------- projects
@app.post("/api/projects", response_model=Project, status_code=201, tags=["projects"])
async def create_project(
    file: UploadFile = File(...),
    db: ProjectStore = Depends(get_store),
) -> Project:
    """Upload a vertical video and kick off the ingest pipeline."""
    if not settings.groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not configured.",
            detail="Add GROQ_API_KEY to .env and restart the server.",
        )
    if not file.filename:
        raise UnsupportedMediaError("The upload had no filename.")

    project = db.create(file.filename)
    destination = db.source_path(project)
    temp_dest = destination.with_suffix(destination.suffix + ".uploading")
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with temp_dest.open("wb") as handle:
            while chunk := await file.read(_UPLOAD_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise UnsupportedMediaError(
                        f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                        detail="Trim the clip or raise MAX_UPLOAD_MB in .env.",
                    )
                handle.write(chunk)
        if written == 0:
            raise UnsupportedMediaError("The uploaded file was empty.")
        temp_dest.replace(destination)  # atomic rename
    except Exception:
        temp_dest.unlink(missing_ok=True)
        db.delete(project.id)
        raise
    finally:
        await file.close()

    project.message = "Queued for processing"
    db.save(project)

    def body(update: Updater) -> dict:
        return pipeline.run_ingest(db, project.id, update)

    job = jobs.submit(project_id=project.id, kind="ingest", body=body)
    log.info("Project %s created (%.1f MB) -> job %s", project.id, written / 1e6, job.id)
    return db.get(project.id)


@app.get("/api/projects", response_model=list[Project], tags=["projects"])
async def list_projects(db: ProjectStore = Depends(get_store)) -> list[Project]:
    return db.list()


@app.get("/api/projects/{project_id}", response_model=Project, tags=["projects"])
async def get_project(project_id: str, db: ProjectStore = Depends(get_store)) -> Project:
    return db.get(project_id)


@app.delete("/api/projects/{project_id}", status_code=204, tags=["projects"])
async def delete_project(project_id: str, db: ProjectStore = Depends(get_store)) -> Response:
    jobs.cancel_project(project_id)
    db.delete(project_id)
    return Response(status_code=204)


@app.get("/api/projects/{project_id}/job", response_model=Job, tags=["projects"])
async def latest_ingest_job(
    project_id: str,
    db: ProjectStore = Depends(get_store),
) -> Job:
    """The most recent ingest job for this project (for progress polling)."""
    if not db.exists(project_id):
        raise ProjectNotFoundError(f"No project with id '{project_id}'.")
    job = jobs.latest_for(project_id, kind="ingest")
    if job is None:
        raise ProjectNotReadyError("No ingest job is recorded for this project.")
    return job


# ---------------------------------------------------------------------- edits
@app.get("/api/projects/{project_id}/timeline", response_model=Timeline, tags=["edit"])
async def get_timeline(project_id: str, db: ProjectStore = Depends(get_store)) -> Timeline:
    return pipeline.build_timeline_for(db.get(project_id))


def _apply_word_edits(project: Project, word_edits: dict[int, str]) -> None:
    if not project.transcript or not word_edits:
        return
    words_by_id = {w.id: w for w in project.transcript.words}
    changed = False
    for wid, new_text in word_edits.items():
        if wid in words_by_id:
            w = words_by_id[wid]
            clean = new_text.strip()
            if clean and clean != w.text:
                if w.raw_text is None:
                    w.raw_text = w.text
                w.text = clean
                w.suggestion = None
                w.suggestion_reason = None
                changed = True
    if changed:
        for sentence in project.transcript.sentences:
            sentence.text = " ".join(
                words_by_id[wid].text for wid in sentence.word_ids if wid in words_by_id
            )
        project.transcript.text = " ".join(s.text for s in project.transcript.sentences)


@app.api_route(
    "/api/projects/{project_id}/edit",
    methods=["PUT", "POST"],
    tags=["edit"],
)
async def update_edit(
    project_id: str,
    payload: EditRequest,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Persist the struck-out words and return the recalculated timeline."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        if (
            payload.client_id is not None
            and payload.sequence is not None
            and payload.sequence <= project.edit_clients.get(payload.client_id, -1)
        ):
            return {"project": project, "timeline": pipeline.build_timeline_for(project)}
        if payload.expected_revision is not None and payload.expected_revision != project.revision:
            raise ProjectNotReadyError("Layihə başqa redaktə ilə dəyişib; yenidən açın.")
        if payload.client_id is not None and payload.sequence is not None:
            project.edit_clients[payload.client_id] = payload.sequence
            project.edit_clients = dict(list(project.edit_clients.items())[-32:])
        project.edit.removed_word_ids = validate_removed_ids(
            payload.removed_word_ids, project.transcript.words
        )
        if payload.trim_silences is not None:
            project.edit.trim_silences = payload.trim_silences
        if payload.word_edits:
            _apply_word_edits(project, payload.word_edits)
        timeline = pipeline.build_timeline_for(project)
    return {"project": db.get(project_id), "timeline": timeline}


@app.api_route(
    "/api/projects/{project_id}/words/{word_id}",
    methods=["PATCH", "PUT"],
    tags=["edit"],
)
async def update_word(
    project_id: str,
    word_id: int,
    payload: WordUpdateRequest,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Edit the text of a single word in the transcript."""
    new_text = payload.text.strip()
    if not new_text:
        raise InvalidEditError("Word text cannot be empty.")

    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        target_word = None
        for word in project.transcript.words:
            if word.id == word_id:
                target_word = word
                break
        if target_word is None:
            raise InvalidEditError(f"No word with id '{word_id}'.")

        if target_word.raw_text is None and target_word.text != new_text:
            target_word.raw_text = target_word.text
        target_word.text = new_text
        target_word.suggestion = None
        target_word.suggestion_reason = None
        project.transcript.suggestions_count = sum(
            bool(w.suggestion) for w in project.transcript.words
        )

        words_by_id = {w.id: w for w in project.transcript.words}
        for sentence in project.transcript.sentences:
            if word_id in sentence.word_ids:
                sentence.text = " ".join(
                    words_by_id[wid].text for wid in sentence.word_ids if wid in words_by_id
                )
        project.transcript.text = " ".join(s.text for s in project.transcript.sentences)
        timeline = pipeline.build_timeline_for(project)

    return {"project": db.get(project_id), "timeline": timeline, "word": target_word}


@app.post("/api/projects/{project_id}/words/merge", tags=["edit"])
async def merge_words(
    project_id: str,
    payload: MergeWordsRequest,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Merge two or more consecutive words into a single token."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")

        words_by_id = {w.id: w for w in project.transcript.words}
        ordered_ids = [w.id for w in project.transcript.words]
        if len(set(payload.word_ids)) != len(payload.word_ids):
            raise InvalidEditError("Eyni söz iki dəfə birləşdirilə bilməz.")
        positions = sorted(ordered_ids.index(i) for i in payload.word_ids if i in words_by_id)
        if positions and positions != list(range(positions[0], positions[-1] + 1)):
            raise InvalidEditError("Yalnız qonşu sözləri birləşdirmək olar.")
        target_words = []
        for wid in payload.word_ids:
            if wid not in words_by_id:
                raise InvalidEditError(f"No word with id '{wid}'.")
            target_words.append(words_by_id[wid])

        target_words.sort(key=lambda w: w.start)
        if len({w.sentence_id for w in target_words}) != 1:
            raise InvalidEditError("Yalnız eyni cümlədəki sözləri birləşdirmək olar.")
        if any(w.id in project.edit.removed_word_ids for w in target_words):
            raise InvalidEditError("Birləşdirmədən əvvəl silinmiş sözləri bərpa edin.")
        first_word = target_words[0]
        last_word = target_words[-1]

        if payload.text and payload.text.strip():
            merged_text = payload.text.strip()
        else:
            merged_text = "".join(w.text for w in target_words)

        raw_parts = [w.raw_text or w.text for w in target_words]
        combined_raw = " ".join(raw_parts)

        first_word.text = merged_text
        first_word.end = last_word.end
        first_word.raw_text = combined_raw
        first_word.suggestion = None
        first_word.suggestion_reason = None

        remove_ids = {w.id for w in target_words[1:]}
        for suggestion in project.broll:
            if suggestion.anchor_word_id in remove_ids:
                suggestion.anchor_word_id = first_word.id
        project.transcript.words = [w for w in project.transcript.words if w.id not in remove_ids]
        project.edit.removed_word_ids = [
            wid for wid in project.edit.removed_word_ids if wid not in remove_ids
        ]

        for sentence in project.transcript.sentences:
            sentence.word_ids = [wid for wid in sentence.word_ids if wid not in remove_ids]

        current_words = {w.id: w for w in project.transcript.words}
        for sentence in project.transcript.sentences:
            sentence.text = " ".join(
                current_words[wid].text for wid in sentence.word_ids if wid in current_words
            )
            if sentence.word_ids:
                sentence.start = current_words[sentence.word_ids[0]].start
                sentence.end = current_words[sentence.word_ids[-1]].end
        project.transcript.text = " ".join(s.text for s in project.transcript.sentences)
        project.transcript.suggestions_count = sum(
            1 for w in project.transcript.words if w.suggestion
        )

        timeline = pipeline.build_timeline_for(project)

    return {
        "project": db.get(project_id),
        "timeline": timeline,
        "merged_word": first_word,
    }


# ------------------------------------------------------------- proofreading
@app.post("/api/projects/{project_id}/proofread", tags=["edit"])
async def proofread_project(
    project_id: str,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Run AI proofreading to detect spelling/grammar errors and produce suggestions."""
    snapshot = db.get(project_id)
    if snapshot.transcript is None or snapshot.status is not ProjectStatus.ready:
        raise ProjectNotReadyError("This project has no transcript yet.")
    warnings: list[str] = []
    words, count = await run_in_threadpool(
        generate_spelling_suggestions,
        snapshot.transcript.words,
        warnings=warnings,
    )
    with db.transaction(project_id) as project:
        if project.revision != snapshot.revision:
            raise ProjectNotReadyError("Mətn yoxlama zamanı dəyişdi; yenidən yoxlayın.")
        project.transcript.words = words
        project.transcript.suggestions_count = count
        project.transcript.ai_warnings = warnings
        project.transcript.ai_provider = settings.normalizer_provider
        project.transcript.ai_model = (
            settings.agy_model if settings.normalizer_provider == "agy" else None
        )

    return {
        "project": db.get(project_id),
        "suggestions_count": count,
        "warnings": warnings,
        "ai_status": "disabled"
        if settings.normalizer_provider == "off"
        else ("degraded" if warnings else "succeeded"),
    }


@app.post("/api/projects/{project_id}/words/{word_id}/suggestion/accept", tags=["edit"])
async def accept_word_suggestion(
    project_id: str,
    word_id: int,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Accept an AI suggested spelling correction for a word."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        target_word = None
        for word in project.transcript.words:
            if word.id == word_id:
                target_word = word
                break
        if target_word is None:
            raise InvalidEditError(f"No word with id '{word_id}'.")
        if not target_word.suggestion:
            raise InvalidEditError(f"Word '{word_id}' has no pending suggestion.")

        if target_word.raw_text is None:
            target_word.raw_text = target_word.text
        target_word.text = target_word.suggestion
        target_word.suggestion = None
        target_word.suggestion_reason = None

        words_by_id = {w.id: w for w in project.transcript.words}
        for sentence in project.transcript.sentences:
            if word_id in sentence.word_ids:
                sentence.text = " ".join(
                    words_by_id[wid].text for wid in sentence.word_ids if wid in words_by_id
                )
        project.transcript.text = " ".join(s.text for s in project.transcript.sentences)
        project.transcript.suggestions_count = sum(
            1 for w in project.transcript.words if w.suggestion
        )
        timeline = pipeline.build_timeline_for(project)

    return {"project": db.get(project_id), "timeline": timeline, "word": target_word}


@app.post("/api/projects/{project_id}/words/{word_id}/suggestion/dismiss", tags=["edit"])
async def dismiss_word_suggestion(
    project_id: str,
    word_id: int,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Dismiss an AI suggested correction for a word."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        target_word = None
        for word in project.transcript.words:
            if word.id == word_id:
                target_word = word
                break
        if target_word is None:
            raise InvalidEditError(f"No word with id '{word_id}'.")

        target_word.suggestion = None
        target_word.suggestion_reason = None
        project.transcript.suggestions_count = sum(
            1 for w in project.transcript.words if w.suggestion
        )

    return {"project": db.get(project_id), "word": target_word}


@app.post("/api/projects/{project_id}/suggestions/accept-all", tags=["edit"])
async def accept_all_suggestions(
    project_id: str,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Accept all pending AI suggestions in the transcript at once."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        changed = 0
        for word in project.transcript.words:
            if word.suggestion:
                if word.raw_text is None:
                    word.raw_text = word.text
                word.text = word.suggestion
                word.suggestion = None
                word.suggestion_reason = None
                changed += 1

        if changed > 0:
            words_by_id = {w.id: w for w in project.transcript.words}
            for sentence in project.transcript.sentences:
                sentence.text = " ".join(
                    words_by_id[wid].text for wid in sentence.word_ids if wid in words_by_id
                )
            project.transcript.text = " ".join(s.text for s in project.transcript.sentences)
        project.transcript.suggestions_count = 0
        timeline = pipeline.build_timeline_for(project)

    return {"project": db.get(project_id), "timeline": timeline, "accepted_count": changed}


@app.post("/api/projects/{project_id}/suggestions/dismiss-all", tags=["edit"])
async def dismiss_all_suggestions(
    project_id: str,
    db: ProjectStore = Depends(get_store),
) -> dict[str, object]:
    """Dismiss all pending AI suggestions in the transcript."""
    with db.transaction(project_id) as project:
        if project.transcript is None or project.status is not ProjectStatus.ready:
            raise ProjectNotReadyError("This project has no transcript yet.")
        for word in project.transcript.words:
            word.suggestion = None
            word.suggestion_reason = None
        project.transcript.suggestions_count = 0

    return {"project": db.get(project_id)}


# --------------------------------------------------------------------- b-roll
def _find_suggestion(project: Project, suggestion_id: str) -> BrollSuggestion:
    for suggestion in project.broll:
        if suggestion.id == suggestion_id:
            return suggestion
    raise BrollNotFoundError(f"No B-roll suggestion '{suggestion_id}'.")


@app.post(
    "/api/projects/{project_id}/broll/{suggestion_id}/search",
    response_model=BrollSuggestion,
    tags=["broll"],
)
async def search_broll(
    project_id: str,
    suggestion_id: str,
    payload: BrollSearchRequest,
    db: ProjectStore = Depends(get_store),
) -> BrollSuggestion:
    """Re-run the Pexels search with a user-supplied query ("Swap")."""
    snapshot = db.get(project_id)
    suggestion = _find_suggestion(snapshot, suggestion_id).model_copy(deep=True)
    await run_in_threadpool(refresh_candidates, suggestion, payload.query.strip())
    with db.transaction(project_id) as project:
        if project.revision != snapshot.revision:
            raise ProjectNotReadyError("Layihə axtarış zamanı dəyişdi; yenidən cəhd edin.")
        project.broll = [suggestion if b.id == suggestion_id else b for b in project.broll]
        # Copy inside the lock so the response cannot observe a later mutation.
        result = suggestion.model_copy(deep=True)
    return result  # noqa: RET504


@app.post(
    "/api/projects/{project_id}/broll/{suggestion_id}/decision",
    response_model=BrollSuggestion,
    tags=["broll"],
)
async def decide_broll(
    project_id: str,
    suggestion_id: str,
    payload: BrollDecisionRequest,
    db: ProjectStore = Depends(get_store),
) -> BrollSuggestion:
    """Accept / dismiss a suggestion, optionally choosing a specific clip."""
    with db.transaction(project_id) as project:
        suggestion = _find_suggestion(project, suggestion_id)
        if payload.clip_id:
            known = {clip.id for clip in suggestion.candidates}
            if payload.clip_id not in known:
                raise BrollNotFoundError(
                    f"Clip '{payload.clip_id}' is not among this suggestion's candidates.",
                    detail="Re-run the search - candidate ids change with every query.",
                )
            suggestion.selected_clip_id = payload.clip_id
        if payload.status is BrollStatus.accepted and suggestion.selected_clip() is None:
            raise ProjectNotReadyError(
                "This suggestion has no clip to accept.",
                detail="Search for a different query first.",
            )
        suggestion.status = payload.status
        # Copy inside the lock so the response cannot observe a later mutation.
        result = suggestion.model_copy(deep=True)
    return result  # noqa: RET504


# --------------------------------------------------------------------- render
@app.post("/api/projects/{project_id}/render", response_model=Job, tags=["render"])
async def start_render(
    project_id: str,
    payload: RenderRequest,
    db: ProjectStore = Depends(get_store),
) -> Job:
    project = db.get(project_id)
    if project.status is not ProjectStatus.ready:
        raise ProjectNotReadyError(
            f"Project is '{project.status.value}', not ready to render.",
            detail=project.message or None,
        )
    # Validate the edit up-front so a bad request fails synchronously with 400.
    pipeline.build_timeline_for(project, payload.removed_word_ids)

    # Lock the project to prevent TOCTOU: two concurrent render requests could
    # both pass the in-flight check and submit duplicate FFmpeg jobs.
    with db.lock_for(project_id):
        in_flight = jobs.latest_for(project_id, kind="render")
        if in_flight is not None and in_flight.status in (JobStatus.queued, JobStatus.running):
            raise ProjectNotReadyError(
                "A render of this project is already in progress.",
                detail="Wait for it to finish (or for it to fail) before starting another.",
            )
        if payload.removed_word_ids is not None:
            with db.transaction(project_id) as fresh:
                fresh.edit.removed_word_ids = validate_removed_ids(
                    payload.removed_word_ids,
                    fresh.transcript.words,
                )
        render_snapshot = db.get(project_id).model_copy(deep=True)

        def body(update: Updater) -> dict:
            return pipeline.run_render(
                db,
                project_id,
                payload.options,
                update,
                snapshot=render_snapshot,
            )

        job = jobs.submit(project_id=project_id, kind="render", body=body)
        with db.transaction(project_id) as fresh:
            fresh.last_render_job_id = job.id
    return job


@app.get("/api/jobs/{job_id}", response_model=Job, tags=["render"])
async def get_job(job_id: str, project_id: str | None = None) -> Job:
    return jobs.get(job_id, project_id)


# ---------------------------------------------------------------------- media
def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            block = handle.read(min(_UPLOAD_CHUNK, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


def _ranged_response(path: Path, request: Request, filename: str | None = None) -> Response:
    """Serve a file with HTTP Range support so <video> can seek."""
    if not path.is_file():
        raise ProjectNotFoundError(f"File not found: {path.name}")
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if filename:
        headers["Content-Disposition"] = (
            "attachment; filename=\"video_reel.mp4\"; filename*=UTF-8''" + quote(filename, safe="")
        )

    range_header = request.headers.get("range")
    match = _RANGE_RE.fullmatch(range_header.strip()) if range_header else None
    if not match:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter_file(path, 0, size), media_type=media_type, headers=headers)

    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    else:  # suffix range: bytes=-N
        length = int(raw_end or 0)
        start = max(0, size - length)
        end = size - 1
    if start >= size or start > end:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )
    end = min(end, size - 1)
    length = end - start + 1
    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        }
    )
    return StreamingResponse(
        _iter_file(path, start, length),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )


@app.get("/api/projects/{project_id}/source", tags=["media"])
async def stream_source(
    project_id: str,
    request: Request,
    db: ProjectStore = Depends(get_store),
) -> Response:
    project = db.get(project_id)
    return _ranged_response(db.source_path(project), request)


@app.get("/api/projects/{project_id}/output", tags=["media"])
async def stream_output(
    project_id: str,
    request: Request,
    db: ProjectStore = Depends(get_store),
    name: str | None = None,
) -> Response:
    project = db.get(project_id)
    target = name or project.output_filename
    if not target:
        raise ProjectNotReadyError("This project has not been rendered yet.")
    path = db.output_dir(project_id) / Path(target).name
    return _ranged_response(path, request)


@app.get("/api/projects/{project_id}/download", tags=["media"])
async def download_output(
    project_id: str,
    request: Request,
    db: ProjectStore = Depends(get_store),
    name: str | None = None,
) -> Response:
    project = db.get(project_id)
    target = name or project.output_filename
    if not target:
        raise ProjectNotReadyError("This project has not been rendered yet.")
    path = db.output_dir(project_id) / Path(target).name
    stem = Path(project.original_filename).stem
    return _ranged_response(path, request, filename=f"{stem}_reel.mp4")


@app.get("/sw.js", include_in_schema=False)
async def service_worker_unregister() -> Response:
    """Self-destruct script for any stale service workers left from previous apps."""
    script = (
        "self.addEventListener('install', () => self.skipWaiting());\n"
        "self.addEventListener('activate', (e) => {\n"
        "  self.registration.unregister();\n"
        "  e.waitUntil(self.clients.claim());\n"
        "});\n"
    )
    return Response(content=script, media_type="application/javascript")

from pydantic import BaseModel

class PublishRequest(BaseModel):
    platform: str
    schedule_time: float | None = None

@app.post("/api/projects/{project_id}/social/generate_text", tags=["social"])
async def api_generate_social_text(
    project_id: str,
    db: ProjectStore = Depends(get_store)
) -> dict[str, object]:
    from .social import generate_social_posts
    from .schemas import SocialPost
    project = db.get(project_id)
    
    posts = await run_in_threadpool(generate_social_posts, project)
    
    with db.transaction(project_id) as p:
        for platform, text in posts.items():
            if platform not in p.social_posts:
                p.social_posts[platform] = SocialPost(platform=platform)
            p.social_posts[platform].text = text
            
    # Serialize cleanly
    return {"social_posts": {k.value: v.model_dump() for k, v in db.get(project_id).social_posts.items()}}

from pydantic import BaseModel
class GenerateCoverRequest(BaseModel):
    cover_title: str | None = None

@app.post("/api/projects/{project_id}/social/generate_cover", tags=["social"])
async def api_generate_social_cover(
    project_id: str,
    req: GenerateCoverRequest,
    db: ProjectStore = Depends(get_store)
) -> dict[str, object]:
    from .social import generate_cover_image
    project = db.get(project_id)
    
    if req.cover_title:
        title = req.cover_title
    else:
        title = project.transcript.text[:30] + "..." if project.transcript else "Yeni Video"
    
    cover_name = await run_in_threadpool(generate_cover_image, project, title, db.source_path(project), db.output_dir(project_id))
    
    with db.transaction(project_id) as p:
        p.social_cover_image = cover_name
        
    return {"cover_image": cover_name}

@app.post("/api/projects/{project_id}/social/publish", tags=["social"])
async def api_social_publish(
    project_id: str,
    req: PublishRequest,
    db: ProjectStore = Depends(get_store)
) -> dict[str, object]:
    # Placeholder for Composio Integration
    with db.transaction(project_id) as p:
        # Just update status to mock the integration
        from .schemas import SocialPlatform
        plat = SocialPlatform(req.platform)
        if plat in p.social_posts:
            post = p.social_posts[plat]
            post.status = "scheduled" if req.schedule_time else "published"
            post.schedule_time = req.schedule_time
    
    return {"status": "ok"}

# The SPA mount must come last so it never shadows an /api route.
mount_frontend(app)
