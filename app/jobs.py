"""Background job execution for the two long-running operations: ingest & render.

A small fixed-size thread pool is used rather than ``asyncio`` tasks because the
work is CPU/IO bound in C libraries (ffmpeg, torch, HTTP uploads) that would
otherwise block the event loop. Job state is persisted to disk so a status poll
still works after a server reload.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .agy_provider import session_scope
from .config import settings
from .errors import EditorError, ProjectNotReadyError
from .runtime import JobCancelled, cancel_event, check_cancelled
from .schemas import Job, JobStatus, PipelineStage, ProjectStatus
from .store import ProjectStore
from .store import store as default_store

log = logging.getLogger(__name__)

#: How often job progress is flushed to disk (ffmpeg emits progress ~2x/sec).
_FLUSH_INTERVAL = 0.4

#: Signature of the callback handed to a job body.
Updater = Callable[..., None]
JobBody = Callable[[Updater], "dict | None"]


class JobManager:
    def __init__(self, store: ProjectStore | None = None, max_workers: int = 2) -> None:
        self.store = store or default_store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="editor-job"
        )
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict = {}

    # ------------------------------------------------------------------ public
    def submit(self, *, project_id: str, kind: str, body: JobBody) -> Job:
        job = Job(id=f"job_{uuid.uuid4().hex[:12]}", project_id=project_id, kind=kind)
        with self._guard:
            active = sum(
                j.status in (JobStatus.queued, JobStatus.running) for j in self._jobs.values()
            )
            if active >= settings.max_job_queue:
                raise ProjectNotReadyError("İş növbəsi doludur; cari işin bitməsini gözləyin.")
            self._jobs[job.id] = job
            self._cancel[job.id] = threading.Event()
        self.store.save_job(job)
        self._futures[job.id] = self._executor.submit(self._run, job, body)
        return job

    def get(self, job_id: str, project_id: str | None = None) -> Job:
        with self._guard:
            job = self._jobs.get(job_id)
        if job is not None:
            return job.model_copy()
        if project_id:
            return self.store.get_job(project_id, job_id)
        return self.store.find_job(job_id)

    def latest_for(self, project_id: str, kind: str | None = None) -> Job | None:
        with self._guard:
            candidates: dict[str, Job] = {
                job.id: job
                for job in self._jobs.values()
                if job.project_id == project_id and (kind is None or job.kind == kind)
            }
        try:
            jobs_dir = self.store.project_dir(project_id) / "jobs"
            if jobs_dir.is_dir():
                for path in jobs_dir.glob("*.json"):
                    if path.stem in candidates:
                        continue
                    try:
                        job = Job.model_validate_json(path.read_text(encoding="utf-8"))
                        if job.project_id == project_id and (kind is None or job.kind == kind):
                            candidates[job.id] = job
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            pass
        if not candidates:
            return None
        best = max(candidates.values(), key=lambda job: job.created_at)
        return best.model_copy()

    def reconcile(self) -> None:
        """Called once at startup, before accepting new work."""
        for path in self.store.root.glob("*/jobs/*.json"):
            try:
                job = Job.model_validate_json(path.read_text())
                if job.status not in (JobStatus.running, JobStatus.queued):
                    continue
                job.status = JobStatus.failed
                job.stage = "interrupted"
                job.message = "Server yenidən başladıldı; işi yenidən başladın."
                self.store.save_job(job)
                if job.kind == "ingest":
                    with self.store.transaction(job.project_id, record_history=False) as project:
                        project.status = ProjectStatus.failed
                        project.stage = PipelineStage.failed
                        project.message = job.message
            except (EditorError, ValueError, OSError):
                log.warning("Could not reconcile a persisted job.")

    def cancel_project(self, project_id: str) -> None:
        with self._guard:
            targets = [
                j
                for j in self._jobs.values()
                if j.project_id == project_id and j.status in (JobStatus.queued, JobStatus.running)
            ]
        for job in targets:
            self._cancel[job.id].set()
            future = self._futures.get(job.id)
            if future is not None and future.cancel():
                job.status = JobStatus.cancelled
                job.stage = "cancelled"
                job.message = "Əməliyyat dayandırıldı."
                if self.store.exists(project_id):
                    self.store.save_job(job)
                    if job.kind == "ingest":
                        with self.store.transaction(project_id, record_history=False) as project:
                            project.status = ProjectStatus.failed
                            project.stage = PipelineStage.failed
                            project.message = job.message

    def shutdown(self) -> None:
        for project_id in {j.project_id for j in self._jobs.values()}:
            self.cancel_project(project_id)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------------- runtime
    def _run(self, job: Job, body: JobBody) -> None:
        token = cancel_event.set(self._cancel[job.id])
        job.status = JobStatus.running
        job.stage = "starting"
        last_flush = 0.0

        def update(
            progress: float | None = None,
            message: str | None = None,
            *,
            stage: str | None = None,
            force: bool = False,
        ) -> None:
            nonlocal last_flush
            check_cancelled()
            if progress is not None:
                job.progress = round(max(0.0, min(1.0, float(progress))), 4)
            if message is not None:
                job.message = message
            changed_stage = stage is not None and stage != job.stage
            if stage is not None:
                job.stage = stage
            now = time.monotonic()
            if force or changed_stage or (now - last_flush) >= _FLUSH_INTERVAL:
                last_flush = now
                self.store.save_job(job)

        try:
            check_cancelled()
            self.store.save_job(job)
            with session_scope():
                result = body(update) or {}
            check_cancelled()
            job.status = JobStatus.succeeded
            job.progress = 1.0
            job.stage = "done"
            job.message = str(result.get("message") or "Completed")
            output = result.get("output_filename")
            job.output_filename = str(output) if output else None
            log_tail = result.get("log_tail")
            if isinstance(log_tail, list):
                job.log_tail = [str(line) for line in log_tail[-40:]]
        except JobCancelled as exc:
            job.status = JobStatus.cancelled
            job.stage = "cancelled"
            job.message = str(exc)
        except EditorError as exc:
            job.status = JobStatus.failed
            job.stage = "failed"
            job.message = exc.message
            job.error = exc.to_dict()
            log.warning("Job %s (%s) failed: %s", job.id, job.kind, exc.message)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            job.status = JobStatus.failed
            job.stage = "failed"
            job.message = "Unexpected error"
            job.error = {
                "code": "internal_error",
                "message": f"{type(exc).__name__}: {exc}",
            }
            log.error("Job %s (%s) crashed:\n%s", job.id, job.kind, traceback.format_exc())
        finally:
            cancel_event.reset(token)
            if self.store.exists(job.project_id):
                self.store.save_job(job)


jobs = JobManager()
