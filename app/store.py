"""Filesystem-backed project persistence.

One directory per project::

    data/projects/<id>/
        project.json          # the serialised Project model
        source.mp4            # the uploaded video, verbatim
        work/                 # audio, ASR chunks, .ass, subtitle PNGs, graph dump
        output/reel_*.mp4     # rendered results
        jobs/<job_id>.json    # persisted job records

Writes are atomic (temp file + ``os.replace``) and guarded by a per-project
re-entrant lock, so concurrent edit/render requests can never interleave a
half-written ``project.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import settings
from .errors import JobNotFoundError, ProjectNotFoundError, UnsupportedMediaError
from .schemas import EditState, Job, Project

log = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_id(identifier: str, name: str = "project") -> str:
    if not isinstance(identifier, str) or not _SAFE_ID_RE.match(identifier):
        raise ProjectNotFoundError(f"Invalid {name} id: '{identifier}'.")
    return identifier


def _atomic_write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


class ProjectStore:
    """Thread-safe CRUD over the project directories."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.projects_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ locks
    def lock_for(self, project_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(project_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[project_id] = lock
            return lock

    @contextmanager
    def transaction(self, project_id: str, *, record_history: bool = True) -> Iterator[Project]:
        """Load, mutate and persist a project under its lock."""
        with self.lock_for(project_id):
            project = self.get(project_id)
            before = self.snapshot(project)
            revision = project.revision
            yield project
            if self.snapshot(project) != before:
                if record_history:
                    project.undo_history = (project.undo_history + [before])[-30:]
                    project.redo_history = []
                project.revision = revision + 1
            self.save(project)

    @staticmethod
    def snapshot(project: Project) -> dict:
        return {
            "edit": project.edit.model_dump(mode="json"),
            "transcript": project.transcript.model_dump(mode="json")
            if project.transcript
            else None,
            "broll": [b.model_dump(mode="json") for b in project.broll],
        }

    def _ensure_live(self, project_id: str) -> None:
        if (self.root / ".deleted" / _validate_id(project_id)).exists():
            raise ProjectNotFoundError("Project was deleted.")

    # ------------------------------------------------------------------ paths
    def project_dir(self, project_id: str) -> Path:
        pid = _validate_id(project_id, "project")
        path = (self.root / pid).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ProjectNotFoundError(f"Invalid project id: '{project_id}'.")
        return path

    def project_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"

    def work_dir(self, project_id: str) -> Path:
        self._ensure_live(project_id)
        path = self.project_dir(project_id) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_dir(self, project_id: str) -> Path:
        self._ensure_live(project_id)
        path = self.project_dir(project_id) / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def jobs_dir(self, project_id: str) -> Path:
        self._ensure_live(project_id)
        path = self.project_dir(project_id) / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def source_path(self, project: Project) -> Path:
        path = (self.project_dir(project.id) / project.source_file).resolve()
        if not path.is_relative_to(self.project_dir(project.id)):
            raise ProjectNotFoundError("Invalid source path.")
        return path

    # ------------------------------------------------------------------- CRUD
    def create(self, original_filename: str) -> Project:
        suffix = Path(original_filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise UnsupportedMediaError(
                f"Unsupported file type '{suffix or 'unknown'}'.",
                detail="Upload one of: " + ", ".join(sorted(ALLOWED_SUFFIXES)),
            )
        project_id = uuid.uuid4().hex[:12]
        project = Project(
            id=project_id,
            original_filename=Path(original_filename).name,
            source_file=f"source{suffix}",
            edit=EditState(trim_silences=settings.trim_silences),
        )
        self.project_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.save(project)
        return project

    def save(self, project: Project) -> Project:
        project.updated_at = time.time()
        with self.lock_for(project.id):
            self._ensure_live(project.id)
            _atomic_write_json(
                self.project_file(project.id),
                json.dumps(
                    {
                        **project.model_dump(mode="json"),
                        "undo_history": project.undo_history,
                        "redo_history": project.redo_history,
                        "edit_clients": project.edit_clients,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        return project

    def get(self, project_id: str) -> Project:
        path = self.project_file(project_id)
        if not path.is_file():
            raise ProjectNotFoundError(f"No project with id '{project_id}'.")
        try:
            return Project.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - corrupted state is a 404-ish
            raise ProjectNotFoundError(
                f"Project '{project_id}' could not be read.",
                detail=str(exc)[:400],
            ) from exc

    def exists(self, project_id: str) -> bool:
        return self.project_file(project_id).is_file()

    def list(self, limit: int = 50) -> list[Project]:
        projects: list[Project] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not (directory / "project.json").is_file():
                continue
            try:
                projects.append(self.get(directory.name))
            except ProjectNotFoundError:
                continue
        projects.sort(key=lambda item: item.created_at, reverse=True)
        return projects[:limit]

    def delete(self, project_id: str) -> None:
        if not self.exists(project_id):
            raise ProjectNotFoundError(f"No project with id '{project_id}'.")
        with self.lock_for(project_id):
            tombstones = self.root / ".deleted"
            tombstones.mkdir(exist_ok=True)
            (tombstones / project_id).touch()
            shutil.rmtree(self.project_dir(project_id))

    # -------------------------------------------------------------- job files
    def save_job(self, job: Job) -> Job:
        job.updated_at = time.time()
        with self.lock_for(job.project_id):
            _atomic_write_json(
                self.jobs_dir(job.project_id) / f"{job.id}.json",
                job.model_dump_json(indent=2),
            )
        return job

    def get_job(self, project_id: str, job_id: str) -> Job:
        _validate_id(job_id, "job")
        path = self.jobs_dir(project_id) / f"{job_id}.json"
        if not path.is_file():
            raise JobNotFoundError(f"No job with id '{job_id}'.")
        try:
            return Job.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise JobNotFoundError(f"Job '{job_id}' could not be read.") from exc

    def find_job(self, job_id: str) -> Job:
        _validate_id(job_id, "job")
        for directory in self.root.iterdir():
            candidate = directory / "jobs" / f"{job_id}.json"
            if candidate.is_file():
                return self.get_job(directory.name, job_id)
        raise JobNotFoundError(f"No job with id '{job_id}'.")


store = ProjectStore()
