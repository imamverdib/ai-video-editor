"""Cooperative cancellation shared by workers, subprocesses and retries."""

from contextvars import ContextVar
from threading import Event

from .errors import EditorError


class JobCancelled(EditorError):
    code = "job_cancelled"
    status_code = 409


cancel_event: ContextVar[Event | None] = ContextVar("cancel_event", default=None)


def check_cancelled() -> None:
    event = cancel_event.get()
    if event is not None and event.is_set():
        raise JobCancelled("Əməliyyat dayandırıldı.")


def interruptible_sleep(seconds: float) -> None:
    event = cancel_event.get() or Event()
    if event.wait(max(0, seconds)):
        check_cancelled()
