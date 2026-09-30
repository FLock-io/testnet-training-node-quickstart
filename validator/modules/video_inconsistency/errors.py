from __future__ import annotations


class VideoSubmissionError(Exception):
    """Raised when a trainer's detector submission is invalid or unrunnable.

    These are the *submitter's* fault (a missing/broken adapter, a model that
    fails to load, exceeds its memory budget, hangs, or crashes). They are scored
    as an invalid submission (score 0) and must NOT crash the long-running
    validator. Genuine infrastructure problems (a broken validation package, a
    disk or network failure) deliberately do NOT use this type, so they propagate
    to the runner and are retried or re-queued instead of zeroing the trainer.

    ``failure_mode`` is a short, stable, machine-readable tag surfaced in the
    metrics' diagnostics.

    ``fatal`` distinguishes errors that end the whole evaluation (the sandbox is
    gone: timeout, memory breach, crash, load failure) from per-clip errors
    (``detect`` raised or returned malformed output while the sandbox is still
    healthy), which the validator retries and then scores as an empty answer for
    that clip.
    """

    def __init__(
        self,
        message: str,
        failure_mode: str = "submission_error",
        *,
        fatal: bool = True,
    ):
        super().__init__(message)
        self.failure_mode = failure_mode
        self.fatal = fatal
