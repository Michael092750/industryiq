"""IngestScheduler: a dependency-free background loop that runs the job.

A single asyncio task polls the job config every ``poll_seconds``; when the job
is enabled, has a path, and its fixed interval has elapsed since the last run, it
runs :meth:`IngestionService.run_once` in a worker thread (the ingest is blocking
-- file IO, embeddings, DB). The loop is resilient: a failed run is swallowed so
the schedule survives, and ``run_once`` is itself non-overlapping (lock-guarded).

Timing uses a monotonic clock and an in-process "last run" marker, so on startup
(or after a restart/redeploy) an enabled job runs on its first eligible tick.
That re-scan is cheap and safe -- unchanged files are hash-skipped, so it never
duplicates content; it just picks up anything added while the service was down.

Lifecycle is owned by the FastAPI app (:mod:`industryiq.api.app`): ``start`` on
startup, ``stop`` on shutdown. With the single uvicorn worker the app runs, there
is exactly one scheduler; multiple workers would each schedule independently.
"""

import asyncio
import time
from collections.abc import Callable

from industryiq.core.ingestion.service import IngestionService


class IngestScheduler:
    """Polls the job config and triggers ``run_once`` on its fixed interval."""

    def __init__(
        self,
        service: IngestionService,
        *,
        poll_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._service = service
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._last_run: float | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Launch the background loop (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to unwind."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run(self) -> None:
        """The poll loop. Runs until cancelled."""
        while True:
            await asyncio.sleep(self._poll_seconds)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 -- a failed run must not kill the schedule
                pass

    async def tick(self) -> bool:
        """Run the job if it is due. Returns whether a run was triggered."""
        config = self._service.config()
        if not config.enabled or not config.path:
            return False
        if not self.due(config.interval_minutes):
            return False
        await asyncio.to_thread(self._service.run_once)
        self._last_run = self._clock()
        return True

    def due(self, interval_minutes: int) -> bool:
        """Whether ``interval_minutes`` have elapsed since the last run (or never ran)."""
        if self._last_run is None:
            return True
        return self._clock() - self._last_run >= interval_minutes * 60
