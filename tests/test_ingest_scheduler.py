"""Unit tests for IngestScheduler's "is it due" decision and tick gating.

No real sleeps and no pytest-asyncio: a fake clock drives the interval math, and
``tick`` is driven with ``asyncio.run`` (it offloads ``run_once`` to a worker
thread internally).
"""

import asyncio

from industryiq.core.ingestion import IngestScheduler
from industryiq.core.ingestion.models import IngestJobConfig, IngestRunResult


class _FakeService:
    """Stands in for IngestionService: canned config, records run_once calls."""

    def __init__(self, config: IngestJobConfig) -> None:
        self._config = config
        self.runs = 0

    def config(self) -> IngestJobConfig:
        return self._config

    def run_once(self, path: object = None) -> IngestRunResult:
        self.runs += 1
        return IngestRunResult()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _enabled() -> IngestJobConfig:
    return IngestJobConfig(path="/data", interval_minutes=10, enabled=True)


def test_due_first_time_when_never_run() -> None:
    scheduler = IngestScheduler(_FakeService(_enabled()), clock=_Clock())
    assert scheduler.due(10) is True


def test_tick_runs_when_due_then_waits_for_interval() -> None:
    clock = _Clock()
    service = _FakeService(_enabled())
    scheduler = IngestScheduler(service, clock=clock)

    assert asyncio.run(scheduler.tick()) is True  # first eligible tick runs
    assert service.runs == 1

    # Immediately after, the interval has not elapsed -> no run.
    assert asyncio.run(scheduler.tick()) is False
    assert service.runs == 1

    # Advance past the 10-minute interval -> runs again.
    clock.now += 10 * 60
    assert asyncio.run(scheduler.tick()) is True
    assert service.runs == 2


def test_tick_skips_when_disabled() -> None:
    config = IngestJobConfig(path="/data", interval_minutes=10, enabled=False)
    service = _FakeService(config)
    scheduler = IngestScheduler(service, clock=_Clock())
    assert asyncio.run(scheduler.tick()) is False
    assert service.runs == 0


def test_tick_skips_when_no_path() -> None:
    config = IngestJobConfig(path=None, interval_minutes=10, enabled=True)
    service = _FakeService(config)
    scheduler = IngestScheduler(service, clock=_Clock())
    assert asyncio.run(scheduler.tick()) is False
    assert service.runs == 0


def test_start_runs_the_loop_then_stop_cancels_it() -> None:
    service = _FakeService(_enabled())
    # Tiny poll interval so the loop reaches its first tick almost immediately.
    scheduler = IngestScheduler(service, poll_seconds=0.01, clock=_Clock())

    async def drive() -> None:
        scheduler.start()
        scheduler.start()  # idempotent: a second start is a no-op
        await asyncio.sleep(0.05)  # let the loop tick at least once
        await scheduler.stop()  # cancels the task cleanly
        await scheduler.stop()  # idempotent after stop

    asyncio.run(drive())
    assert service.runs >= 1
