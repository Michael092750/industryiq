"""A tiny step timer for per-operation observability.

Used by :class:`ChatService` to record how long each phase of a turn takes. The
clock is injectable (defaulting to :func:`time.perf_counter`) so tests can feed a
deterministic clock and assert exact durations.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class StepTimer:
    """Accumulate named step durations, in milliseconds."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the wrapped block and store it under ``name`` (milliseconds).

        Records even if the block raises, so a failed step still reports how long
        it ran before failing.
        """
        start = self._clock()
        try:
            yield
        finally:
            self.timings_ms[name] = round((self._clock() - start) * 1000, 3)
