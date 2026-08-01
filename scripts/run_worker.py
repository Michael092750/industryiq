"""Run one Option-C agents worker process.

Usage:
    python scripts/run_worker.py [consumer_name]

Prerequisites:
    REDIS_URL          workers coordinate through Redis Streams (required for C)
    RAG_PROVIDER=anthropic + ANTHROPIC_API_KEY   for real subtask answers

Demo:
    AGENT_FAILURE_MODE=crash_once  makes this a "flaky" worker that fails the first
    attempt of each node -- start one flaky worker next to healthy ones to stage the
    kill-a-worker beat: a healthy worker reclaims the abandoned task and the run
    still completes.

Stop with Ctrl-C (SIGINT); the worker drains its current loop and exits.
"""

import os
import signal
import sys
import threading

from industryiq.api.deps import build_worker


def main() -> None:
    consumer = sys.argv[1] if len(sys.argv) > 1 else f"worker-{os.getpid()}"
    worker = build_worker(consumer)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    mode = os.getenv("AGENT_FAILURE_MODE", "off")
    print(f"[{consumer}] started (failure_mode={mode}); Ctrl-C to stop", flush=True)
    worker.run_forever(stop)
    print(f"[{consumer}] stopped", flush=True)


if __name__ == "__main__":
    main()
