"""Redis connection factory -- the seam every Redis-backed adapter builds on.

Redis is the project's *hot* tier: fast, shared across processes, and ephemeral
by design. It is where agent working-memory and cross-agent context will live,
distinct from the durable knowledge base (pgvector/Milvus) and the system of
record (Postgres). This module owns only the connection; the stores that use it
land in later steps.

Kept tiny on purpose:

* :func:`build_redis_client` turns a URL into a client. ``decode_responses=True``
  so values come back as ``str`` (a :class:`Redis[str]`), which is what the
  context stores want -- no manual ``bytes.decode`` at every call site.
* :func:`ping` is a liveness probe that answers ``True``/``False`` instead of
  raising, so a health check can report "configured but unreachable" cleanly.

The client is lazy: :meth:`Redis.from_url` opens no socket until the first
command, so constructing one is cheap and cannot fail on an unreachable server --
only :func:`ping` (or real use) touches the network.
"""

from redis import Redis
from redis.exceptions import RedisError


def build_redis_client(url: str) -> Redis:
    """Build a Redis client from a ``redis://`` URL.

    Connection is lazy (no socket until the first command). ``decode_responses``
    makes reads return ``str`` rather than ``bytes``.
    """
    return Redis.from_url(url, decode_responses=True)


def ping(client: Redis) -> bool:
    """Return whether the server answers PING; ``False`` on any connection error."""
    try:
        return bool(client.ping())
    except RedisError:
        return False
