"""Application settings, loaded from environment variables.

Kept deliberately tiny for now; more settings (database URL, provider choice)
are added as later phases need them.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration."""

    # Secret required to access debug endpoints. When None, debug endpoints are
    # disabled entirely (they respond 404). Set DEBUG_API_KEY to enable them.
    debug_api_key: str | None = None


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh each call)."""
    return Settings(debug_api_key=os.getenv("DEBUG_API_KEY"))
