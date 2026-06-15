"""Concrete adapters implementing the chat ports.

These satisfy the ``Protocol``s in :mod:`ragproject.core.chat.ports` and are the
only chat modules that touch infrastructure (Postgres) or providers (the LLM).
The domain (``models``, ``ports``, ``prompting``) and application (``service``)
layers depend on the ports, never on anything here -- wiring happens in
:mod:`ragproject.api.deps`.
"""
