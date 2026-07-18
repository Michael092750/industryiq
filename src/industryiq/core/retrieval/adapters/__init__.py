"""Concrete adapters implementing the retrieval ports.

These satisfy the ``Protocol``s in :mod:`industryiq.core.retrieval.ports` and are
the only retrieval modules that touch infrastructure (Redis) or providers (the
LLM). The service layer depends on the ports, never on anything here -- wiring
happens in :mod:`industryiq.api.deps`.
"""
