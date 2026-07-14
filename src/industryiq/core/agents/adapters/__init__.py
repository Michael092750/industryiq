"""Concrete adapters for the agents ports (in-memory doubles + Redis backends).

The Redis adapters are imported only where they are wired
(:mod:`industryiq.api.deps`), to keep the package import light.
"""
