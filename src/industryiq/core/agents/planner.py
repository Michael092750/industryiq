"""Planner: decompose a question into a :class:`Plan` over the capabilities.

The planner *is* the dispatch intelligence: it reads the request plus each
capability's prescriptive description and emits a DAG of subtasks (which capability,
what inputs, what it depends on). We let the LLM produce it as JSON and validate the
result hard -- an unparseable, unknown-capability, or cyclic plan falls back to a
single node, so a run never hard-fails on the planner. Opus 4.8 under-reaches on
decomposition, so the prompt explicitly asks it to fan out per distinct entity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from industryiq.core.agents.models import Plan, PlanNode
from industryiq.core.agents.ports import Planner
from industryiq.core.generation import LLM

_PROMPT = """You are a planning agent. Break the user's request into subtasks, each \
dispatched to exactly one capability below. FAN OUT into parallel subtasks -- one per \
distinct entity/industry/company the request names -- because running them in parallel \
is the whole point; only add a `depends_on` edge when a subtask truly needs another's \
result first.

Capabilities:
{capabilities}

Return ONLY a JSON object of this exact shape (no prose, no code fence):
{{"nodes": [{{"node_id": "n1", "capability": "<one of the names above>", \
"inputs": {{...}}, "depends_on": []}}]}}
- node_id: unique per node.
- depends_on: list of node_ids that must finish before this one (usually []).
- inputs: the argument dict the chosen capability expects.

Request: {question}
JSON:"""


class LlmPlanner(Planner):
    """Ask the LLM for a plan; validate it; fall back to a single node on any doubt."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def plan(self, run_id: str, question: str, capabilities: Mapping[str, str]) -> Plan:
        if not capabilities:
            raise ValueError("cannot plan with no capabilities")
        prompt = _PROMPT.format(
            capabilities="\n".join(f"- {name}: {desc}" for name, desc in capabilities.items()),
            question=question,
        )
        nodes = _parse_nodes(self._llm.generate(prompt), capabilities)
        if nodes is None:
            nodes = _fallback_nodes(question, capabilities)
        return Plan(run_id=run_id, question=question, nodes=tuple(nodes))


def _fallback_nodes(question: str, capabilities: Mapping[str, str]) -> list[PlanNode]:
    """A safe one-node plan: run the first capability on the whole question."""
    first = next(iter(capabilities))
    return [PlanNode(node_id="n1", capability=first, inputs={"question": question})]


def _parse_nodes(raw: str, capabilities: Mapping[str, str]) -> list[PlanNode] | None:
    """Parse + validate the LLM's JSON plan, or ``None`` if it is not trustworthy."""
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None

    nodes: list[PlanNode] = []
    ids: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            return None
        node_id = item.get("node_id")
        capability = item.get("capability")
        inputs = item.get("inputs", {})
        depends_on = item.get("depends_on", [])
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            return None
        if not isinstance(capability, str) or capability not in capabilities:
            return None
        if not isinstance(inputs, dict):
            return None
        if not isinstance(depends_on, list) or not all(isinstance(dep, str) for dep in depends_on):
            return None
        ids.add(node_id)
        nodes.append(
            PlanNode(
                node_id=node_id,
                capability=capability,
                inputs=inputs,
                depends_on=tuple(depends_on),
            )
        )

    # Every dependency must name a real node, and the graph must be schedulable
    # (acyclic) -- otherwise the supervisor would deadlock waiting on it.
    if any(dep not in ids for node in nodes for dep in node.depends_on):
        return None
    if not _is_schedulable(nodes):
        return None
    return nodes


def _is_schedulable(nodes: list[PlanNode]) -> bool:
    """True if the DAG can be fully ordered -- i.e. it has no cycle."""
    done: set[str] = set()
    remaining = list(nodes)
    progressed = True
    while remaining and progressed:
        progressed = False
        still: list[PlanNode] = []
        for node in remaining:
            if all(dep in done for dep in node.depends_on):
                done.add(node.node_id)
                progressed = True
            else:
                still.append(node)
        remaining = still
    return not remaining


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction: whole string, fenced block, or first {...} span."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
