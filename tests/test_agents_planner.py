"""Tests for the LLM planner: JSON parsing, validation, and safe fallback."""

import json

from industryiq.core.agents.planner import LlmPlanner
from industryiq.core.generation import FakeLLM

CAPS = {"industry_analysis": "analyse one industry from the report corpus"}


def _plan_json(*nodes: dict[str, object]) -> str:
    return json.dumps({"nodes": list(nodes)})


def test_planner_parses_a_fan_out_plan() -> None:
    raw = _plan_json(
        {"node_id": "n1", "capability": "industry_analysis", "inputs": {"industry": "AI"}},
        {"node_id": "n2", "capability": "industry_analysis", "inputs": {"industry": "finance"}},
    )
    plan = LlmPlanner(FakeLLM(raw)).plan("run1", "compare AI and finance", CAPS)
    assert plan.run_id == "run1"
    assert len(plan.nodes) == 2
    assert {node.inputs["industry"] for node in plan.nodes} == {"AI", "finance"}


def test_planner_extracts_json_from_a_code_fence() -> None:
    raw = (
        "Here is the plan:\n```json\n"
        + _plan_json({"node_id": "n1", "capability": "industry_analysis", "inputs": {}})
        + "\n```"
    )
    plan = LlmPlanner(FakeLLM(raw)).plan("r", "q", CAPS)
    assert len(plan.nodes) == 1


def test_planner_preserves_dependency_edges() -> None:
    raw = _plan_json(
        {"node_id": "n1", "capability": "industry_analysis", "inputs": {}},
        {"node_id": "n2", "capability": "industry_analysis", "inputs": {}, "depends_on": ["n1"]},
    )
    plan = LlmPlanner(FakeLLM(raw)).plan("r", "q", CAPS)
    node2 = next(node for node in plan.nodes if node.node_id == "n2")
    assert node2.depends_on == ("n1",)


def test_planner_falls_back_on_non_json() -> None:
    plan = LlmPlanner(FakeLLM("I cannot help with that.")).plan("r", "q about AI", CAPS)
    assert len(plan.nodes) == 1
    assert plan.nodes[0].capability == "industry_analysis"
    assert plan.nodes[0].inputs == {"question": "q about AI"}


def test_planner_falls_back_on_unknown_capability() -> None:
    raw = _plan_json({"node_id": "n1", "capability": "web_search", "inputs": {}})
    plan = LlmPlanner(FakeLLM(raw)).plan("r", "q", CAPS)
    assert plan.nodes[0].capability == "industry_analysis"  # rejected -> fell back


def test_planner_falls_back_on_a_cycle() -> None:
    raw = _plan_json(
        {"node_id": "n1", "capability": "industry_analysis", "inputs": {}, "depends_on": ["n2"]},
        {"node_id": "n2", "capability": "industry_analysis", "inputs": {}, "depends_on": ["n1"]},
    )
    plan = LlmPlanner(FakeLLM(raw)).plan("r", "q", CAPS)
    assert len(plan.nodes) == 1  # cyclic plan rejected -> fell back
