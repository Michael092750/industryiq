"""End-to-end wiring smoke test: deps build working orchestrator objects offline.

Uses the default offline providers (RAG_PROVIDER unset -> FakeLLM/FakeEmbedder,
REDIS_URL unset -> in-memory queue/blackboard/ledger), so it needs no network. The
FakeLLM never returns JSON, so the planner exercises its single-node fallback --
which is enough to prove the registry -> planner -> executor chain is wired.
"""

import industryiq.api.deps as deps


def _clear_caches() -> None:
    for factory in (
        deps.get_redis,
        deps.get_blackboard,
        deps.get_task_queue,
        deps.get_run_ledger,
        deps.get_retrieval_service,
        deps.get_session_documents,
        deps.get_capability_registry,
        deps.get_planner,
        deps.get_supervisor,
        deps.get_turn_orchestrator,
    ):
        factory.cache_clear()


def _plan_for(question: str) -> object:
    registry = deps.get_capability_registry()
    catalog = {name: cap.description for name, cap in registry.items()}
    return deps.get_planner().plan("run-test", question, catalog)


def test_registry_and_planner_wire_up() -> None:
    _clear_caches()
    try:
        registry = deps.get_capability_registry()
        assert "industry_analysis" in registry
        plan = deps.get_planner().plan(
            "r", "compare AI and finance", {n: c.description for n, c in registry.items()}
        )
        assert plan.nodes  # fell back to a single node (FakeLLM output isn't JSON)
    finally:
        _clear_caches()


def test_local_executor_runs_offline() -> None:
    _clear_caches()
    try:
        result = deps.build_local_executor().run(_plan_for("AI market size"))
        assert result.completed
        assert result.answer
    finally:
        _clear_caches()


def test_distributed_executor_runs_offline() -> None:
    _clear_caches()
    try:
        plan = _plan_for("AI market size")
        worker = deps.build_worker("w1")
        result = deps.get_supervisor().run(plan, on_poll=worker.run_once)
        assert result.completed
    finally:
        _clear_caches()
