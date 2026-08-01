"""HTTP routes for the multi-agent orchestrator (Option B vs Option C).

A thin layer over the planner + executors: plan the request into subtasks, run
them either in-process (Option B, ``executor="local"``) or across the distributed
worker fleet (Option C, ``executor="distributed"``), and return the synthesized,
cited answer. The orchestration logic lives in ``industryiq.core.agents``; nothing
here but validate -> plan -> run -> serialize.
"""

from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from industryiq.api.deps import (
    build_local_executor,
    get_capability_registry,
    get_current_user,
    get_planner,
    get_supervisor,
)
from industryiq.core.agents.capabilities import WorkerCrash
from industryiq.core.auth import User

CurrentUser = Annotated[User, Depends(get_current_user)]

router = APIRouter(prefix="/agents", tags=["agents"])


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    executor: Literal["local", "distributed"] = "distributed"
    # Demo only: stage a mid-run crash. For "local" it makes the run raise -> the
    # route returns a "run lost" result (Option B has no recovery). For "distributed"
    # the crash is staged by running a worker with AGENT_FAILURE_MODE=crash_once.
    inject_failure: bool = False


class RunResponse(BaseModel):
    run_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]]
    completed: list[str]
    failed: list[str]


@router.post("/run")
def run_agents(req: RunRequest, _user: CurrentUser) -> RunResponse:
    """Plan ``question`` into subtasks and run it on the chosen executor."""
    run_id = uuid4().hex
    registry = get_capability_registry()
    catalog = {name: capability.description for name, capability in registry.items()}
    plan = get_planner().plan(run_id, req.question, catalog)

    if req.executor == "local":
        try:
            result = build_local_executor(inject_failure=req.inject_failure).run(plan)
        except WorkerCrash:
            # Option B has no recovery: a crashed node loses the whole run.
            return RunResponse(
                run_id=run_id,
                question=req.question,
                answer="[RUN LOST] a worker crashed mid-run; Option B has no recovery.",
                sources=[],
                completed=[],
                failed=[node.node_id for node in plan.nodes],
            )
    else:
        result = get_supervisor().run(plan)

    return RunResponse(
        run_id=result.run_id,
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        completed=list(result.completed),
        failed=list(result.failed),
    )
