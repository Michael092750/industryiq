"""Benchmark the multi-agent chat service's *tiering* — with answer quality.

Companion to ``run_chat_benchmark.py`` (which measures one-shot RAG answer quality).
This one measures the thing the multi-agent integration adds: **does a turn route to
the right tier, and — for complex turns — how wide does the planner fan out, at what
latency?** Plus, where a query has verified corpus facts (``gold_needles``), it grades
the final answer with the same LLM-as-judge (``judge.py``).

For each query in ``agent_queries.json`` it runs the *real* components the app wires
(``deps``): the 3-way ``LlmRouter``, then the tier the router chose —

* ``simple``   -> a plain LLM reply,
* ``retrieve`` -> ``RetrievalService.gather`` + grounded answer (today's RAG),
* ``complex``  -> ``LlmPlanner`` -> ``LocalExecutor`` fan-out -> streamed ``Synthesizer``.

and records: **routed_ok** (actual tier == ``expected_tier``), the plan's **fan_out**
(node count) on complex turns, per-tier **latency**, and the judge's **correctness**
where graded. The headline is routing accuracy — especially complex recall (how many
of the labelled-complex queries the router actually sent to the planner).

Capabilities: the planner sees ``industry_analysis`` (corpus RAG) and — when
``ANTHROPIC_API_KEY`` is set — ``web_search`` (Anthropic's server-side web search), so
a complex query can fan out across tools.

Provider / backend / cost: same as ``run_chat_benchmark`` — ``RAG_PROVIDER=anthropic``
(or ``bedrock``) with the embedder that populated the store, a live ``--backend``
(pgvector default / milvus), and real tokens (router + planner + per-subtask + synthesis
+ optional judge per query). Web search and the judge always call the Anthropic API and
need ``ANTHROPIC_API_KEY``.

Usage
-----
    python benchmarks/run_agent_benchmark.py
    python benchmarks/run_agent_benchmark.py --limit 5 --no-judge
    python benchmarks/run_agent_benchmark.py --out agents.json --label baseline
"""

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import judge as judge_lib
import metrics
import textmatch
from run_chat_benchmark import build_providers, build_store, count_chunks

from industryiq.config import Settings, get_settings
from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.capabilities import IndustryAnalysisCapability, WebSearchCapability
from industryiq.core.agents.executor_local import LocalExecutor
from industryiq.core.agents.models import Plan
from industryiq.core.agents.planner import LlmPlanner
from industryiq.core.agents.ports import Capability
from industryiq.core.agents.synthesis import Synthesizer
from industryiq.core.chat.adapters.routing import LlmRouter
from industryiq.core.chat.models import RouteDecision
from industryiq.core.generation import GenerativeLLM, generate_answer
from industryiq.core.retrieval import (
    FixedStrategyRouter,
    LlmQueryRewriter,
    NoOpExpander,
    RetrievalService,
    Retriever,
    ThresholdFilter,
)
from industryiq.core.vectorstore import Hit

HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = HERE / "agent_queries.json"

TIERS = ("simple", "retrieve", "complex")


# --------------------------------------------------------------------------- #
# Component wiring (mirrors api/deps, minus the offline `fake` provider)
# --------------------------------------------------------------------------- #
@dataclass
class Components:
    router: LlmRouter
    planner: LlmPlanner
    registry: Mapping[str, Capability]
    executor: LocalExecutor
    synthesizer: Synthesizer
    retrieval: RetrievalService
    llm: GenerativeLLM


def build_components(settings: Settings, backend: str, k: int, max_web_searches: int) -> Components:
    """Build the real router + planner + registry + executor the app serves."""
    embedder, llm = build_providers(settings)
    store = build_store(settings, backend, embedder.dim)
    retrieval = RetrievalService(
        retriever=Retriever(embedder, store, min_chunk_chars=settings.retrieval_min_chunk_chars),
        rewriter=LlmQueryRewriter(llm),
        relevance_filter=ThresholdFilter.from_settings(
            settings.chat_relevance_threshold,
            bm25=settings.chat_bm25_threshold,
            normalized=settings.chat_normalized_threshold,
        ),
        strategy_router=FixedStrategyRouter(),
        expander=NoOpExpander(),
    )
    registry: dict[str, Capability] = {
        "industry_analysis": IndustryAnalysisCapability(retrieval, llm, k=k),
    }
    if settings.anthropic_api_key:
        registry["web_search"] = WebSearchCapability(
            model_id=settings.anthropic_llm_model_id,
            api_key=settings.anthropic_api_key,
            max_searches=max_web_searches,
        )
    return Components(
        router=LlmRouter(llm, settings.chat_kb_description),
        planner=LlmPlanner(llm),
        registry=registry,
        executor=LocalExecutor(registry, InMemoryBlackboard(), Synthesizer(llm)),
        synthesizer=Synthesizer(llm),
        retrieval=retrieval,
        llm=llm,
    )


def tier_of(decision: RouteDecision) -> str:
    """Map a router verdict to a tier label."""
    if decision.needs_planning:
        return "complex"
    return "retrieve" if decision.should_retrieve else "simple"


# --------------------------------------------------------------------------- #
# Answering (one path per tier), returning answer + grounding + fan-out
# --------------------------------------------------------------------------- #
def _hits_from_sources(sources: list[dict[str, Any]]) -> list[Hit]:
    return [
        Hit(id=str(s.get("source") or ""), score=0.0, metadata={"source": s.get("source")})
        for s in sources
    ]


def answer(components: Components, tier: str, query: str, k: int) -> tuple[str, list[Hit], int]:
    """Answer ``query`` on ``tier``; return (answer, hits-for-judge, fan_out).

    ``fan_out`` is the plan's node count on the complex tier, else 0.
    """
    if tier == "complex":
        plan: Plan = components.planner.plan(
            f"bench-{time.time_ns()}",
            query,
            {name: cap.description for name, cap in components.registry.items()},
        )
        results = components.executor.execute(plan)
        run_result = components.synthesizer.synthesize(plan, results)
        return run_result.answer, _hits_from_sources(run_result.sources), len(plan.nodes)
    if tier == "retrieve":
        result = components.retrieval.gather("bench", query, [], k)
        return generate_answer(query, result.hits, components.llm), result.hits, 0
    return components.llm.generate(query), [], 0  # simple: plain LLM reply


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    id: str
    expected_tier: str
    actual_tier: str
    routed_ok: bool
    fan_out: int
    route_ms: float
    answer_ms: float
    rag_hit: bool | None
    verdict: judge_lib.JudgeVerdict | None
    error: str | None


def evaluate(
    components: Components,
    judge: judge_lib.JudgeLLM | None,
    queries: list[dict[str, Any]],
    k: int,
) -> tuple[list[Record], list[dict[str, Any]]]:
    records: list[Record] = []
    rows: list[dict[str, Any]] = []
    for q in queries:
        expected = q["expected_tier"]
        error: str | None = None
        actual = ""
        fan_out = 0
        route_ms = answer_ms = 0.0
        hits: list[Hit] = []
        text = ""
        try:
            t0 = time.perf_counter()
            decision = components.router.route([], q["query"])
            route_ms = (time.perf_counter() - t0) * 1000
            actual = tier_of(decision)
            t1 = time.perf_counter()
            text, hits, fan_out = answer(components, actual, q["query"], k)
            answer_ms = (time.perf_counter() - t1) * 1000
        except Exception as exc:  # noqa: BLE001 -- record and keep the paid run going
            error = f"{type(exc).__name__}: {exc}"

        routed_ok = error is None and actual == expected
        needles = q.get("gold_needles")
        rag_hit: bool | None = None
        if needles is not None and error is None:
            rag_hit = any(textmatch.contains_any(h.metadata.get("text", ""), needles) for h in hits)

        verdict: judge_lib.JudgeVerdict | None = None
        if judge is not None and needles and error is None:
            try:
                verdict = judge.score(q["query"], " / ".join(needles), hits, text)
            except Exception as exc:  # noqa: BLE001
                error = f"judge: {type(exc).__name__}: {exc}"

        records.append(
            Record(
                q["id"],
                expected,
                actual,
                routed_ok,
                fan_out,
                route_ms,
                answer_ms,
                rag_hit,
                verdict,
                error,
            )
        )
        row: dict[str, Any] = {
            "id": q["id"],
            "expected_tier": expected,
            "actual_tier": actual,
            "routed_ok": routed_ok,
            "fan_out": fan_out,
            "route_ms": round(route_ms, 1),
            "answer_ms": round(answer_ms, 1),
        }
        if rag_hit is not None:
            row["rag_hit"] = rag_hit
        if verdict is not None:
            row["correctness"] = verdict.correctness
        if error is not None:
            row["error"] = error
        rows.append(row)
    return records, rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: list[Record]) -> dict[str, Any]:
    ok = [r for r in records if r.error is None]
    summary: dict[str, Any] = {
        "queries": len(records),
        "errors": sum(1 for r in records if r.error),
        "routing_accuracy": round(_mean([1.0 if r.routed_ok else 0.0 for r in ok]), 3),
    }
    # Per-expected-tier accuracy + the complex recall headline.
    per_tier: dict[str, Any] = {}
    for tier in TIERS:
        group = [r for r in ok if r.expected_tier == tier]
        if group:
            per_tier[tier] = {
                "n": len(group),
                "accuracy": round(_mean([1.0 if r.routed_ok else 0.0 for r in group]), 3),
            }
    summary["per_expected_tier"] = per_tier
    summary["complex_recall"] = per_tier.get("complex", {}).get("accuracy")
    # Fan-out on turns that actually ran the planner.
    fan = [r.fan_out for r in ok if r.actual_tier == "complex"]
    summary["complex_fan_out"] = _round_stats(metrics.summarize([float(v) for v in fan]))
    # Latency, split by the tier that actually ran.
    summary["route_ms"] = _round_stats(metrics.summarize([r.route_ms for r in ok]))
    for tier in TIERS:
        lat = [r.answer_ms for r in ok if r.actual_tier == tier]
        if lat:
            summary[f"answer_ms_{tier}"] = _round_stats(metrics.summarize(lat))
    # Judge (where graded).
    judged = [r.verdict for r in records if r.verdict is not None]
    if judged:
        summary["judged"] = len(judged)
        summary["correctness"] = round(_mean([v.correctness for v in judged]), 3)
        summary["rag_hit_rate"] = round(
            _mean([1.0 if r.rag_hit else 0.0 for r in records if r.rag_hit is not None]), 3
        )
    return summary


def _round_stats(stats: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 1) for key, value in stats.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--backend", choices=["pgvector", "milvus"], default="pgvector")
    parser.add_argument("--provider", choices=["anthropic", "bedrock"], default=None)
    parser.add_argument("--k", type=int, default=None, help="Top-k for the retrieve tier.")
    parser.add_argument(
        "--max-web-searches", type=int, default=None, help="Cap web_search uses per call."
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM judge (routing/latency only)."
    )
    parser.add_argument("--judge-model", default=judge_lib.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = Settings(**{**settings.__dict__, "provider": args.provider})
    if args.backend == "pgvector" and not settings.database_url:
        raise SystemExit("DATABASE_URL is not set (the live Postgres store to retrieve from).")

    k = args.k or settings.chat_retrieval_k
    max_web = args.max_web_searches or settings.agent_web_search_max_uses
    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"][: args.limit] if args.limit is not None else spec["queries"]

    components = build_components(settings, args.backend, k, max_web)
    judge = (
        None
        if args.no_judge
        else judge_lib.JudgeLLM(model_id=args.judge_model, api_key=settings.anthropic_api_key)
    )

    config: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "backend": args.backend,
        "provider": settings.provider,
        "chat_model": (
            settings.bedrock_llm_model_id
            if settings.provider == "bedrock"
            else settings.anthropic_llm_model_id
        ),
        "judge_model": None if judge is None else judge.model_id,
        "capabilities": sorted(components.registry),
        "k": k,
        "max_web_searches": max_web,
        "queries_file": args.queries.name,
        "n_queries": len(queries),
        "n_chunks": count_chunks(settings.database_url) if settings.database_url else None,
    }
    print("SETUP: " + json.dumps(config, ensure_ascii=False))

    records, rows = evaluate(components, judge, queries, k)
    summary = summarize(records)

    if args.out:
        args.out.write_text(
            json.dumps(
                {"config": config, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    print(f"\n{'=' * 72}\nAGENT TIERING\n{'=' * 72}")
    for row in rows:
        print("  " + json.dumps(row, ensure_ascii=False))
    print("  " + "-" * 68)
    print("  SUMMARY: " + json.dumps(summary, ensure_ascii=False))
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
