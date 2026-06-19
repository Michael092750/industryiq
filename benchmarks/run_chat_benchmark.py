"""Benchmark the chatbot's *answer generation* with an LLM-as-judge.

Companion to the retrieval benchmark (``run_benchmark.py``). That one asks "did
RAG surface the right chunks?"; this one asks the orthogonal question: **given the
chunks it retrieved, how good is the chatbot's answer?** For each question it runs
the **real** chat pipeline -- the same :class:`ChatService` the app wires up (route
-> rewrite -> retrieve -> filter -> generate) -- then has an *independent* judge
model (default Claude Opus 4.8 -- see ``judge.py``) score the answer **against the
retrieved context only**.

Why judge against the context, not the gold answer? Because the goal is to measure
the *chatbot*, not the retriever. The judge never sees the gold answer; it only
sees what RAG actually returned. So a retrieval miss does not penalize the chatbot
-- when the context lacks the answer, correctly saying so scores full marks. The
questions are the same gold set as the retrieval benchmark, chosen so retrieval
*usually* hits (reported as ``rag_hit_rate``), keeping most cases about extraction
and grounding rather than graceful refusal.

The judge scores two decoupled axes (see ``judge.py``): ``correctness`` against the
gold answer (end-to-end quality, the signal a retrieval change moves) and
``groundedness`` against the retrieved context (faithfulness).

A fixed yardstick for comparing techniques
------------------------------------------
The chat pipeline is ports-and-adapters, so techniques are swappable. The query
rewriter is hot-swappable from the command line: ``--rewriter NAME`` picks an entry
from the ``REWRITERS`` registry (default mirrors production), and the choice is
recorded in the run's config. Adopt a new technique by registering one factory and
rerunning -- no edits to the runner. (The router and relevance filter are likewise
configurable, via ``CHAT_ROUTER`` / ``CHAT_RELEVANCE_THRESHOLD``.) Because it drives
the real ``ChatService``, even backend changes not exposed as a flag are reflected
automatically.

Tag each run with ``--label``/``--out`` and diff the JSON. Read the diff on two
axes: ``correctness`` / ``rag_hit_rate`` / ``context_sufficient_rate`` rise when a
technique feeds the chatbot better context, while
``correctness_when_context_sufficient`` (the bot's extraction skill on the cases
where the answer *was* retrieved) and ``groundedness`` are intrinsic to the chat
model and should stay flat.

Conversations are kept in memory (never written to your database) and each question
runs single-turn, so this measures one-shot answering. Costs real tokens: one chat
turn plus one judge completion per question.

Provider
--------
The chat LLM + embedder follow ``RAG_PROVIDER`` (``anthropic`` or ``bedrock``),
exactly like the retrieval benchmark -- the embedder must match the one that
populated the pgvector table. The judge always calls the Anthropic API directly
and needs ``ANTHROPIC_API_KEY`` regardless of provider (that is the "other" API).

Usage
-----
    python benchmarks/run_chat_benchmark.py
    python benchmarks/run_chat_benchmark.py --limit 5
    python benchmarks/run_chat_benchmark.py --rewriter noop --label baseline --out baseline.json
    python benchmarks/run_chat_benchmark.py --rewriter llm  --label rewrite  --out rewrite.json
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import judge as judge_lib
import metrics

from ragproject.config import Settings, get_settings
from ragproject.core.chat import (
    AlwaysRetrieveRouter,
    ChatPolicy,
    ChatService,
    InMemoryConversationStore,
    LlmQueryRewriter,
    LlmRouter,
    NoOpQueryRewriter,
    QueryRewriter,
    RetrievalRouter,
    ThresholdFilter,
)
from ragproject.core.embeddings import Embedder
from ragproject.core.generation import GenerativeLLM
from ragproject.core.pgvectorstore import PgVectorStore
from ragproject.core.retrieval import Retriever
from ragproject.core.vectorstore import Hit

HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES = HERE / "queries.json"


# --------------------------------------------------------------------------- #
# Hot-swappable query rewriters (the "technique" axis)
# --------------------------------------------------------------------------- #
# ChatService depends on a QueryRewriter port, so swapping rewriting techniques is
# a composition choice. Register each technique here as name -> factory(llm); pick
# one with ``--rewriter`` and it is recorded in the run's config for diffing. The
# default (``llm``) mirrors production (``deps`` wires ``LlmQueryRewriter``). Add a
# technique by implementing a ``QueryRewriter`` adapter in the backend and adding
# one line here -- nothing else in the benchmark changes.
#
# The router and relevance filter are already swappable via ``CHAT_ROUTER`` and
# ``CHAT_RELEVANCE_THRESHOLD`` (``build_chat_service`` reads them from settings);
# follow this same registry pattern if you want non-settings variants of those.
#
# Note: ``LlmQueryRewriter`` only condenses *follow-ups* -- on the single-turn
# questions here it returns the question unchanged, so ``llm`` and ``noop`` score
# identically. To see a rewriter move the metrics in this benchmark, it must rewrite
# standalone queries (e.g. expansion / HyDE), which a new technique typically does.
REWRITERS: dict[str, Callable[[GenerativeLLM], QueryRewriter]] = {
    "llm": LlmQueryRewriter,  # production: condense follow-ups with the LLM
    "noop": lambda _llm: NoOpQueryRewriter(),  # baseline: pass the question through
}


# --------------------------------------------------------------------------- #
# Provider wiring (isolated composition root, mirrors api/deps.py)
# --------------------------------------------------------------------------- #
def build_providers(settings: Settings) -> tuple[Embedder, GenerativeLLM]:
    """The embedder + chat LLM for ``settings.provider`` -- the same real providers
    the app wires up (mirrors ``api/deps._build_ai_providers``), minus the offline
    ``fake`` option: this benchmark queries the live pgvector store and calls a real
    chat model, so the embedder must match the table and the LLM must really answer.
    """
    if settings.provider == "bedrock":
        from ragproject.core.bedrock import BedrockEmbedder, BedrockLLM

        return (
            BedrockEmbedder(model_id=settings.bedrock_embed_model_id, region=settings.aws_region),
            BedrockLLM(model_id=settings.bedrock_llm_model_id, region=settings.aws_region),
        )
    if settings.provider == "anthropic":
        from ragproject.core.anthropic_llm import AnthropicLLM
        from ragproject.core.local_embeddings import LocalEmbedder

        return LocalEmbedder(), AnthropicLLM(
            model_id=settings.anthropic_llm_model_id, api_key=settings.anthropic_api_key
        )
    raise SystemExit(
        f"provider {settings.provider!r} has no real chat LLM/embedder; this benchmark "
        "calls a real model and queries live pgvector. Set RAG_PROVIDER=anthropic (or bedrock)."
    )


def build_chat_service(
    settings: Settings, embedder: Embedder, llm: GenerativeLLM, k: int, rewriter: QueryRewriter
) -> ChatService:
    """The real ``ChatService`` the app serves (mirrors ``deps.get_chat_service``),
    against the live pgvector store -- so router/rewriter/filter changes you make to
    the backend show up here. ``rewriter`` is the hot-swappable technique under test
    (see ``REWRITERS``); the only other swap is an in-memory conversation store, so
    benchmark turns are never written to your database.
    """
    router: RetrievalRouter = (
        LlmRouter(llm, settings.chat_kb_description)
        if settings.chat_router == "llm"
        else AlwaysRetrieveRouter()
    )
    return ChatService(
        retriever=Retriever(embedder, PgVectorStore(settings.database_url, dim=embedder.dim)),
        router=router,
        rewriter=rewriter,
        llm=llm,
        store=InMemoryConversationStore(),
        relevance_filter=ThresholdFilter(settings.chat_relevance_threshold),
        policy=ChatPolicy(k=k, history_limit=settings.chat_history_turns),
    )


def chat_model_id(settings: Settings) -> str:
    """The chat model string in play, for the run's recorded config."""
    return (
        settings.bedrock_llm_model_id
        if settings.provider == "bedrock"
        else settings.anthropic_llm_model_id
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    """Per-query results: the chat answer's verdict (or an error), plus per-step
    timings. ``rag_hit`` records whether a gold needle actually landed in the
    retrieved context -- the premise check, kept separate from the judge's score."""

    id: str
    rag_hit: bool
    verdict: judge_lib.JudgeVerdict | None
    error: str | None
    retrieve_ms: float
    generate_ms: float
    judge_ms: float


def answer_question(
    service: ChatService, title: str, question: str
) -> tuple[list[Hit], str, float, float]:
    """Run one single-turn chat through the real pipeline. Returns the answer, the
    context it was grounded in (the filtered hits the judge will see), and the
    retrieve/generate step timings the service measured.
    """
    conversation = service.start(title)
    result = service.reply(conversation.id, question)
    timings = result.timings_ms
    return result.hits, result.answer, timings.get("retrieve", 0.0), timings.get("generate", 0.0)


def is_rag_hit(hits: list[Hit], needles: list[str]) -> bool:
    """True if any retrieved chunk's text contains any gold needle (case-insensitive)
    -- i.e. retrieval actually surfaced the answer for this query."""
    lowered = [n.lower() for n in needles]
    return any(n in hit.metadata.get("text", "").lower() for hit in hits for n in lowered)


def evaluate(
    service: ChatService,
    judge: judge_lib.JudgeLLM,
    queries: list[dict[str, Any]],
) -> tuple[list[Record], list[dict[str, Any]]]:
    """Answer and judge each query, returning raw records plus per-query display rows.

    Each query is wrapped so one judge failure (refusal, transport error) records an
    error and continues, rather than aborting a long, paid run.
    """
    records: list[Record] = []
    rows: list[dict[str, Any]] = []
    for q in queries:
        hits, answer, retrieve_ms, generate_ms = answer_question(service, q["id"], q["query"])
        rag_hit = is_rag_hit(hits, q["gold_needles"])
        # The gold needles are the verbatim answer facts -- use them as the
        # reference the judge grades correctness against.
        reference = " / ".join(q["gold_needles"])

        verdict: judge_lib.JudgeVerdict | None = None
        error: str | None = None
        judge_start = time.perf_counter()
        try:
            verdict = judge.score(q["query"], reference, hits, answer)
        except Exception as exc:  # noqa: BLE001 -- record and keep the paid run going
            error = f"{type(exc).__name__}: {exc}"
        judge_ms = (time.perf_counter() - judge_start) * 1000

        records.append(Record(q["id"], rag_hit, verdict, error, retrieve_ms, generate_ms, judge_ms))
        row: dict[str, Any] = {
            "id": q["id"],
            "rag_hit": rag_hit,
            "generate_ms": round(generate_ms, 2),
            "judge_ms": round(judge_ms, 2),
        }
        if verdict is not None:
            row.update(
                {
                    "correctness": verdict.correctness,
                    "groundedness": verdict.groundedness,
                    "citation": verdict.citation,
                    "context_sufficient": verdict.context_sufficient,
                    "rationale": verdict.rationale,
                }
            )
        else:
            row["error"] = error
        rows.append(row)
    return records, rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_stats(stats: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 2) for key, value in stats.items()}


def chat_summary(records: list[Record]) -> dict[str, Any]:
    scored = [r for r in records if r.verdict is not None]
    verdicts = [r.verdict for r in scored if r.verdict is not None]
    summary: dict[str, Any] = {
        "queries": len(records),
        "scored": len(scored),
        "errors": sum(1 for r in records if r.error),
        "rag_hit_rate": round(_mean([1.0 if r.rag_hit else 0.0 for r in records]), 3),
    }
    if verdicts:
        # End-to-end quality (vs the gold answer) -- the axis a retrieval change moves.
        summary["correctness"] = round(_mean([v.correctness for v in verdicts]), 3)
        summary["correctness_pass_rate@4"] = round(
            _mean([1.0 if v.correctness >= 4 else 0.0 for v in verdicts]), 3
        )
        # Faithfulness (vs retrieved context) and citation -- intrinsic chat-model traits.
        summary["groundedness"] = round(_mean([v.groundedness for v in verdicts]), 3)
        summary["citation"] = round(_mean([v.citation for v in verdicts]), 3)
        summary["context_sufficient_rate"] = round(
            _mean([1.0 if v.context_sufficient else 0.0 for v in verdicts]), 3
        )
        # The chatbot's intrinsic extraction skill: correctness on the cases where the
        # answer *was* retrieved, so retrieval noise is factored out. Should stay ~flat
        # across retrieval changes -- if it moves, the chat model/judge changed, not RAG.
        sufficient = [v.correctness for v in verdicts if v.context_sufficient]
        summary["correctness_when_context_sufficient"] = (
            round(_mean(sufficient), 3) if sufficient else None
        )
    summary["retrieve_ms"] = _round_stats(metrics.summarize([r.retrieve_ms for r in records]))
    summary["generate_ms"] = _round_stats(metrics.summarize([r.generate_ms for r in records]))
    summary["judge_ms"] = _round_stats(metrics.summarize([r.judge_ms for r in scored]))
    return summary


def print_section(title: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    for row in rows:
        print("  " + json.dumps(row, ensure_ascii=False))
    print("  " + "-" * 68)
    print("  SUMMARY: " + json.dumps(summary, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rewriter",
        choices=sorted(REWRITERS),
        default="llm",
        help="Which query-rewriting technique to wire into the chat pipeline (the thing under "
        "test). Default 'llm' mirrors production. Register new techniques in REWRITERS.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Free-text tag recorded in the output config, to name a run when comparing "
        "(e.g. 'baseline', 'hyde').",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--provider",
        choices=["anthropic", "bedrock"],
        default=None,
        help="Override RAG_PROVIDER for this run (must match the table's embedder).",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Override the chatbot's LLM model id (the model under test).",
    )
    parser.add_argument(
        "--judge-model",
        default=judge_lib.DEFAULT_JUDGE_MODEL,
        help="Judge model id (should differ from the chat model). Always via the Anthropic API.",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Top-k to retrieve (default: CHAT_RETRIEVAL_K)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write full results as JSON to this path."
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.chat_model:
        provider = args.provider or settings.provider
        key = "bedrock_llm_model_id" if provider == "bedrock" else "anthropic_llm_model_id"
        overrides[key] = args.chat_model
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})

    k = args.k or settings.chat_retrieval_k
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set (the live Postgres store to retrieve from).")

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = spec["queries"]
    if args.limit is not None:
        queries = queries[: args.limit]

    embedder, llm = build_providers(settings)
    rewriter = REWRITERS[args.rewriter](llm)
    service = build_chat_service(settings, embedder, llm, k, rewriter)
    judge = judge_lib.JudgeLLM(model_id=args.judge_model, api_key=settings.anthropic_api_key)

    print(
        f"provider={settings.provider}  rewriter={args.rewriter}  k={k}  "
        f"queries={len(queries)}  chat_model={chat_model_id(settings)}  "
        f"judge_model={judge.model_id}"
    )

    records, rows = evaluate(service, judge, queries)
    summary = chat_summary(records)
    print_section("CHAT (LLM-as-judge)", summary, rows)

    results: dict[str, Any] = {
        "config": {
            "provider": settings.provider,
            "rewriter": args.rewriter,
            "label": args.label,
            "k": k,
            "chat_model": chat_model_id(settings),
            "judge_model": judge.model_id,
        },
        "summary": summary,
        "rows": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
