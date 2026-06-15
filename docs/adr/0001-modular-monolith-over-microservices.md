# ADR 0001: Multi-round chat as a modular monolith, not a microservice

- Status: Accepted
- Date: 2026-06-14

## Context

The system needs multi-round (stateful) conversation on top of the existing
stateless RAG query/retrieval. One option considered was deploying the chatbot
as a separate microservice from retrieval/ingestion. The motivation for that
option was "modular, testable, decoupled" code — not any measured scaling need.

## Decision

Build chat as a **module inside the existing application** (`ragproject.core.chat`)
using a **ports-and-adapters** (hexagonal) design, rather than as a separate
deployable service.

- `ChatService` is high-level policy and depends only on abstractions:
  `RetrievalPort`, `QueryRewriter`, `ConversationStore` (and the existing `LLM`
  port). It contains no SQL, no prompt strings, and no provider calls.
- Concrete adapters (`Retriever`, `LlmQueryRewriter`, `PgConversationStore` /
  `InMemoryConversationStore`, Bedrock/Fake LLM) are selected and assembled in a
  single composition root, `api/deps.py`.

## Rationale

- **Decoupling is a code property, not a deployment property.** A Python
  `Protocol` boundary decouples more cleanly than an HTTP call, which adds a
  fragile wire contract, serialization, version skew, and partial-failure modes.
- **A chat↔retrieval network split is a bad seam.** Each turn must rewrite the
  question using history, then retrieve, then generate — tightly coupled steps.
  A network boundary there is a distributed monolith: all the runtime coupling,
  plus latency, minus safe cross-boundary refactoring.
- **Testability improves in-process.** `ChatService` is unit-tested end to end
  with in-memory fakes, no DB and no network (`tests/test_chat_service.py`). The
  same flow across services would need contract tests or docker-compose.
- **No operational driver exists.** Independent scaling, separate deploy
  cadence, distinct runtime profiles, and team ownership are the reasons to
  split a service; none currently apply.

## How SOLID shows up

- **S** — one responsibility per unit: `QueryRewriter` (condensing),
  `ConversationStore` (persistence), `prompting` (pure prompt text),
  `ChatService` (orchestration only).
- **O** — extend by adding adapters (e.g. a reranking `RetrievalPort` wrapper)
  without editing `ChatService`.
- **L** — `InMemory` / `Pg` stores and `Llm` / `NoOp` rewriters are
  interchangeable behind their ports; tests substitute fakes to prove it.
- **I** — `RetrievalPort` exposes only `retrieve`, narrower than the full
  `Retriever` class, so chat depends on nothing it does not use.
- **D** — the service depends on ports; concretions are chosen only in
  `api/deps.py`.

## Consequences

- Single build, deploy, and local-dev story; fast in-process tests.
- **Future extraction stays cheap:** to move retrieval into its own service
  later, implement `RetrievalPort` with an HTTP client adapter and swap it in
  the composition root — `ChatService` is untouched.

## Update — streaming added behind the predicted seam

Token streaming was later added exactly as this ADR anticipated: a new
`StreamingLLM` port (separate from `LLM`, per Interface Segregation), a
`GenerativeLLM` combination the providers implement, and a `ChatService.reply_stream`
generator that yields `StreamStart → StreamToken* → StreamEnd` events and persists
the turn once the stream completes. The SSE endpoint (`POST …/messages/stream`)
translates those events to `text/event-stream`. No existing port or the sync
`reply()` contract had to change — `reply()` now simply joins the stream.

## Out of scope (deliberately deferred)

- **Auth / per-user scoping** — conversations are not user-scoped yet; a
  `user_id` column plus a filter is the clean extension.
