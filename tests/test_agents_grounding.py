"""Grounding on the planned (fan-out) path -- pure, offline.

Two things have to be true before a synthesized answer can be checked at all, and
they are what most of this file pins down:

* a node's grounding text has to survive the trip through ``CapabilityResult`` to
  the blackboard, or there is nothing downstream to verify against;
* every node's ``[n]`` markers have to be re-indexed into one shared numbering, or
  node A's ``[1]`` and node B's ``[1]`` collide in the combined answer.

Then the gate itself is exercised at both seams it now runs on: per node (at the
executor, before a result is posted) and on the composed answer.
"""

from industryiq.core.agents.adapters.blackboard_memory import InMemoryBlackboard
from industryiq.core.agents.adapters.queue_memory import InMemoryTaskQueue
from industryiq.core.agents.executor_local import LocalExecutor
from industryiq.core.agents.grounding import (
    check_node_result,
    global_citation_index,
    hits_from_sources,
    node_question,
    renumber_result,
)
from industryiq.core.agents.models import CapabilityResult, Plan, PlanNode
from industryiq.core.agents.synthesis import Synthesizer, merge_sources
from industryiq.core.agents.worker import DEFAULT_QUEUE, Worker
from industryiq.core.generation import FakeLLM
from industryiq.core.grounding import DEFAULT_ABSTENTION, DeterministicGroundingGate


def _source(label: str, text: str = "chunk text", score: float = 0.9) -> dict[str, object]:
    return {"source": label, "score": score, "text": text}


def _result(summary: str, *labels: str) -> CapabilityResult:
    return CapabilityResult(summary=summary, sources=[_source(label) for label in labels])


# --- rebuilding the gate's input shape --------------------------------------------


def test_hits_from_sources_carries_label_score_and_text() -> None:
    [hit] = hits_from_sources([_source("McKinsey", text="AI market is large", score=0.75)])
    assert hit.id == "McKinsey"
    assert hit.score == 0.75
    assert hit.metadata["text"] == "AI market is large"


def test_hits_from_sources_tolerates_a_citation_with_no_text() -> None:
    # The web-search shape: a URL and a title, no local page content.
    [hit] = hits_from_sources([{"source": "https://a.com", "title": "A"}])
    assert hit.metadata["text"] == ""
    assert hit.score == 0.0


def test_node_question_prefers_the_inputs_then_falls_back_to_the_run() -> None:
    assert node_question({"question": "how big?"}, "the run") == "how big?"
    assert node_question({"query": "how big?"}, "the run") == "how big?"
    assert node_question({"industry": "AI"}, "the run") == "the run"
    assert node_question({"question": "   "}, "the run") == "the run"


# --- the citation namespace -------------------------------------------------------


def test_two_nodes_citing_1_are_renumbered_apart() -> None:
    # The bug this exists to prevent: both nodes numbered their own hits [1], so
    # without re-indexing the combined answer says [1] about two different documents.
    a = _result("AI is big [1]", "mck")
    b = _result("finance is bigger [1]", "bcg")
    merged = merge_sources([a, b])
    index = global_citation_index(merged)
    assert renumber_result(a, index).summary == "AI is big [1]"
    assert renumber_result(b, index).summary == "finance is bigger [2]"


def test_renumbering_maps_every_local_marker_of_one_node() -> None:
    a = _result("first [1] second [2]", "mck", "bcg")
    b = _result("third [1]", "bain")
    index = global_citation_index(merge_sources([a, b]))
    assert renumber_result(b, index).summary == "third [3]"


def test_two_chunks_of_one_document_share_a_global_marker() -> None:
    # Sources de-dup by document, so a node's [1] and [2] over two chunks of the
    # same report both resolve to that one document's marker.
    node = CapabilityResult(
        summary="one [1] two [2]",
        sources=[_source("mck", text="chunk a"), _source("mck", text="chunk b")],
    )
    index = global_citation_index(merge_sources([node]))
    assert renumber_result(node, index).summary == "one [1] two [1]"


def test_a_fabricated_marker_is_dropped_not_carried_through() -> None:
    node = _result("supported [1] but invented [7].", "mck")
    index = global_citation_index(merge_sources([node]))
    # [7] pointed at nothing; leaving a number in would silently mis-attribute.
    assert renumber_result(node, index).summary == "supported [1] but invented."


def test_merged_sources_accumulate_each_chunks_text() -> None:
    # Both chunks ground claims about the same report, so the merged entry has to
    # carry both or a check against it fails an answer it should pass.
    a = CapabilityResult(summary="a", sources=[_source("mck", text="chunk a")])
    b = CapabilityResult(summary="b", sources=[_source("mck", text="chunk b")])
    [merged] = merge_sources([a, b])
    assert merged["text"] == "chunk a\n\nchunk b"


def test_merged_sources_do_not_repeat_identical_text() -> None:
    a = CapabilityResult(summary="a", sources=[_source("mck", text="same")])
    b = CapabilityResult(summary="b", sources=[_source("mck", text="same")])
    [merged] = merge_sources([a, b])
    assert merged["text"] == "same"


# --- the gate, per node -----------------------------------------------------------


def test_check_node_result_without_a_gate_is_a_passthrough() -> None:
    result = _result("untouched [1]", "mck")
    assert check_node_result(None, "q", result) is result


def test_check_node_result_abstains_when_a_node_found_nothing() -> None:
    gate = DeterministicGroundingGate()
    checked = check_node_result(gate, "q", CapabilityResult(summary="confident nonsense"))
    assert checked.summary == DEFAULT_ABSTENTION
    assert checked.data == {"grounded": False, "grounding_reason": "no supporting context"}


def test_check_node_result_keeps_a_grounded_node_as_is() -> None:
    result = _result("grounded [1]", "mck")
    assert check_node_result(DeterministicGroundingGate(), "q", result) is result


def test_check_node_result_records_a_bad_citation_without_rewriting_it() -> None:
    # Renumbering drops the marker structurally, so the gate only has to record it.
    checked = check_node_result(DeterministicGroundingGate(), "q", _result("claim [9]", "mck"))
    assert checked.summary == "claim [9]"
    assert checked.data is not None and checked.data["grounded"] is False


def test_check_node_result_passes_through_a_cited_but_textless_result() -> None:
    # web_search runs server-side: URLs come back, page content does not. Gating on
    # empty text would abstain from a good answer rather than check it strictly.
    web = CapabilityResult(summary="the web answer", sources=[{"source": "https://a.com"}])
    assert check_node_result(DeterministicGroundingGate(), "q", web) is web


# --- the gate, at the executor seam -----------------------------------------------


class _UngroundedCapability:
    """Answers confidently while retrieving nothing -- the case the gate exists for."""

    name = "industry_analysis"
    description = "stub"

    def run(self, inputs: dict[str, object]) -> CapabilityResult:
        return CapabilityResult(summary="the market is worth $5T")


def _one_node_plan() -> Plan:
    return Plan(
        run_id="run1",
        question="how big is the AI market?",
        nodes=(PlanNode("n1", "industry_analysis", {"industry": "AI"}),),
    )


def test_local_executor_gates_a_node_before_it_reaches_the_blackboard() -> None:
    blackboard = InMemoryBlackboard()
    executor = LocalExecutor(
        {"industry_analysis": _UngroundedCapability()},
        blackboard,
        Synthesizer(),
        grounding=DeterministicGroundingGate(),
    )
    results = executor.execute(_one_node_plan())
    assert results["n1"].summary == DEFAULT_ABSTENTION
    # And the *posted* result is the gated one, so a resumed run reads it back too.
    assert blackboard.read("run1", "n1")["summary"] == DEFAULT_ABSTENTION


def test_local_executor_without_a_gate_posts_the_ungrounded_answer() -> None:
    registry = {"industry_analysis": _UngroundedCapability()}
    executor = LocalExecutor(registry, InMemoryBlackboard(), Synthesizer())
    assert executor.execute(_one_node_plan())["n1"].summary == "the market is worth $5T"


def test_worker_gates_a_node_before_it_reaches_the_blackboard() -> None:
    queue, blackboard = InMemoryTaskQueue(), InMemoryBlackboard()
    queue.enqueue(
        DEFAULT_QUEUE,
        {
            "run_id": "run1",
            "node_id": "n1",
            "capability": "industry_analysis",
            "inputs": {"industry": "AI"},
            "question": "how big is the AI market?",
        },
    )
    worker = Worker(
        queue,
        {"industry_analysis": _UngroundedCapability()},
        blackboard,
        consumer="w1",
        grounding=DeterministicGroundingGate(),
    )
    worker.run_once()
    assert blackboard.read("run1", "n1")["summary"] == DEFAULT_ABSTENTION
    assert queue.pending(DEFAULT_QUEUE) == 0  # gated, not failed: the task is acked


# --- the gate, on the composed answer ---------------------------------------------


def _two_node_results() -> tuple[Plan, dict[str, CapabilityResult]]:
    plan = Plan(
        run_id="run1",
        question="compare AI and finance",
        nodes=(
            PlanNode("n1", "industry_analysis", {"industry": "AI"}),
            PlanNode("n2", "industry_analysis", {"industry": "finance"}),
        ),
    )
    return plan, {"n1": _result("AI [1]", "mck"), "n2": _result("finance [1]", "bcg")}


def test_synthesis_prompt_carries_global_markers_and_the_source_list() -> None:
    llm = FakeLLM("AI leads [1] over finance [2]")
    plan, results = _two_node_results()
    Synthesizer(llm).synthesize(plan, results)
    assert llm.last_prompt is not None
    assert "[1] mck\n[2] bcg" in llm.last_prompt  # the numbered Sources block
    assert "finance [2]" in llm.last_prompt  # node n2 re-indexed away from its local [1]


def test_synthesize_abstains_when_the_composed_answer_has_no_sources() -> None:
    plan = Plan(run_id="run1", question="q", nodes=(PlanNode("n1", "industry_analysis", {}),))
    results = {"n1": CapabilityResult(summary="unsupported claim")}
    run = Synthesizer(FakeLLM("composed"), grounding=DeterministicGroundingGate()).synthesize(
        plan, results
    )
    assert run.answer == DEFAULT_ABSTENTION


def test_synthesize_caveats_a_citation_the_composition_invented() -> None:
    plan, results = _two_node_results()
    synth = Synthesizer(FakeLLM("combined [5]"), grounding=DeterministicGroundingGate())
    assert "could not be matched" in synth.synthesize(plan, results).answer


def test_synthesize_leaves_a_clean_composed_answer_alone() -> None:
    plan, results = _two_node_results()
    synth = Synthesizer(FakeLLM("clean [1] and [2]"), grounding=DeterministicGroundingGate())
    assert synth.synthesize(plan, results).answer == "clean [1] and [2]"


def test_stream_appends_the_caveat_rather_than_replacing_the_answer() -> None:
    # Tokens are already out by then, so streaming can only flag -- and the flag has
    # to arrive as a further token, not a silent edit.
    plan, results = _two_node_results()
    synth = Synthesizer(FakeLLM("combined [5]"), grounding=DeterministicGroundingGate())
    streamed = "".join(synth.stream(plan, results))
    assert streamed.startswith("combined [5]")
    assert "could not be matched" in streamed


def test_stream_without_a_gate_is_unchanged() -> None:
    plan, results = _two_node_results()
    streamed = "".join(Synthesizer(FakeLLM("combined [5]")).stream(plan, results))
    assert streamed == "combined [5]"


def test_run_result_sources_are_the_merged_list_the_markers_point_into() -> None:
    plan, results = _two_node_results()
    run = Synthesizer(FakeLLM("a [1] b [2]")).synthesize(plan, results)
    assert [src["source"] for src in run.sources] == ["mck", "bcg"]
