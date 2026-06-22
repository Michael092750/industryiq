"""LLM-as-judge for the chat benchmark.

A second, independent model -- by default Claude Opus 4.8, deliberately stronger
than and different from the chatbot's own LLM -- scores each chat answer on **two
decoupled axes**, so one harness serves both questions you can ask of a RAG chat:

* ``correctness`` -- does the answer match the REFERENCE (gold) answer? This is the
  *end-to-end* signal: it's what moves when a retrieval-side change (e.g. query
  rewriting) surfaces the answer more often. Declining to answer when the reference
  *has* an answer scores low -- a miss is a miss, however gracefully phrased.
* ``groundedness`` -- is the answer supported by the CONTEXT the chatbot retrieved?
  This is the *faithfulness* signal, independent of correctness: a correct answer
  can be ungrounded (the model used outside knowledge), and a grounded answer can
  be wrong.

Keeping them separate is the whole point. ``correctness`` lets you compare
retrieval techniques; ``correctness`` restricted to the cases where the context was
sufficient (computed in the runner) isolates the chatbot's intrinsic extraction
skill; ``groundedness`` flags hallucination. Using an independent, stronger judge
model avoids the self-evaluation bias of a model grading its own output.

Pure prompt construction (:func:`build_judge_prompt`) is separated from the API
call (:class:`JudgeLLM`), mirroring the split in ``core/generation.py``.
"""

import json
from dataclasses import dataclass

from anthropic import Anthropic

from industryiq.core.vectorstore import Hit

# A different (and stronger) model than the chatbot's, so the evaluation is
# independent rather than a model grading itself.
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

_SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}

# Structured-output schema: forces the judge to return exactly these fields with
# 1-5 integer scores, so parsing never has to cope with free-form prose.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "context_sufficient": {"type": "boolean"},
        "correctness": _SCORE,
        "groundedness": _SCORE,
        "citation": _SCORE,
        "rationale": {"type": "string"},
    },
    "required": [
        "context_sufficient",
        "correctness",
        "groundedness",
        "citation",
        "rationale",
    ],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You evaluate a retrieval-augmented chatbot's ANSWER on independent axes. You "
    "are given a QUESTION, a REFERENCE ANSWER (the ground truth), the CONTEXT "
    "passages the chatbot retrieved, and the chatbot's ANSWER. Score each axis as "
    "an integer 1-5:\n"
    "- correctness: does the ANSWER answer the QUESTION correctly and completely, "
    "judged against the REFERENCE ANSWER? Match on meaning, not wording -- "
    "equivalent numbers, units, and paraphrases are correct. 5 = fully correct; "
    "1 = wrong, missing, or the bot declined to answer. Declining (e.g. 'that is "
    "not in the context') when the REFERENCE has an answer is NOT correct -- score "
    "it low. Judge correctness against the REFERENCE, never excused by what was "
    "retrieved.\n"
    "- groundedness: is every factual claim in the ANSWER supported by the CONTEXT "
    "passages? 5 = fully grounded in the context; 1 = key claims are absent from "
    "the context (the bot used outside knowledge or made it up). This is "
    "independent of correctness: a correct answer can be ungrounded, and a grounded "
    "answer can be wrong.\n"
    "- citation: did the answer cite sources as [n] pointing at the numbered CONTEXT "
    "passages, as the chatbot was instructed to? 5 = correct citations for the "
    "claims made; 1 = none or wrong.\n\n"
    "Also report context_sufficient: does the CONTEXT contain what is needed to "
    "answer? This is a diagnostic about retrieval and must NOT influence the scores. "
    "Give a one-sentence rationale."
)


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge scoring of a single chat answer (all scores are integers 1-5).

    ``correctness`` is graded against the reference answer (end-to-end quality);
    ``groundedness`` against the retrieved context (faithfulness). ``context_sufficient``
    is a retrieval diagnostic that does not affect the scores.
    """

    context_sufficient: bool
    correctness: int
    groundedness: int
    citation: int
    rationale: str


def format_context(hits: list[Hit]) -> str:
    """Number the retrieved passages ``[1]..`` exactly as the chatbot's own prompt
    does, so the answer's ``[n]`` citations line up with what the judge sees."""
    if not hits:
        return "(no relevant context found)"
    return "\n".join(f"[{i}] {hit.metadata.get('text', '')}" for i, hit in enumerate(hits, start=1))


def build_judge_prompt(question: str, reference: str, hits: list[Hit], answer: str) -> str:
    """Assemble the judge's user message from the question, the reference (gold)
    answer, the retrieved context, and the chatbot's answer (pure; no I/O)."""
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER (ground truth):\n{reference}\n\n"
        f"CONTEXT (the passages the chatbot retrieved):\n{format_context(hits)}\n\n"
        f"CHATBOT ANSWER:\n{answer}"
    )


class JudgeLLM:
    """Score chat answers with an independent judge model via the Anthropic API.

    Uses structured outputs (a fixed JSON schema) so every call returns the same
    well-typed fields -- no brittle parsing of free-form text. Extended thinking is
    left off deliberately: the rubric is bounded and the schema constrains the
    output, so a strong model scores reliably without it, which keeps a many-query
    run fast and cheap. For a harder rubric, turn adaptive thinking on.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        # api_key=None lets the SDK fall back to the ANTHROPIC_API_KEY env var.
        self._client = Anthropic(api_key=api_key)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def score(self, question: str, reference: str, hits: list[Hit], answer: str) -> JudgeVerdict:
        """Return the judge's structured verdict for one (question, reference, context, answer)."""
        message = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            system=JUDGE_SYSTEM,
            messages=[
                {"role": "user", "content": build_judge_prompt(question, reference, hits, answer)}
            ],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        text = next((block.text for block in message.content if block.type == "text"), "")
        data = json.loads(text)
        return JudgeVerdict(
            context_sufficient=bool(data["context_sufficient"]),
            correctness=int(data["correctness"]),
            groundedness=int(data["groundedness"]),
            citation=int(data["citation"]),
            rationale=str(data["rationale"]),
        )
