"""LLM sufficiency judge over the answer-bound chunk set.

After retrieve+rerank+select picks the top-N chunks the answer node will see,
this module decides whether those chunks actually *contain* what the question
asks — or whether the pipeline should loop back and search again.

This is the **v1.1 judge** the original score heuristic's docstring anticipated
("if the heuristic proves insufficient on the eval, add an LLM judge as v1.1 —
same interface, different implementation"). The heuristic is gone: it was proven
dead and *inverted* on the frozen eval — it fires on none of the questions, and a
cross-encoder scores TOPICAL SIMILARITY, not answer presence, so the
unanswerable `disclaim-001` (median 5.31) outscored the fully-answerable
`multi-001` (2.79). No threshold separates them.

The judge instead READS the chunks (`build_sufficiency_verdict_messages` /
`SufficiencyVerdict` in `prompts.py`, the locked `v3` prompt). It returns a
routing verdict — `sufficient` (answer from these) / `insufficient` (search
again) — with a short reason grounded in the sources.

Interface unchanged for callers: still returns `SufficiencyResult` with
`sufficient` + `reason`; the score-only fields (`top_1_score`/`median_score`)
were removed with the heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import get_llm, with_schema
from .prompts import SufficiencyVerdict, build_sufficiency_verdict_messages
from .vector_store import RetrievalHit


@dataclass
class SufficiencyResult:
    """The outcome of one sufficiency check.

    `reason` is the judge's grounded justification — it lands in the trace JSON,
    drives the disclaim prompt when insufficient, and (once regeneration is
    loop-aware) is the brief fed back to the planner.
    """

    sufficient: bool
    reason: str
    n_chunks: int


async def judge_sufficiency(
    question: str, selected: list[RetrievalHit]
) -> SufficiencyResult:
    """Decide whether the selected chunks contain what the question asks.

    An empty selection is trivially insufficient — no LLM call needed (typical
    for adversarial questions where the corpus genuinely lacks the answer).

    Otherwise the judge reads the chunks and returns its routing verdict. The
    call is backend-agnostic (`with_schema`) and uses the model's default
    non-thinking pass — the shape `v3` was validated under. The only retry is
    the structured-output parse retry; a rare mid-sentence `reason` (an upstream
    Ollama/Gemma constrained-JSON quirk) is tolerated,
    not re-rolled — the verdict itself is a constrained enum and never truncates.
    """
    if not selected:
        return SufficiencyResult(
            sufficient=False,
            reason="no chunks were selected (corpus has no relevant content)",
            n_chunks=0,
        )

    llm = get_llm()
    structured = with_schema(llm, SufficiencyVerdict).with_retry(
        stop_after_attempt=2,
    )
    messages = build_sufficiency_verdict_messages(question, selected)
    verdict: SufficiencyVerdict = await structured.ainvoke(messages)
    return SufficiencyResult(
        sufficient=verdict.sources_answer_question,
        reason=verdict.reason,
        n_chunks=len(selected),
    )
