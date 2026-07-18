"""Freeze the REFINEMENT step — one stage, run only on the INSUFFICIENT questions.

The chain, on the loop side:
  ... → selection → sufficiency → **refinement (here)** → [search → ... → selection → sufficiency again] → ...

This reads `sufficiency.json`, takes the questions the judge marked INSUFFICIENT,
and runs the SAME production refinement planner the graph's `generate_queries_node`
runs on a loop iteration (`build_query_generation_messages_needs` with prior
queries + the judge's reason → `RefinedQueries`). It APPENDS each decision as a new
round on that question's entry in `eval/fixtures/queries.json` (schema `rounds/v1`),
so one file holds the whole convergence loop — planner round 0 plus refinement
rounds 1+. The priors handed to the planner are every query from every earlier
round (do-not-repeat). No separate `refinement.json` is written.

  - searchable = false → STOP round. The gap is unfillable by search; this question
    is terminal and will be answered as a disclaim over its current chunks. The
    round carries no queries, so freeze_search skips it.
  - searchable = true  → SEARCH round. It emits new queries. freeze_search picks
    them up automatically (they join the queue; round 0 stays locked), then
    fetch → chunks → embeddings → selection → sufficiency again — the convergence
    loop, bounded by MAX_ITERATIONS.

Only the planner's LLM call hits the endpoint (one per insufficient question). No
search, no fetch. Resumable: already-frozen qids are skipped unless --force.

Prereq: run `freeze_sufficiency.py` first (this reads its verdicts).

Usage:
    uv run python eval/freeze_refinement.py
    uv run python eval/freeze_refinement.py --only stress-disclaim-001
    uv run python eval/freeze_refinement.py --force
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.llm import get_llm, with_schema  # noqa: E402
from answer.prompts import (  # noqa: E402
    RefinedQueries,
    build_query_generation_messages_needs,
)

FIXTURES = _EVAL_DIR / "fixtures"
SELECTION = FIXTURES / "selection_question.json"
QUERIES = FIXTURES / "queries.json"
SUFFICIENCY = FIXTURES / "sufficiency.json"
CALL_TIMEOUT_S = 300.0


def _questions_map() -> dict[str, str]:
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    return {q["qid"]: q["question"] for q in sel["questions"]}


def _all_queries(entry: dict) -> list[str]:
    """Every query across every round of a queries.json entry (rounds shape,
    legacy flat fallback). Used as the do-not-repeat priors for refinement."""
    if "rounds" in entry:
        return [x["query"] for rd in entry["rounds"] for x in rd.get("queries", [])]
    return [x["query"] for x in entry.get("queries", [])]


def _has_refinement_round(entry: dict) -> bool:
    return any(rd.get("source") == "refinement" for rd in entry.get("rounds", []))


async def _refine(question: str, prior: list[str], reason: str) -> RefinedQueries:
    structured = with_schema(get_llm(), RefinedQueries).with_retry(
        stop_after_attempt=2)
    msgs = build_query_generation_messages_needs(
        question, prior_queries=prior, sufficiency_reason=reason)
    return await asyncio.wait_for(structured.ainvoke(msgs), timeout=CALL_TIMEOUT_S)


def _save_queries(payload: dict) -> None:
    payload["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    QUERIES.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")


async def _amain(args) -> None:
    if not SUFFICIENCY.exists():
        print(f"missing {SUFFICIENCY.relative_to(_PROJECT_ROOT)} — run "
              "`freeze_sufficiency.py` first")
        sys.exit(2)
    if not QUERIES.exists():
        print(f"missing {QUERIES.relative_to(_PROJECT_ROOT)} — run "
              "`freeze_queries.py` first")
        sys.exit(2)
    suff = json.loads(SUFFICIENCY.read_text(encoding="utf-8")).get("questions", {})
    qtext = _questions_map()

    # sufficiency.json is iteration-keyed (schema iterations/v1): the LATEST
    # iteration's verdict is the current routing decision.
    def _latest(v: dict) -> dict:
        its = v.get("iterations") or []
        return its[-1] if its else {}

    # We append the refinement round straight into queries.json — one file holds
    # the whole convergence loop per question (planner round + refinement rounds).
    payload = json.loads(QUERIES.read_text(encoding="utf-8"))
    by_qid = {e["qid"]: e for e in payload.get("questions", [])}

    # Only the currently-INSUFFICIENT questions need refinement.
    insufficient = [qid for qid, v in suff.items() if not _latest(v).get("sufficient")]
    if args.only:
        want = {q.strip() for q in args.only.split(",") if q.strip()}
        insufficient = [q for q in insufficient if q in want]

    if not insufficient:
        print("no insufficient questions in sufficiency.json — nothing to refine.")
        return

    print(f"insufficient: {insufficient}\n")
    n_search = n_stop = 0
    for qid in insufficient:
        entry = by_qid.get(qid)
        if entry is None:
            print(f"[warn] {qid} not in queries.json — skipping")
            continue
        entry.setdefault("rounds", [])
        if _has_refinement_round(entry):
            if not args.force:
                print(f"[skip] {qid} (already has a refinement round; --force to redo)")
                # Count what's already there so the summary is accurate.
                for rd in entry["rounds"]:
                    if rd.get("source") == "refinement":
                        n_search += bool(rd.get("searchable"))
                        n_stop += not rd.get("searchable")
                continue
            # --force: drop existing refinement rounds, keep the planner round(s).
            entry["rounds"] = [rd for rd in entry["rounds"]
                               if rd.get("source") != "refinement"]

        reason = _latest(suff[qid]).get("reason", "")
        prior = _all_queries(entry)  # do-not-repeat: every earlier query
        try:
            r = await _refine(qtext[qid], prior, reason)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {qid}: {type(exc).__name__}: {exc}")
            continue
        next_round = max((rd.get("round", 0) for rd in entry["rounds"]), default=-1) + 1
        entry["rounds"].append({
            "round": next_round,
            "source": "refinement",
            "searchable": r.searchable,
            "reason": r.reason,
            "queries": [{"query": q.query, "why": q.why} for q in r.queries],
        })
        if not r.searchable:
            n_stop += 1
            print(f"[freeze] {qid}: round {next_round} STOP (unfillable → disclaim)")
        else:
            n_search += 1
            qs = [q.query for q in r.queries]
            print(f"[freeze] {qid}: round {next_round} SEARCH → {qs}")
        # Save after every question so a mid-run stop keeps what's done.
        _save_queries(payload)

    print(f"\nfrozen into: {QUERIES.relative_to(_PROJECT_ROOT)}  "
          f"({n_search} need a search, {n_stop} stop)")
    if n_search:
        print("Next: freeze_search picks up the new refinement round automatically "
              "(searchable rounds join the queue; STOP rounds are skipped). Run "
              "freeze_search/fetch/chunks/embeddings, update their selection, then "
              "re-run freeze_sufficiency on them.")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated qids to refine")
    ap.add_argument("--force", action="store_true",
                    help="re-freeze even if a decision already exists")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
