"""Freeze stage 1: generate the planner's queries for every dataset question, ONCE.

Emits a machine-readable artifact -- `eval/fixtures/queries.json` -- that every
downstream freeze/replay tool reads. This exists because the planner is
STOCHASTIC: re-running it gives different queries each time (one run drew
"Anthropic Claude Pro max context window 2026", another "...window size"), so
anything built on a fresh generation drifts under you. Freeze once, compare
against a fixed input forever.

JSON is the contract, not the report. `query_playground.py` renders markdown for
reading, which is right for judging quality by eye -- but markdown can't feed a
tool, which is why we previously had to parse prose or fall back to stale traces.

Resumable: a qid already present in queries.json is skipped, so a partial run can
be re-run to fill gaps. `--force` regenerates everything from scratch.

Shape (`schema: rounds/v1`): each qid holds a list of ROUNDS, not a flat query
list. This first pass writes round 0 (`source: "planner"`); `freeze_refinement.py`
later appends round 1+ (`source: "refinement"`) into the same file, so the whole
convergence loop for a question lives in one entry.

Needs the chat endpoint. No search, no fetch. Light imports (no answer.pipeline).

Usage:
    uv run python eval/freeze_queries.py
    uv run python eval/freeze_queries.py --only stress-multi-003
    uv run python eval/freeze_queries.py --force
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.llm import get_llm  # noqa: E402
from answer.prompts import (  # noqa: E402
    PlannedQueries,
    build_query_generation_messages_needs,
)

DEFAULT_DATASET = _EVAL_DIR / "dataset_stress.yaml"
FIXTURES_DIR = _EVAL_DIR / "fixtures"
OUT_PATH = FIXTURES_DIR / "queries.json"
CALL_TIMEOUT_S = 180.0


def load_dataset(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [{"qid": q["id"], "question": q["question"]} for q in data["questions"]]


def load_existing(path: Path) -> dict[str, dict]:
    """Existing frozen entries, keyed by qid (empty if the file is absent)."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {e["qid"]: e for e in payload.get("questions", [])}


async def plan_one(qid: str, question: str, attempts: int = 3) -> list[dict]:
    # The first call after a cold Ollama start is the fragile one (model still
    # loading into VRAM): it can TIME OUT or return output the structured parser
    # rejects — which is what intermittently drops the first question
    # (stress-multi-001). .with_retry does not cover a timeout (it escapes) nor a
    # well-formed EMPTY list, so wrap the whole call and retry on BOTH an
    # exception and an empty result, printing WHY each attempt failed so the real
    # cause is visible in the log instead of guessed at.
    llm = get_llm()
    structured = llm.with_structured_output(PlannedQueries).with_retry(
        stop_after_attempt=2,
    )
    msgs = build_query_generation_messages_needs(question)
    for attempt in range(1, attempts + 1):
        try:
            result = await asyncio.wait_for(structured.ainvoke(msgs), timeout=CALL_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            print(f"[{qid}] attempt {attempt}/{attempts} FAILED — "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        queries = [
            {"query": pq.query.strip(), "why": pq.why}
            for pq in result.queries
            if pq.query.strip()
        ]
        if queries:
            return queries
        print(f"[{qid}] attempt {attempt}/{attempts} — planner returned no queries",
              flush=True)
    return []


async def _amain(args) -> None:
    rows = load_dataset(Path(args.dataset))
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in rows if r["qid"] in wanted]
    if not rows:
        print("no questions selected")
        sys.exit(2)

    frozen = {} if args.force else load_existing(OUT_PATH)
    todo = [r for r in rows if r["qid"] not in frozen]

    print(f"dataset: {len(rows)} question(s) | already frozen: "
          f"{len(rows) - len(todo)} | to generate: {len(todo)}\n")

    if todo:
        # Warm the model so the FIRST real question isn't the cold-start casualty
        # (that first post-load call is what intermittently drops stress-multi-001).
        print("warming up the model (first call after load is the slow one)…", flush=True)
        try:
            await asyncio.wait_for(get_llm().ainvoke("ready?"), timeout=CALL_TIMEOUT_S)
            print("  warm.\n", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  warmup failed (continuing anyway): {type(exc).__name__}: {exc}\n",
                  flush=True)

    for r in todo:
        qid = r["qid"]
        queries = await plan_one(qid, r["question"])
        if not queries:
            # Leaving a qid unfrozen keeps queries.json < N, so the freeze loop
            # regenerates every run and search never converges. After retries on
            # both timeouts and empty results, fall back to the question itself so
            # the run CONVERGES — clearly marked so a genuine planner failure is
            # auditable, not silently masked.
            print(f"[{qid}] ALL {3} attempts failed — FALLBACK query = the question",
                  flush=True)
            queries = [{"query": r["question"],
                        "why": "fallback: planner produced nothing after retries"}]
        frozen[qid] = {
            "qid": qid,
            "question": r["question"],
            "rounds": [{"round": 0, "source": "planner", "queries": queries}],
        }
        print(f"[{qid}] {len(queries)} query(ies)")
        for q in queries:
            print(f"    - {q['query']}")

    # Write in dataset order so the artifact is stable across runs.
    ordered = [frozen[r["qid"]] for r in load_dataset(Path(args.dataset))
               if r["qid"] in frozen]
    FIXTURES_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "schema": "rounds/v1",
                "model": os.environ.get("OLLAMA_MODEL", "?"),
                "planner": "build_query_generation_messages_needs",
                "dataset": Path(args.dataset).name,
                "questions": ordered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    missing = [r["qid"] for r in rows if r["qid"] not in frozen]
    print()
    print("=" * 60)
    print(f"frozen: {len(ordered)} question(s) -> {OUT_PATH.relative_to(_PROJECT_ROOT)}")
    if missing:
        print(f"MISSING {len(missing)}: {', '.join(missing)}")
        print("re-run to fill the gaps (already-frozen entries are skipped)")
    print("=" * 60)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    ap.add_argument("--only", type=str, default=None,
                    help="question id(s), comma-separated")
    ap.add_argument("--force", action="store_true",
                    help="regenerate every question, ignoring what's frozen")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
