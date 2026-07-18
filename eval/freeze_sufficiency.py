"""Freeze the SUFFICIENCY step — one stage of the freeze chain, nothing else.

The chain freezes each stage as its own artifact:
  queries → search → pages → chunks → embeddings → selection → **sufficiency (here)**

This runs ONLY the sufficiency judge over the frozen selection and freezes the
verdict per question to `eval/fixtures/sufficiency.json` (schema `iterations/v1`).
No refinement, no answer — those are their own steps. Faithful: it calls the SAME
production function the graph's `sufficiency_node` calls (`judge_sufficiency`), so
the frozen verdict is exactly what the live pipeline would produce over these chunks.

ITERATION-AWARE. A question can be judged more than once as it goes round the
refinement loop, so verdicts are keyed by ITERATION, not just qid:
`questions[qid].iterations = [{iteration, sufficient, reason, n_chunks}, …]`. A
question's current iteration is the max SEARCHABLE round in `queries.json` (round 0
= iter 0; a searchable refinement round 1 = iter 1; a STOP round does NOT advance
it). This is what removes the need for `--only`/`--force`: a plain re-run judges
exactly the `(qid, iteration)` pairs it has not judged yet — so after a refinement
search + re-selection, only the question that advanced an iteration is re-judged,
and the earlier verdict is preserved as history (the loop narrative).

Only the judge's LLM call hits the endpoint (one per question judged). No search,
no fetch. Resumable: an already-judged `(qid, iteration)` is skipped unless --force.

Usage:
    uv run python eval/freeze_sufficiency.py            # judge whatever advanced
    uv run python eval/freeze_sufficiency.py --only stress-disclaim-002
    uv run python eval/freeze_sufficiency.py --force    # re-judge current iteration
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.sufficiency import judge_sufficiency  # noqa: E402

FIXTURES = _EVAL_DIR / "fixtures"
SELECTION = FIXTURES / "selection_question.json"
QUERIES = FIXTURES / "queries.json"
OUT = FIXTURES / "sufficiency.json"
CALL_TIMEOUT_S = 300.0


def _selection_at(q: dict, iteration: int) -> dict | None:
    """The selection this question saw AT `iteration`.

    `selection_question.json` is iteration-keyed (schema `iterations/v1`): a
    question that went round the refinement loop has one entry per iteration,
    each over a larger pool. We judge the iteration we're currently ON, not
    whatever happens to be last.
    """
    for itr in q.get("iterations", []):
        if itr["iteration"] == iteration:
            return itr
    return None


def _hits_for(itr: dict):
    """Selection chunks -> objects the judge duck-types on (.domain/.text)."""
    hits = []
    for g in itr["groups"]:
        for c in g["chunks"]:
            hits.append(SimpleNamespace(domain=c["domain"], text=c["text"]))
    return hits


def _iteration_by_qid() -> dict[str, int]:
    """Current iteration per qid = max SEARCHABLE round in queries.json.

    Round 0 (planner) always counts; a STOP refinement round (searchable=false)
    does not advance the iteration. Missing file -> everything at iteration 0.
    """
    if not QUERIES.exists():
        return {}
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for e in data.get("questions", []):
        searchable = [rd["round"] for rd in e.get("rounds", [])
                      if rd.get("searchable", True)]
        out[e["qid"]] = max(searchable) if searchable else 0
    return out


async def _amain(args) -> None:
    if not SELECTION.exists():
        print(f"missing {SELECTION.relative_to(_PROJECT_ROOT)} — run "
              "`replay_retrieval.py --mode question` first")
        sys.exit(2)
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8")).get("questions", {})

    iter_by_qid = _iteration_by_qid()

    rows = sel["questions"]
    if args.only:
        want = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in rows if r["qid"] in want]

    def _save() -> None:
        OUT.write_text(json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema": "iterations/v1",
            "source_selection": SELECTION.name,
            "source_queries": QUERIES.name,
            "questions": out,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"selection: {SELECTION.name} · {len(rows)} question(s)\n")
    for r in rows:
        qid = r["qid"]
        cur = iter_by_qid.get(qid, 0)
        iters = out.get(qid, {}).get("iterations", [])
        if any(it["iteration"] == cur and not args.force for it in iters):
            print(f"[skip] {qid} (iteration {cur} already judged; --force to redo)")
            continue
        itr = _selection_at(r, cur)
        if itr is None:
            print(f"[warn] {qid}: no selection for iteration {cur} — skipping "
                  "(re-run replay_retrieval.py --mode question)")
            continue
        try:
            suf = await asyncio.wait_for(
                judge_sufficiency(r["question"], _hits_for(itr)),
                timeout=CALL_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {qid}: {type(exc).__name__}: {exc}")
            continue
        # Replace this iteration's verdict if it exists (--force), else append;
        # keep the list ordered by iteration so history reads top-to-bottom.
        kept = [it for it in iters if it["iteration"] != cur]
        kept.append({
            "iteration": cur,
            "sufficient": suf.sufficient,
            "reason": suf.reason,
            "n_chunks": suf.n_chunks,
        })
        kept.sort(key=lambda it: it["iteration"])
        out.setdefault(qid, {})["iterations"] = kept
        mark = "sufficient" if suf.sufficient else "INSUFFICIENT"
        print(f"[freeze] {qid}: iteration {cur} → {mark}")
        _save()  # save after every question so a mid-run stop keeps what's done

    print(f"\nfrozen: {OUT.relative_to(_PROJECT_ROOT)}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated qids to freeze")
    ap.add_argument("--force", action="store_true",
                    help="re-freeze even if a verdict already exists")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
