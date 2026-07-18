"""Freeze the ANSWER node — draft + ground (NOT cite).

The chain, end to end:
  queries → search → pages → chunks → embeddings → selection → sufficiency
  → [refinement → search → … → sufficiency]* → **answer (here)** → cite

Every question is TERMINAL by the time this runs: either the judge found its
sources sufficient, or the refinement planner returned a STOP (unfillable by
search), or the loop hit MAX_ITERATIONS. So the answer is generated exactly ONCE
per question, over its SETTLED chunk set — which is why this artifact is keyed by
qid with no iteration dimension. The loop history lives upstream (queries.json
rounds, sufficiency.json iterations); the answer is the terminal output.

Faithful: it calls the SAME production function the graph's `answer_node` calls
(`run_answer_stage`), so the frozen output is exactly what the live pipeline would
produce over these chunks. That is the draft + ground pair:
  1. freeform + thinking draft (a disclaim directive primes an honest disclaim
     when the latest sufficiency verdict is insufficient),
  2. the attribution grounding gate (unsupported claims → bounded re-draft).

CITATION IS A SEPARATE NODE and a separate freeze. This writes the grounded
`draft` + the per-claim attribution (`claims`); the `[n]`-annotated text and the
citation list are produced by `cite_draft`/`cite_node` and frozen by
`eval/freeze_annotations.py` → `annotations.json`. Splitting the freeze mirrors
the graph: one freeze per node.

Reads the LATEST sufficiency iteration as the verdict (schema `iterations/v1`) —
that is the routing decision that reached this stage, and it is what drives the
disclaim path for stress-disclaim-001 (STOP) and -002 (MAX_ITERATIONS).

Needs the chat endpoint. Each question costs >=2 LLM calls (draft + attribution),
more if the grounding gate re-drafts. No search, no fetch, no embedding.
Resumable: an already-frozen qid is skipped unless --force; saved after every
question so a mid-run stop keeps what's done.

Usage:
    uv run python eval/freeze_answers.py
    uv run python eval/freeze_answers.py --only stress-disclaim-002
    uv run python eval/freeze_answers.py --force
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

from answer.pipeline import run_answer_stage  # noqa: E402
from answer.sufficiency import SufficiencyResult  # noqa: E402
from answer.vector_store import RetrievalHit  # noqa: E402

FIXTURES = _EVAL_DIR / "fixtures"
SELECTION = FIXTURES / "selection_question.json"
SUFFICIENCY = FIXTURES / "sufficiency.json"
OUT = FIXTURES / "answers.json"
CALL_TIMEOUT_S = 900.0  # draft(thinking) + attribution + possible re-draft


def _settled_selection(q: dict) -> dict | None:
    """The question's FINAL selection — the highest iteration it reached.

    `selection_question.json` is iteration-keyed (`iterations/v1`). Every question
    is terminal by the time the answer stage runs, so the answer is generated over
    the last iteration's chunk set; the earlier iterations are the loop's history.
    """
    its = q.get("iterations") or []
    return max(its, key=lambda i: i["iteration"]) if its else None


def _hits_for(itr: dict) -> list[RetrievalHit]:
    """Selection chunks -> RetrievalHit, the shape the answer stage consumes.

    `selection_question.json` names the chunk `id`; RetrievalHit calls it
    `chunk_id`. `embedding` is irrelevant here (the answer/attribution paths
    never read it), so it stays empty.
    """
    hits: list[RetrievalHit] = []
    for g in itr["groups"]:
        for c in g["chunks"]:
            hits.append(RetrievalHit(
                chunk_id=c["id"],
                text=c["text"],
                url=c.get("url", ""),
                domain=c.get("domain", ""),
                position=int(c.get("position", 0)),
                retrieval_score=float(c.get("retrieval_score", 0.0)),
                embedding=[],
                rerank_score=c.get("rerank_score"),
            ))
    return hits


def _latest_verdict(v: dict) -> dict:
    """The LATEST sufficiency iteration — the verdict that reached this stage."""
    its = v.get("iterations") or []
    return its[-1] if its else {}


async def _amain(args) -> None:
    for path, hint in ((SELECTION, "replay_retrieval.py --mode question"),
                       (SUFFICIENCY, "freeze_sufficiency.py")):
        if not path.exists():
            print(f"missing {path.relative_to(_PROJECT_ROOT)} — run `{hint}` first")
            sys.exit(2)

    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    suff = json.loads(SUFFICIENCY.read_text(encoding="utf-8")).get("questions", {})

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8")).get("questions", {})

    rows = sel["questions"]
    if args.only:
        want = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in rows if r["qid"] in want]
    if not rows:
        print("no questions selected")
        sys.exit(2)

    def _save() -> None:
        OUT.write_text(json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_selection": SELECTION.name,
            "source_sufficiency": SUFFICIENCY.name,
            "questions": out,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"selection: {SELECTION.name} · {len(rows)} question(s)\n")
    for r in rows:
        qid = r["qid"]
        if qid in out and not args.force:
            print(f"[skip] {qid} (already frozen; --force to redo)")
            continue

        verdict = _latest_verdict(suff.get(qid, {}))
        if not verdict:
            print(f"[warn] {qid}: no sufficiency verdict — skipping "
                  "(run freeze_sufficiency.py first)")
            continue
        sufficiency = SufficiencyResult(
            sufficient=bool(verdict["sufficient"]),
            reason=verdict.get("reason", ""),
            n_chunks=int(verdict.get("n_chunks", 0)),
        )
        itr = _settled_selection(r)
        if itr is None:
            print(f"[warn] {qid}: no selection iterations — skipping "
                  "(run replay_retrieval.py --mode question first)")
            continue
        hits = _hits_for(itr)
        mode = "ANSWER" if sufficiency.sufficient else "DISCLAIM"
        print(f"[{qid}] {mode} · {len(hits)} chunks · "
              f"iteration {verdict.get('iteration', 0)}", flush=True)

        try:
            result = await asyncio.wait_for(
                run_answer_stage(r["question"], hits, sufficiency),
                timeout=CALL_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {qid}: {type(exc).__name__}: {exc}")
            continue

        attribution = result.get("attribution")
        claims = [c.model_dump() for c in (attribution.claims if attribution else [])]
        n_unsupported = sum(1 for c in claims if not c["supported"])

        out[qid] = {
            "question": r["question"],
            "disclaim": not sufficiency.sufficient,
            "sufficiency": {
                "iteration": verdict.get("iteration", 0),
                "sufficient": sufficiency.sufficient,
                "reason": sufficiency.reason,
            },
            "n_chunks": len(hits),
            "answer_attempts": result.get("answer_attempts"),
            # The `answer` NODE's output: the freeform grounded draft (no `[n]`
            # markers) + the per-claim attribution. Citation is the `cite` node's
            # job and its own freeze (`freeze_annotations.py` -> annotations.json),
            # so the cited text / citation list live there, not here.
            "draft": result.get("draft", ""),
            "claims": claims,
            "n_claims": len(claims),
            "n_unsupported": n_unsupported,
        }
        print(f"    → draft {len(result.get('draft', ''))} chars "
              f"· {len(claims)} claim(s), {n_unsupported} unsupported "
              f"· {result.get('answer_attempts')} attempt(s)")
        _save()  # save after every question so a mid-run stop keeps what's done

    n_disclaim = sum(1 for v in out.values() if v["disclaim"])
    print(f"\nfrozen: {OUT.relative_to(_PROJECT_ROOT)}  "
          f"({len(out)} answered, {n_disclaim} disclaim)")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated qids to freeze")
    ap.add_argument("--force", action="store_true",
                    help="re-freeze even if an answer already exists")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
