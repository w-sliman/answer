"""Freeze the CITE node — deterministic `[n]` annotation of the grounded draft.

The chain, end to end:
  … → selection → sufficiency → answer (draft + ground) → **cite (here)**

The answer node froze the grounded `draft` + per-claim attribution to
`answers.json`. This runs the SAME production function the graph's `cite_node`
calls (`cite_draft` → `cite.annotate_in_place`) over that frozen draft +
attribution, and freezes the cited result to `eval/fixtures/annotations.json`:
the draft annotated in place with `[n]` markers, plus the flattened per-URL
citation list. One freeze per node — this mirrors the graph split.

MODEL-FREE and OFFLINE. Citation is deterministic containment matching, no LLM,
no endpoint, no network — so this is cheap and infinitely repeatable, which is
exactly why it is worth isolating as its own stage: the annotation becomes
inspectable and verifiable on its own, not buried in the answer output.

Inputs, both already frozen:
  - `answers.json`  — per qid: the grounded `draft` and `claims` (the attribution
    review, rehydrated here into an `AttributionResult`).
  - `selection_question.json` — the SETTLED (latest-iteration) chunks, rehydrated
    into `RetrievalHit` so `chunk_id` → URL resolves exactly as it did live.

Resumable: an already-frozen qid is skipped unless --force; saved after each.

Usage:
    uv run python eval/freeze_annotations.py
    uv run python eval/freeze_annotations.py --only stress-disclaim-002
    uv run python eval/freeze_annotations.py --force
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.pipeline import cite_draft  # noqa: E402
from answer.prompts import AttributionResult  # noqa: E402
from answer.vector_store import RetrievalHit  # noqa: E402

FIXTURES = _EVAL_DIR / "fixtures"
ANSWERS = FIXTURES / "answers.json"
SELECTION = FIXTURES / "selection_question.json"
OUT = FIXTURES / "annotations.json"


def _settled_selection(q: dict) -> dict | None:
    """The question's FINAL selection iteration — the chunks the answer was over."""
    its = q.get("iterations") or []
    return max(its, key=lambda i: i["iteration"]) if its else None


def _hits_for(itr: dict) -> list[RetrievalHit]:
    """Selection chunks -> RetrievalHit — identical reconstruction to freeze_answers,
    so `chunk_id` → URL resolves exactly as it did when the draft was produced."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated qids to freeze")
    ap.add_argument("--force", action="store_true",
                    help="re-freeze even if an annotation already exists")
    args = ap.parse_args()

    for path, hint in ((ANSWERS, "freeze_answers.py"),
                       (SELECTION, "replay_retrieval.py --mode question")):
        if not path.exists():
            print(f"missing {path.relative_to(_PROJECT_ROOT)} — run `{hint}` first")
            sys.exit(2)

    answers = json.loads(ANSWERS.read_text(encoding="utf-8")).get("questions", {})
    sel = {q["qid"]: q for q in
           json.loads(SELECTION.read_text(encoding="utf-8"))["questions"]}

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8")).get("questions", {})

    qids = list(answers)
    if args.only:
        want = {q.strip() for q in args.only.split(",") if q.strip()}
        qids = [q for q in qids if q in want]
    if not qids:
        print("no questions selected")
        sys.exit(2)

    def _save() -> None:
        OUT.write_text(json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_answers": ANSWERS.name,
            "source_selection": SELECTION.name,
            "questions": out,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"answers: {ANSWERS.name} · {len(qids)} question(s)  (model-free)\n")
    for qid in qids:
        if qid in out and not args.force:
            print(f"[skip] {qid} (already frozen; --force to redo)")
            continue
        a = answers[qid]
        itr = _settled_selection(sel.get(qid, {}))
        if itr is None:
            print(f"[warn] {qid}: no selection iteration — skipping")
            continue

        draft = a.get("draft", "")
        # Rehydrate the attribution review from the frozen claims, then run the
        # exact production cite path over it.
        attribution = AttributionResult.model_validate({"claims": a.get("claims", [])})
        hits = _hits_for(itr)
        answer = cite_draft(draft, attribution, hits)

        n_markers = len(re.findall(r"\[\d+\]", answer.text))
        n_supported = sum(1 for c in attribution.claims if c.supported)
        out[qid] = {
            "question": a.get("question", ""),
            "disclaim": a.get("disclaim", False),
            "draft": draft,
            "text": answer.text,
            # Per-sentence, per-marker evidence: each [n] on a line carries the
            # SPECIFIC quote(s) backing that line (per-claim, artifacts stripped).
            "placements": [p.model_dump() for p in answer.placements],
            # Per-URL bibliography ("sources used"), quotes aggregated + cleaned.
            "citations": [c.model_dump() for c in answer.citations],
            "n_citations": len(answer.citations),
            "n_markers": n_markers,
            "n_supported_claims": n_supported,
        }
        print(f"[{qid}] {len(answer.citations)} citation(s) · {n_markers} marker(s) "
              f"· {len(answer.placements)} placement(s) · from {n_supported} supported claim(s)")
        _save()

    print(f"\nfrozen: {OUT.relative_to(_PROJECT_ROOT)}  ({len(out)} annotated)")


if __name__ == "__main__":
    main()
