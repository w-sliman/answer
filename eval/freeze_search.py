"""Freeze stage 2: lock BRAVE search results for every frozen query.

Reads `eval/fixtures/queries.json` (from `freeze_queries.py`) and writes ONE
artifact, `eval/fixtures/search_results.json`, holding each query's raw results.
Downstream stages replay off it with ZERO network -- which is the point: every
`run_eval` used to re-search the same questions from scratch, and that
self-inflicted traffic is what gets our IP rate-limited.

Same single-file shape as `colab_freeze_search.py`, deliberately: the Colab twin
(run it for a fresh IP when brave throttles this machine) produces a file that
drops straight in here with no conversion.

Works as a flat QUEUE OF QUERIES, not a walk over questions: progress is
`[5/15] (r0) <query>` with each query's result count, because when you're watching
a rate limit the unit that matters is the query, not which question it belongs to.
Which qid a query rolls up to only matters at save time.

ROUNDS-AWARE (`queries.json` schema `rounds/v1`): a question can carry a planner
round (0) plus refinement rounds (1+). This queues the searchable queries across
ALL rounds, skips STOP rounds (`searchable=false`, no queries), and tags each
result entry with its `round`. Already-locked (round, query) pairs are never
re-searched, so a refinement round adds only its new queries to the queue while
round 0 stays frozen.

BRAVE ONLY, deliberately. Two reasons:
 1. A frozen corpus should be ONE consistent source. Letting ddgs fall back to
    yahoo/mojeek/etc. would give a patchwork -- some questions brave, some not --
    and any later comparison would be confounded by which engine happened to be
    up at freeze time.
 2. Fallback exists to answer NOW; this tool doesn't need to. When brave is
    throttled we stop and you re-run later. Resuming beats degrading.

NO RETRY, on purpose: brave's limit is time-based, so retrying inside a lockout
just burns the window. The moment a query fails we stop. The file is rewritten
after EVERY locked query, so nothing is ever lost -- re-run later and it picks up
exactly where it stopped. A query that never locked is simply absent; we never
write an empty entry, because an empty fixture is worse than a missing one (it
silently poisons everything downstream and looks like a model problem).

Rate limits are indistinguishable from real misses: ddgs SWALLOWS brave's HTTP
429 and reports it as "No results found." (`err` is None). If a query stays stuck
across several re-runs while others succeed, it's probably a genuine empty.

Usage:
    uv run python eval/freeze_search.py                # fill whatever's missing
    uv run python eval/freeze_search.py --only stress-multi-003
    uv run python eval/freeze_search.py --status       # report, search nothing
    uv run python eval/freeze_search.py --force        # re-search everything
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.search import search  # noqa: E402

BACKEND = "brave"
FIXTURES_DIR = _EVAL_DIR / "fixtures"
QUERIES_PATH = FIXTURES_DIR / "queries.json"
RESULTS_PATH = FIXTURES_DIR / "search_results.json"
MAX_RESULTS = 10
DEFAULT_DELAY_S = 3.0


def searchable_queries(entry: dict) -> list[tuple[int, str]]:
    """(round, query) for every SEARCHABLE query in a queries.json entry.

    `schema: rounds/v1`: round 0 is the planner, round 1+ are refinement rounds;
    a STOP round carries `searchable=false` and no queries and is skipped here.
    """
    out: list[tuple[int, str]] = []
    for rd in entry["rounds"]:
        if rd.get("searchable", True):  # the planner round has no 'searchable'
            out.extend((rd["round"], q["query"]) for q in rd.get("queries", []))
    return out


def load_locked(qids: list[str]) -> dict[str, dict[tuple[int, str], list[dict]]]:
    """Locked results keyed by qid -> (round, query) -> results (empty if no file)."""
    locked: dict[str, dict[tuple[int, str], list[dict]]] = {q: {} for q in qids}
    if not RESULTS_PATH.exists():
        return locked
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    for entry in payload.get("questions", []):
        qid = entry["qid"]
        if qid in locked:
            for e in entry.get("queries", []):
                if e.get("results"):
                    locked[qid][(e["round"], e["query"])] = e["results"]
    return locked


def save(rows: list[dict], planned: dict[str, list[tuple[int, str]]],
         locked: dict[str, dict[tuple[int, str], list[dict]]]) -> None:
    """Rewrite the whole artifact. Called after every locked query."""
    out = []
    for r in rows:
        qid = r["qid"]
        # Preserve planned (round, order); only emit queries we have results for.
        entries = [{"round": rnd, "query": q, "results": locked[qid][(rnd, q)]}
                   for (rnd, q) in planned[qid] if (rnd, q) in locked[qid]]
        out.append({
            "qid": qid,
            "question": r["question"],
            "complete": len(entries) == len(planned[qid]),
            "n_planned": len(planned[qid]),
            "queries": entries,
        })
    FIXTURES_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "backend": BACKEND,
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "questions": out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="question id(s), comma-separated")
    ap.add_argument("--status", action="store_true",
                    help="report what's locked and exit; issues no searches")
    ap.add_argument("--force", action="store_true",
                    help="re-search every query, ignoring what's locked")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S,
                    help=f"seconds to wait after each query (default {DEFAULT_DELAY_S})")
    args = ap.parse_args()

    if not QUERIES_PATH.exists():
        print(f"missing {QUERIES_PATH.relative_to(_PROJECT_ROOT)} — "
              "run `uv run python eval/freeze_queries.py` first")
        sys.exit(2)

    all_rows = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["questions"]
    rows = all_rows
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in all_rows if r["qid"] in wanted]
    if not rows:
        print("no questions selected")
        sys.exit(2)

    # Flatten to one queue of (round, query); qid only matters when saving.
    planned = {r["qid"]: searchable_queries(r) for r in all_rows}
    locked = load_locked([r["qid"] for r in all_rows])
    if args.force:
        for r in rows:
            locked[r["qid"]] = {}

    selected = {r["qid"] for r in rows}
    queue = [(qid, rnd, q) for qid in planned if qid in selected
             for (rnd, q) in planned[qid] if (rnd, q) not in locked[qid]]
    total = sum(len(planned[qid]) for qid in selected)
    have = sum(len(locked[qid]) for qid in selected)

    print(f"{total} queries total · {have} already locked · {len(queue)} to search")
    print(f"backend={BACKEND} · {args.delay}s after each query\n")

    if args.status:
        for r in rows:
            qid = r["qid"]
            n, want = len(locked[qid]), len(planned[qid])
            print(f"  {'OK ' if n == want else '   '}{n}/{want}  {qid}")
        sys.exit(0)

    if not queue:
        print("nothing to do — everything is locked.")
        sys.exit(0)

    ok = 0
    stopped = False
    for i, (qid, rnd, q) in enumerate(queue, start=1):
        print(f"[{i}/{len(queue)}] (r{rnd}) {q}", flush=True)
        try:
            res = search(q, max_results=MAX_RESULTS, backend=BACKEND)
        except Exception as exc:  # noqa: BLE001
            # ddgs reports brave's 429 as "No results found." — indistinguishable.
            print(f"          ✗ {type(exc).__name__}: {exc}")
            stopped = True
            break
        if not res:
            print("          ✗ 0 results")
            stopped = True
            break
        locked[qid][(rnd, q)] = [{"title": x.title, "url": x.url, "snippet": x.snippet}
                                 for x in res]
        save(all_rows, planned, locked)  # persist every turn — never lose work
        print(f"          ✓ {len(res)} results", flush=True)
        ok += 1
        if i < len(queue):
            time.sleep(args.delay)

    save(all_rows, planned, locked)
    have = sum(len(locked[qid]) for qid in selected)

    print()
    print("=" * 60)
    print(f"locked: {have}/{total} queries  ->  "
          f"{RESULTS_PATH.relative_to(_PROJECT_ROOT)}")
    if stopped or have < total:
        print(f"STOPPED — brave is refusing. {total - have} queries remain.")
        print("re-run later — locked queries are never re-searched.")
        print("or run eval/colab_freeze_search.py in Colab for a fresh IP.")
    else:
        print("all queries locked. no re-run needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
