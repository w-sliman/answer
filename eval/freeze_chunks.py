"""Freeze stage 4: chunk the frozen pages into `eval/fixtures/chunks.json`.

Reads `pages.json` (+ `search_results.json` for the url -> question mapping) and
writes one flat, stable chunk list. No model, no network -- chunking is a pure
function of (page text, chunk_size, chunk_overlap), so this is cheap to re-run
whenever you tune the knobs.

WHY FREEZE SOMETHING DETERMINISTIC? Not for compute -- for STABLE IDENTITY.
`Chunk.id` is a content hash over (url, position, content), so the same page
always yields the same ids. Pinning them means "chunk 7" refers to the same text
across every downstream experiment, and any change you observe is the change you
made rather than a chunk boundary drifting underneath you.

FLAT AND KEYED BY URL, not grouped per question -- same reasoning as pages.json.
A chunk belongs to a page, not to a question, and pages are shared across
questions. The url -> question mapping lives in search_results.json, so a
per-question (or per-QUERY) chunk set is a join away:

    urls = {r["url"] for e in <question>["queries"] for r in e["results"]}
    chunks_for_question = [c for c in chunks if c["url"] in urls]

That keeps per-query retrieval open as a future option with zero duplication.

MIN-CHARS FILTER. `fetch_pages` only distinguishes empty from non-empty, so
title-only stubs and paywall teasers land as "successes" -- 46-633 chars. They
chunk into junk that competes for retrieval slots against real content (the
50-char ossphere.dev page is already on record for exactly this). We filter HERE
rather than in `freeze_fetch.py` on purpose: pages.json stays a faithful record
of what was actually fetched, and the threshold stays a tunable, reversible
downstream decision instead of destroying data at capture time.

WHAT IS NOT FROZEN, DELIBERATELY: retrieve and rerank. Those score chunks
against a query and are exactly what we want to experiment on (per-query
retrieval, MMR, thresholds) -- freezing them would freeze the variable under
test. Embeddings are the next thing worth freezing: expensive, and deterministic
given (chunk text, model).

Chunk size/overlap come from `config.py` (CHUNK_SIZE_CHARS / CHUNK_OVERLAP_CHARS)
because `chunk_page()` binds them at module level; edit config.py to tune, then
re-run this. The values in force are recorded in the artifact.

Usage:
    uv run python eval/freeze_chunks.py
    uv run python eval/freeze_chunks.py --min-chars 500
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.chunking import chunk_page  # noqa: E402
from answer.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS  # noqa: E402

FIXTURES_DIR = _EVAL_DIR / "fixtures"
SEARCH_PATH = FIXTURES_DIR / "search_results.json"
PAGES_PATH = FIXTURES_DIR / "pages.json"
OUT_PATH = FIXTURES_DIR / "chunks.json"

# Below this a "page" is a title-only stub or a paywall teaser, not content.
# The six known offenders sit at 46/50/77/145/443/633 chars; real pages are
# essentially never under ~150 words.
DEFAULT_MIN_CHARS = 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help=f"skip pages shorter than this (default {DEFAULT_MIN_CHARS})")
    args = ap.parse_args()

    for p in (SEARCH_PATH, PAGES_PATH):
        if not p.exists():
            print(f"missing {p.relative_to(_PROJECT_ROOT)} — run the earlier "
                  "freeze stages first")
            sys.exit(2)

    rows = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))["questions"]
    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))["pages"]

    kept, skipped = {}, []
    for url, text in pages.items():
        if len(text) < args.min_chars:
            skipped.append({"url": url, "chars": len(text)})
        else:
            kept[url] = text

    print(f"{len(pages)} pages · {len(kept)} kept · {len(skipped)} skipped "
          f"(<{args.min_chars} chars)")
    for s in sorted(skipped, key=lambda x: x["chars"]):
        print(f"  skip {s['chars']:>5} chars  {s['url'][:74]}")

    print(f"chunking at {CHUNK_SIZE_CHARS}/{CHUNK_OVERLAP_CHARS} chars "
          "(from config.py)")

    chunks = []
    for url, text in kept.items():
        for c in chunk_page(url, text):
            chunks.append({
                "id": c.id,
                "url": c.url,
                "domain": c.domain,
                "position": c.position,
                "text": c.text,
            })

    FIXTURES_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "chunk_size": CHUNK_SIZE_CHARS,
                "chunk_overlap": CHUNK_OVERLAP_CHARS,
                "min_page_chars": args.min_chars,
                "n_pages_kept": len(kept),
                "n_pages_skipped": len(skipped),
                "skipped_pages": skipped,
                "n_chunks": len(chunks),
                "chunks": chunks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Per-question coverage, via the url -> question join.
    by_url: dict[str, int] = {}
    for c in chunks:
        by_url[c["url"]] = by_url.get(c["url"], 0) + 1
    print(f"\n{'qid':22} {'pages':>7} {'chunks':>7}")
    for r in rows:
        urls = {x["url"] for e in r["queries"] for x in e["results"]}
        have = [u for u in urls if u in kept]
        n = sum(by_url.get(u, 0) for u in have)
        flag = "  <-- LOW" if n < 10 else ""
        print(f"{r['qid']:22} {len(have):>7} {n:>7}{flag}")

    print()
    print("=" * 60)
    print(f"{len(chunks)} chunks from {len(kept)} pages  ->  "
          f"{OUT_PATH.relative_to(_PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
