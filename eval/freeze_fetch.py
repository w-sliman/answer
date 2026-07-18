"""Freeze stage 3: fetch every URL in the frozen search results, once.

Reads `eval/fixtures/search_results.json` and writes `eval/fixtures/pages.json`
== {url: markdown}. This is the last link in the freeze chain: with queries,
search results, and pages all frozen, every downstream stage (chunk -> retrieve
-> rerank -> answer -> critique) replays with ZERO network, instantly and
deterministically.

KEYED BY URL, NOT BY QUERY, on purpose. Page content is a property of the URL,
not of the query that surfaced it, and queries overlap heavily (that is why
search_node dedups by URL). Storing per query would hold the same page two or
three times and let the copies drift. The query -> url mapping already lives in
search_results.json, so per-query grouping is a JOIN away, for free:

    urls_for_query = [r["url"] for r in <search_results query entry>["results"]]
    pages_for_query = {u: pages[u] for u in urls_for_query if u in pages}

UNLIKE `freeze_search.py`, THIS DOES NOT STOP ON FAILURE. A search failure means
one rate-limited engine refusing everything, so stopping is right. A fetch
failure is per-site and usually permanent for that URL alone -- a 404, a JS
wall, a timeout, a bot blocker -- and says nothing about the next URL. So we
record it and carry on.

Only non-empty pages are stored (`fetch_page` returns '' on failure). A re-run
retries whatever is missing, so a transient failure costs one re-run and a
permanently dead URL simply never lands.

ONE `fetch_pages()` CALL PER PROCESS — do not loop it. Each call opens its own
`async with AsyncWebCrawler(...)`, i.e. a fresh browser launch, and after ~3
launches in a single process Playwright is exhausted and every later launch
silently returns nothing. Measured 2026-07-15: fetching in batches of 8 gave
7/8, 7/8, 7/8 and then 0/8 for all fifteen remaining batches (21/142 overall) --
a hard cliff, not flaky sites. `arun_many` already dispatches concurrently and
manages browser lifecycle and memory pressure; batching on top of it defeats
the design. Hence a single call here, and `--limit` if you want to split the
work: each invocation is a NEW process, so it gets a fresh browser.

Usage:
    uv run python eval/freeze_fetch.py               # fetch whatever's missing
    uv run python eval/freeze_fetch.py --status      # report, fetch nothing
    uv run python eval/freeze_fetch.py --only stress-multi-003
    uv run python eval/freeze_fetch.py --force       # re-fetch everything
    uv run python eval/freeze_fetch.py --limit 40    # cap this run; re-run for more
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

from answer.fetch import fetch_pages  # noqa: E402

FIXTURES_DIR = _EVAL_DIR / "fixtures"
SEARCH_PATH = FIXTURES_DIR / "search_results.json"
PAGES_PATH = FIXTURES_DIR / "pages.json"



def load_pages() -> dict[str, str]:
    if not PAGES_PATH.exists():
        return {}
    return json.loads(PAGES_PATH.read_text(encoding="utf-8")).get("pages", {})


def save_pages(pages: dict[str, str]) -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    PAGES_PATH.write_text(
        json.dumps(
            {
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_pages": len(pages),
                "pages": pages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def collect_urls(rows: list[dict]) -> list[str]:
    """Unique URLs across every query, first-seen order (same as search_node)."""
    seen: set[str] = set()
    urls: list[str] = []
    for q in rows:
        for e in q.get("queries", []):
            for r in e.get("results", []):
                u = r.get("url")
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
    return urls


async def _amain(args) -> None:
    if not SEARCH_PATH.exists():
        print(f"missing {SEARCH_PATH.relative_to(_PROJECT_ROOT)} — "
              "run `uv run python eval/freeze_search.py` first")
        sys.exit(2)

    rows = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in rows if r["qid"] in wanted]
    if not rows:
        print("no questions selected")
        sys.exit(2)

    urls = collect_urls(rows)
    pages = {} if args.force else load_pages()
    todo = [u for u in urls if u not in pages]
    capped = len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(urls)} unique URLs · {len(urls) - capped} already fetched · "
          f"{capped} missing")
    if args.limit and capped > len(todo):
        print(f"--limit {args.limit}: fetching {len(todo)} this run "
              f"({capped - len(todo)} left for the next)")
    print(f"one dispatcher run, {len(todo)} URLs — crawl4ai handles concurrency\n")

    if args.status:
        for r in rows:
            qurls = {x["url"] for e in r.get("queries", []) for x in e["results"]}
            have = sum(1 for u in qurls if u in pages)
            print(f"  {'OK ' if have == len(qurls) else '   '}"
                  f"{have}/{len(qurls)}  {r['qid']}")
        sys.exit(0)

    if not todo:
        print("nothing to do — every URL is fetched.")
        sys.exit(0)

    # ONE call — see the module docstring. Looping this kills the crawler.
    print("fetching… (quiet for a few minutes; crawl4ai reports at the end)",
          flush=True)
    try:
        got = await fetch_pages(todo)
    except Exception as exc:  # noqa: BLE001
        print(f"fetch_pages failed outright — {type(exc).__name__}: {exc}")
        got = {}

    ok = 0
    for u in todo:
        text = (got.get(u) or "").strip()
        if text:
            pages[u] = text
            ok += 1
            print(f"  ✓ {len(text):>7,} chars  {u[:78]}")
        else:
            # Empty == fetch failed (404 / JS wall / timeout / blocked).
            print(f"  ✗ empty            {u[:78]}")
    save_pages(pages)

    missing = [u for u in urls if u not in pages]
    print()
    print("=" * 60)
    print(f"fetched this run: {ok}/{len(todo)}")
    print(f"pages: {len(urls) - len(missing)}/{len(urls)} URLs  ->  "
          f"{PAGES_PATH.relative_to(_PROJECT_ROOT)}")
    if missing:
        print(f"MISSING {len(missing)} (404 / blocked / timeout — re-run to retry):")
        for u in missing[:10]:
            print(f"  - {u[:88]}")
        if len(missing) > 10:
            print(f"  … and {len(missing) - 10} more")
    else:
        print("every URL fetched. the freeze chain is complete.")
    print("=" * 60)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="question id(s), comma-separated")
    ap.add_argument("--status", action="store_true",
                    help="report what's fetched and exit; fetches nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch every URL, ignoring what's stored")
    ap.add_argument("--limit", type=int, default=None,
                    help="max URLs to fetch in THIS run (each run = fresh "
                         "browser); omit to fetch everything missing")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
