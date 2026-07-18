"""Search wrapper around ddgs.

Normalizes results from ddgs.text() and ddgs.news() into a single SearchResult
dataclass, so the rest of the pipeline never touches ddgs directly. Swap the
search provider by replacing the body of `search()` — the interface stays the
same. (Which *engines* ddgs itself queries is a separate knob: `_TEXT_BACKENDS`.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from ddgs import DDGS
from ddgs.exceptions import DDGSException

# Text engines, in preference order. Explicitly pinned -- do NOT use the
# ddgs default of backend="auto".
#
# Why: ddgs sorts engines by `priority` DESCENDING, and its text registry is
#   wikipedia (2.0), grokipedia (1.9), then brave/duckduckgo/yahoo/mojeek/
#   yandex (all 1.0).
# Meanwhile ddgs derives its engine fan-out from max_results:
#   max_workers = min(unique_providers, ceil(max_results / 10) + 1)
# so our max_results=10 gives max_workers=2. Together those mean "auto" queries
# wikipedia + grokipedia FIRST on every call, and if those two return 10 hits it
# breaks before consulting a real search engine at all. Two encyclopedias are a
# bad front door for a web-research pipeline -- it happened in the wild
# (stress-refine-001 answered off en.wikipedia.org and under-answered).
# Passing an explicit list skips that front-loading entirely.
#
# brave = its own independent index (preferred). duckduckgo = provider "bing",
# the broadest coverage, so it backs brave up when brave is thin or rate-limits.
# Deliberately excluded: wikipedia/grokipedia (encyclopedias, not web search),
# yahoo (also provider "bing" -> redundant with duckduckgo), mojeek (tiny index),
# yandex (frequently blocked, skews non-us-en).
#
# Listing two costs no more than one: ddgs stops as soon as it has max_results,
# so the second engine is fallback depth, not extra traffic. Order is preserved
# (equal priority + a stable sort).
_TEXT_BACKENDS = "brave,duckduckgo"

# ddgs's engines occasionally throw a transient connection error or a spurious
# "No results found." -- proven to self-heal on a short retry (see
# corpus_playground.py, where this exact pattern turned a hard failure into a
# clean result seconds later). Before this, search() had zero retry protection,
# so one transient hiccup could kill an entire pipeline run for a question.
#
# CAVEAT (measured, not theory): ddgs raises DDGSException only when EVERY
# engine it tried came back empty, and the exception carries the LAST engine
# error it recorded -- so the engine named in the traceback is not necessarily
# the culprit. That also means a retry re-queries the same exhausted engines,
# which is why widening 3x8s -> 5x10s did NOT help a rate-limited run (it lost
# more than half the questions anyway). Retries only absorb genuinely transient blips; a
# sustained rate-limit needs time or fewer requests, not a longer retry window.
_RETRIES = 5
_RETRY_DELAY_S = 10

# Minimum spacing between outbound ddgs calls, enforced process-wide.
#
# The pipeline plans 2-3 queries per question and previously fired them
# back-to-back with zero delay, each one already a multi-engine burst -- then
# leaned on _RETRIES to dig out of the resulting lockout. That is backwards:
# the retry only fires once we are ALREADY blocked, and it re-queries the same
# exhausted engines (see the caveat above). "Add delays between requests" is
# the one concrete rate-limit remedy the ddgs docs, the maintainer, and the
# open ratelimit issue all agree on. Pacing prevents the lockout instead of
# trying to out-wait it.
#
# This throttles on call START times (classic rate limiting), so it costs
# nothing when calls are naturally slow -- it only sleeps when we would
# otherwise burst. Living here rather than in `search_node` means every caller
# is paced: the pipeline, and the eval playgrounds, and across questions in a
# run_eval sweep (which hammered the whole stress set in sequence).
#
# Assumes sequential callers (the pipeline's search_node is a serial loop). Not
# thread-safe by design -- if search() ever gets called concurrently, this needs
# a lock.
_MIN_QUERY_INTERVAL_S = 2.0
_last_call_ts: float = 0.0


def _throttle() -> None:
    """Sleep just long enough to keep ddgs calls `_MIN_QUERY_INTERVAL_S` apart."""
    global _last_call_ts
    wait = _MIN_QUERY_INTERVAL_S - (time.monotonic() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published_date: date | None = None
    source_engine: str | None = None


def search(
    query: str,
    *,
    max_results: int = 10,
    freshness: Literal["d", "w", "m", "y"] | None = None,
    mode: Literal["text", "news"] = "text",
    region: str = "us-en",
    backend: str | None = None,
) -> list[SearchResult]:
    """Run a metasearch query.

    Args:
        query: the search string.
        max_results: cap on results returned.
        freshness: ddgs `timelimit` — d/w/m/y. None = no filter.
        mode: "text" for general web, "news" for dated news results.
        region: ddgs region code.
        backend: override the engine list for this call (text mode only).
            Defaults to `_TEXT_BACKENDS`. Used by the corpus-freezing tools to
            pin a SINGLE engine, so a frozen fixture is one consistent source
            rather than a patchwork of whichever engine happened to be up.

    Text search is pinned to `_TEXT_BACKENDS` (see the note there — the ddgs
    "auto" default front-loads encyclopedias). News is left on ddgs's default:
    its registry is only bing/duckduckgo/yahoo, so there is no encyclopedia to
    exclude, and `brave` isn't a valid news engine.

    Calls are paced process-wide to `_MIN_QUERY_INTERVAL_S` apart (see
    `_throttle`), so callers can loop over queries without bursting.

    Retries up to `_RETRIES` times with a `_RETRY_DELAY_S` pause on a
    transient `DDGSException` (backend connection errors, spurious "No
    results found."). Raises the last exception if every attempt fails.
    """
    client = DDGS()

    last_exc: DDGSException | None = None
    for attempt in range(_RETRIES):
        _throttle()
        try:
            if mode == "news":
                raw = client.news(
                    query=query,
                    region=region,
                    timelimit=freshness,
                    max_results=max_results,
                )
                return [_from_news(r) for r in raw]

            raw = client.text(
                query=query,
                region=region,
                timelimit=freshness,
                max_results=max_results,
                backend=backend or _TEXT_BACKENDS,
            )
            return [_from_text(r) for r in raw]
        except DDGSException as e:
            last_exc = e
            if attempt < _RETRIES - 1:
                time.sleep(_RETRY_DELAY_S)
    raise last_exc


def _from_text(r: dict) -> SearchResult:
    return SearchResult(
        title=r.get("title", ""),
        url=r.get("href", ""),
        snippet=r.get("body", ""),
    )


def _from_news(r: dict) -> SearchResult:
    raw_date = r.get("date")
    parsed: date | None = None
    if isinstance(raw_date, str):
        try:
            parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
        except ValueError:
            parsed = None
    return SearchResult(
        title=r.get("title", ""),
        url=r.get("url", ""),
        snippet=r.get("body", ""),
        published_date=parsed,
        source_engine=r.get("source"),
    )
