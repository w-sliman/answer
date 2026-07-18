"""Page fetch wrapper around crawl4ai.

Returns LLM-ready markdown for a URL. Uses crawl4ai's PruningContentFilter
to strip nav/footer/sidebar boilerplate so what reaches the LLM is the
article body, not page chrome. Swap the backend by replacing the body of
these functions — the interface (`url -> str`) stays the same.

Notes drawn from docs/crawl4ai_quickstart.py:
- `result.markdown` is a MarkdownGenerationResult object, not a string.
  `.fit_markdown` is the pruned, LLM-ready version; `.raw_markdown` is full.
- excluded_tags + remove_overlay_elements remove common page chrome.
- exclude_*_links cuts noise from external/social links.
"""
from __future__ import annotations

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


def _run_config(query: str | None = None) -> CrawlerRunConfig:
    """Build a crawler config. When `query` is provided, uses BM25 to keep
    only spans relevant to the query (much higher signal-to-noise for our
    RAG use case). Falls back to generic prose-pruning when no query is
    available (e.g. ad-hoc `fetch_page(url)` calls without a question).
    """
    if query:
        content_filter = BM25ContentFilter(user_query=query, bm25_threshold=1.0)
    else:
        content_filter = PruningContentFilter(
            threshold=0.48,
            threshold_type="fixed",
            min_word_threshold=0,
        )
    return CrawlerRunConfig(
        # BYPASS for v0 — caching can mask bugs (we hit one where an empty
        # cached entry kept getting served). Reintroduce thoughtfully later.
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["nav", "footer", "aside"],
        remove_overlay_elements=True,
        exclude_external_links=True,
        exclude_social_media_links=True,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={"ignore_links": True},
        ),
        # Silence the per-URL [FETCH]/[SCRAPE]/[COMPLETE] progress lines
        # that crawl4ai emits to stdout by default. The pipeline's own
        # status prints summarize fetch outcomes; we don't need crawl4ai's
        # progress chatter on top.
        verbose=False,
    )


_MIN_FIT_CHARS = 200  # below this we treat fit_markdown as "filter overshot"


def _extract_markdown(result) -> str:
    """Prefer fit_markdown (pruned for LLMs); fall back to raw if it's trivially short.

    PruningContentFilter can over-prune some sites (Wikipedia is one), leaving
    fit_markdown nearly empty. A naive `fit or raw` falls into the truthiness
    trap because '\\n' is truthy. We require fit to be substantively long
    before preferring it.
    """
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    fit = (getattr(md, "fit_markdown", None) or "").strip()
    raw = (getattr(md, "raw_markdown", None) or "").strip()
    if len(fit) >= _MIN_FIT_CHARS:
        return fit
    return raw


async def fetch_page(url: str, query: str | None = None) -> str:
    """Fetch one URL, return its main content as markdown ('' on failure).

    When `query` is provided, BM25 is used to keep only spans relevant to
    the query. Without a query, falls back to generic prose-pruning.
    """
    config = _run_config(query)
    # `verbose=False` here suppresses crawl4ai's startup banner (the
    # "[INIT] → Crawl4AI x.y.z" line). The per-URL progress lines are
    # silenced by the same flag on CrawlerRunConfig.
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=config)
        if not result.success:
            return ""
        return _extract_markdown(result)


async def fetch_pages(urls: list[str], query: str | None = None) -> dict[str, str]:
    """Fetch multiple URLs concurrently via crawl4ai's `arun_many`.

    Uses crawl4ai's built-in batch API — a single `AsyncWebCrawler` context
    handles all URLs, and its internal dispatcher manages concurrency,
    browser-instance lifecycle, and memory pressure. This is the officially
    recommended pattern for multi-URL fetches and what the recent v0.7.x
    releases have been actively improving:

      - "Dispatcher Bug Fix: Fixed sequential processing bottleneck in
         arun_many for fast-completing tasks"
      - "Browser Manager Fixes: Resolved race conditions in concurrent
         page creation with thread-safe locking"
      - "Memory Management Refactor"

    Note on v0's earlier known issue. We previously avoided sharing a
    single crawler context because manually looping `crawler.arun()` calls
    inside one `async with` block produced empty-markdown bugs on some
    sites (see `known_issues.md`). `arun_many` is a *different* code
    path — purpose-built for batching, with its own dispatcher — and the
    bugs it had were the ones the v0.7.x release notes call out as fixed.
    If we ever see the empty-markdown regression resurface here, the
    fallback is a one-line revert to a sequential `for url in urls` loop.

    `query` is forwarded so each page is BM25-filtered to spans relevant
    to the question (when provided) or generically pruned (when not).

    Empty input returns `{}` immediately — no crawler launch, no overhead.
    Results are matched back to URLs by each `CrawlResult.url`, not by list
    position — `arun_many`'s dispatcher completes URLs out of request order,
    so a positional zip silently mispairs content with the wrong URL. One
    entry per requested URL is guaranteed (a URL crawl4ai never returns a
    result for is recorded as `""`, same as an explicit failure).
    """
    if not urls:
        return {}
    config = _run_config(query)
    out: dict[str, str] = {}
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            results = await crawler.arun_many(urls, config=config)
    except Exception as e:
        # Batch-level failure (browser launch, dispatcher crash, etc.).
        # Surface it as all-empty rather than killing the pipeline — the
        # downstream code treats empty pages as "fetch failed, drop this
        # URL", which is the right behavior here.
        print(f"[fetch_pages] arun_many failed: {e!r}")
        return {url: "" for url in urls}

    # `arun_many`'s dispatcher runs URLs concurrently, so `results` comes
    # back in COMPLETION order, not request order — zipping positionally
    # against `urls` silently pairs each URL with whatever finished in that
    # slot (confirmed in the wild: a fetched page's content landed under a
    # sibling URL while its own URL showed up "missing"). Match by each
    # result's own `.url` instead, which crawl4ai always sets.
    for result in results:
        key = getattr(result, "url", None) if result is not None else None
        if key is None:
            continue
        if not getattr(result, "success", False):
            print(f"[fetch_pages] {key} failed")
            out[key] = ""
            continue
        out[key] = _extract_markdown(result)
    # Any requested URL crawl4ai never returned a result for at all (batch
    # dropped it) is treated as a failure too, same as an explicit one.
    for url in urls:
        out.setdefault(url, "")
    return out
