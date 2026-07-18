"""Live search integration tests.

These hit ddgs (and through it, real search engines), so they're marked
`network`. Skip with `pytest -m 'not network'` for offline runs.
"""
from __future__ import annotations

import pytest

from answer.search import SearchResult, search


@pytest.mark.network
def test_search_text_returns_results():
    results = search("python programming language", max_results=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.network
def test_search_text_results_have_urls_and_titles():
    results = search("langgraph framework", max_results=3)
    assert results, "expected at least one result"
    for r in results:
        assert r.url.startswith("http"), f"bad url: {r.url!r}"
        assert r.title, "title should be non-empty"


@pytest.mark.network
def test_search_news_returns_dated_results():
    results = search("openai", max_results=3, mode="news", freshness="m")
    assert results, "expected at least one news result"
    # At least one news result should carry a parsed published_date.
    assert any(r.published_date is not None for r in results), (
        "expected at least one news result with a parsed published_date"
    )
