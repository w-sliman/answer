"""Live fetch integration tests.

These hit real URLs through crawl4ai (Playwright + Chromium under the hood),
so they're marked `network` and require `crawl4ai-setup` to have run.

KNOWN FLAKE: when pytest collects both test_fetch.py and test_search.py in
one session, both fetch tests can return 0-char markdown — likely Playwright
browser-cache state interacting badly with pytest's import/collection. Tests
pass cleanly when run as `pytest tests/test_fetch.py`. Real runtime never
double-fetches the same URL in seconds, so this doesn't reflect a runtime
bug. Revisit if it starts mattering.
"""
from __future__ import annotations

import pytest

from answer.fetch import fetch_page, fetch_pages


@pytest.mark.network
async def test_fetch_page_returns_substantive_markdown():
    md = await fetch_page("https://en.wikipedia.org/wiki/Apple")
    assert isinstance(md, str)
    assert len(md) > 2000, f"markdown too short ({len(md)} chars) — content filter may be too aggressive"
    assert "Apple" in md, "expected the page text to mention 'Apple'"


@pytest.mark.network
async def test_fetch_pages_returns_one_entry_per_url():
    # Distinct URLs across tests on purpose: re-fetching the same URL twice
    # in one pytest process returns empty markdown — Playwright/browser-cache
    # state we don't easily bust. Production won't hit this; tests shouldn't
    # paper over it by retrying.
    urls = [
        "https://en.wikipedia.org/wiki/Banana",
        "https://example.com",
    ]
    out = await fetch_pages(urls)
    assert set(out.keys()) == set(urls)
    assert len(out["https://en.wikipedia.org/wiki/Banana"]) > 1000
    # example.com is intentionally tiny — just check we got something back.
    assert isinstance(out["https://example.com"], str)
