"""Shared test fixtures.

Loads `.env` so tests that touch the LLM or LangSmith pick up the same
config the app uses at runtime.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
