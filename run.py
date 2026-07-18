"""Entry point. Loads env, runs the LangGraph pipeline, prints + saves a trace.

Usage:
    uv run python run.py "your question here"

The console shows each stage as it runs (via the pipeline's own prints).
A JSON snapshot of the final state lands in `traces/`. LangSmith captures
the LLM calls remotely when LANGSMITH_TRACING=true is set in `.env`.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

# Silence langgraph's transitive `allowed_objects` deprecation warning fired
# at import time. Filter by class (reliable) plus message regex (belt-and-
# suspenders).
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=r"The default value of .allowed_objects.")

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from answer.pipeline import answer_question  # noqa: E402

TRACES_DIR = Path(__file__).parent / "traces"


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text)[:n].strip("_")
    return s or "run"


def _to_jsonable(obj):
    """Recursively convert dataclasses + Pydantic models to plain dicts.

    Drops `embedding` keys from any dict on the way through — the 768-dim
    EmbeddingGemma vectors carried on RetrievalHit aren't useful in traces
    and would balloon the JSON by ~kBs per chunk. They live in the Chroma
    collection if we ever need them again offline.
    """
    if isinstance(obj, BaseModel):
        return _to_jsonable(obj.model_dump())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {
            k: _to_jsonable(v) for k, v in obj.items() if k != "embedding"
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_trace(state: dict, question: str) -> Path:
    TRACES_DIR.mkdir(exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(question)}.json"
    path = TRACES_DIR / fname
    path.write_text(json.dumps(_to_jsonable(state), indent=2, default=str))
    return path


async def _amain(question: str) -> None:
    state = await answer_question(question)
    path = _save_trace(state, question)

    # Pretty-print the final answer + sources for the human at the keyboard.
    answer = state.get("final_answer")
    if answer is not None:
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer.text)
        print("\nSources:")
        for c in answer.citations:
            print(f"  [{c.n}] {c.url}")
            for q in c.quotes:
                snippet = q if len(q) <= 120 else q[:117] + "..."
                print(f'        • "{snippet}"')

    print(f"\n[trace saved to {path}]")


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print('Usage: uv run python run.py "your question"')
        sys.exit(1)
    asyncio.run(_amain(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
