"""Freeze stage 6: embed the frozen questions and queries (QUERY-side prefix).

Writes `eval/fixtures/query_embeddings.json` == {text: vector} for every string
we could ever retrieve WITH: the 11 dataset questions and the 15 frozen planner
queries. With this, the whole retrieval stage runs with NOTHING served -- cosine
against frozen vectors is pure local math -- leaving answer generation as the
only live model call downstream.

WHY THIS IS SAFE TO FREEZE (a correction). The first cut of the freeze chain
argued query vectors must stay live because "they change per experiment". That
was wrong: the queries are frozen in queries.json, and per-query retrieval uses
those same strings. Tuning top_k, MMR lambda or the reranker changes none of the
text. The set of strings we could embed on the query side is small and already
known -- 15 questions + 22 queries -- so it freezes as cleanly as the chunks do.

SEPARATE FILE FROM embeddings.json, ON PURPOSE. EmbeddingGemma is asymmetric:
`embed_query` and `embed_documents` wrap text in DIFFERENT task prefixes, and
swapping them silently costs ~10-20% recall (see embedding.py). Two artifacts,
each recording its own `prefix`, makes that mistake structurally hard instead of
something you have to remember. Never merge these maps.

BOTH questions AND queries, because they back different baselines:
  - production `retrieve_node` retrieves with the QUESTION embedding
  - per-query retrieval retrieves with each QUERY embedding
Freezing both means that A/B costs no endpoint.

Keyed by the text itself: lookup is trivial and staleness is visible -- edit a
query and it is simply absent, so a re-run embeds exactly what changed.

Needs the embedding endpoint (~26 calls, seconds). Check OLLAMA_BASE_URL points
where you think and OLLAMA_EMBED_MODEL matches THAT backend's tag.

Usage:
    uv run python eval/freeze_query_embeddings.py
    uv run python eval/freeze_query_embeddings.py --status
    uv run python eval/freeze_query_embeddings.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.embedding import embed_query  # noqa: E402

FIXTURES_DIR = _EVAL_DIR / "fixtures"
QUERIES_PATH = FIXTURES_DIR / "queries.json"
OUT_PATH = FIXTURES_DIR / "query_embeddings.json"
ROUND_DP = 6


def _model_tag() -> tuple[str, str]:
    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    if backend == "openai":
        return backend, os.environ.get("OPENAI_EMBED_MODEL", "?")
    return backend, os.environ.get("OLLAMA_EMBED_MODEL", "?")


def load_existing() -> tuple[dict, dict]:
    if not OUT_PATH.exists():
        return {}, {}
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return payload, payload.get("embeddings", {})


def save(vecs: dict[str, list[float]], backend: str, model: str,
         kinds: dict[str, str]) -> None:
    dim = len(next(iter(vecs.values()))) if vecs else 0
    FIXTURES_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend": backend,
                "embed_model": model,
                # NOT interchangeable with embeddings.json (prefix=document).
                "prefix": "query",
                "dim": dim,
                "n": len(vecs),
                # what each string is, so consumers can pick a baseline
                "kinds": kinds,
                "embeddings": vecs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true",
                    help="report what's embedded and exit; embeds nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-embed everything, ignoring what's stored")
    args = ap.parse_args()

    if not QUERIES_PATH.exists():
        print(f"missing {QUERIES_PATH.relative_to(_PROJECT_ROOT)} — "
              "run `uv run python eval/freeze_queries.py` first")
        sys.exit(2)

    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    rows = payload["questions"]

    # kinds: text -> "question" | "query". A string is recorded once even if it
    # somehow appears as both; questions are added first so they win the label.
    kinds: dict[str, str] = {}
    for r in rows:
        kinds.setdefault(r["question"], "question")
    for r in rows:
        for rd in r.get("rounds", []):          # queries.json groups queries by planner round
            for q in rd.get("queries", []):
                kinds.setdefault(q["query"], "query")

    backend, model = _model_tag()
    header, vecs = ({}, {}) if args.force else load_existing()

    prev_model = header.get("embed_model")
    if vecs and prev_model and prev_model != model:
        print(f"TAG MISMATCH: artifact was built with {prev_model!r}, "
              f"env says {model!r}.")
        print("  Same model under different tags (embeddinggemma:300m == "
              "embeddinggemma:latest) is safe; genuinely different models are "
              "a different vector space.")
        print("Refusing to mix. Fix the env var, or pass --force.")
        sys.exit(2)

    todo = [t for t in kinds if t not in vecs]
    n_q = sum(1 for k in kinds.values() if k == "question")
    n_s = sum(1 for k in kinds.values() if k == "query")
    print(f"{len(kinds)} strings ({n_q} questions + {n_s} queries) · "
          f"{len(kinds) - len(todo)} already embedded · {len(todo)} to embed")
    print(f"backend={backend} · model={model} · QUERY-side prefix")

    if args.status:
        sys.exit(0)
    if not todo:
        print("\nnothing to do — everything is embedded.")
        sys.exit(0)

    print()
    for i, text in enumerate(todo, start=1):
        try:
            v = embed_query(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] FAILED — {type(exc).__name__}: {exc}")
            print("stopping; re-run to resume (embedded strings are skipped).")
            break
        vecs[text] = [round(float(x), ROUND_DP) for x in v]
        save(vecs, backend, model, kinds)  # persist every string
        print(f"[{i}/{len(todo)}] {kinds[text]:<8} {text[:62]!r}", flush=True)

    save(vecs, backend, model, kinds)
    missing = [t for t in kinds if t not in vecs]
    dim = len(next(iter(vecs.values()))) if vecs else 0
    print()
    print("=" * 60)
    print(f"embedded: {len(vecs)}/{len(kinds)} · dim={dim}  ->  "
          f"{OUT_PATH.relative_to(_PROJECT_ROOT)}")
    if missing:
        print(f"MISSING {len(missing)} — re-run to finish.")
    else:
        print("all questions + queries embedded. retrieval now needs NOTHING "
              "served.")
    print("=" * 60)


if __name__ == "__main__":
    main()
