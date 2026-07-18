"""Freeze stage 5: embed the frozen chunks into `eval/fixtures/embeddings.json`.

Reads `chunks.json`, embeds every chunk with the DOCUMENT-side task prefix, and
writes {chunk_id: vector}. This is the last expensive upstream step: with it
frozen, retrieval experiments cost nothing but local math.

DOCUMENT SIDE ONLY, deliberately. EmbeddingGemma is asymmetric -- `embedding.py`
wraps documents and queries in different task prefixes, and swapping them
silently costs ~10-20% recall. Chunk embeddings are stable, so they freeze well.
QUERY embeddings are NOT frozen: they change with every retrieval experiment
(and with your per-query retrieval idea), and they are one cheap call each.

TAGS ARE LOAD-BEARING, which is why they live in the artifact header. A vector
only means anything inside the vector space of the model that produced it. Embed
with one model, retrieve with another, and every cosine score is quietly wrong --
no error, just bad results. So we record `embed_model`, `backend`, `dim` and
`prefix`, and refuse to append to an artifact built by a different model unless
you pass --force.

Two known traps this guards:
  - `embeddinggemma:300m` (Colab) and `embeddinggemma:latest` (local) are the
    SAME model under different tags. A strict equality check would cry stale
    when nothing is wrong, so a tag mismatch WARNS and asks, rather than
    hard-failing on a false positive.
  - The openai/llama.cpp path L2-normalizes server-side (--embd-normalize 2)
    while the Ollama path does not necessarily, so the backend is recorded too:
    cosine floors tuned against one are not automatically valid for the other.

STALENESS IS FREE. `Chunk.id` is a content hash over (url, position, content),
so edited chunks get new ids: old vectors orphan themselves and new chunks show
up missing. Re-running embeds exactly what changed -- no cache invalidation to
reason about.

Resumable: saved after every batch, and an already-embedded chunk is skipped.

Needs the embedding endpoint. Check before running that OLLAMA_BASE_URL points
where you think, and OLLAMA_EMBED_MODEL matches THAT backend's tag.

Usage:
    uv run python eval/freeze_embeddings.py
    uv run python eval/freeze_embeddings.py --status     # report, embed nothing
    uv run python eval/freeze_embeddings.py --batch 32
    uv run python eval/freeze_embeddings.py --force      # re-embed everything
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

from answer.embedding import embed_documents  # noqa: E402

FIXTURES_DIR = _EVAL_DIR / "fixtures"
CHUNKS_PATH = FIXTURES_DIR / "chunks.json"
OUT_PATH = FIXTURES_DIR / "embeddings.json"
DEFAULT_BATCH = 64
ROUND_DP = 6  # 768 floats x ~2k chunks; full repr bloats the file for no gain


def _model_tag() -> tuple[str, str]:
    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    if backend == "openai":
        return backend, os.environ.get("OPENAI_EMBED_MODEL", "?")
    return backend, os.environ.get("OLLAMA_EMBED_MODEL", "?")


def load_existing() -> tuple[dict, dict]:
    """(header, {chunk_id: vector}) — empty if there's no artifact yet."""
    if not OUT_PATH.exists():
        return {}, {}
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return payload, payload.get("embeddings", {})


def save(vecs: dict[str, list[float]], backend: str, model: str,
         n_chunks: int) -> None:
    dim = len(next(iter(vecs.values()))) if vecs else 0
    FIXTURES_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend": backend,
                "embed_model": model,
                "prefix": "document",  # embed_documents(); queries are NOT frozen
                "dim": dim,
                "n_embedded": len(vecs),
                "n_chunks_total": n_chunks,
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
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"chunks per embed call (default {DEFAULT_BATCH})")
    args = ap.parse_args()

    if not CHUNKS_PATH.exists():
        print(f"missing {CHUNKS_PATH.relative_to(_PROJECT_ROOT)} — "
              "run `uv run python eval/freeze_chunks.py` first")
        sys.exit(2)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]
    backend, model = _model_tag()
    header, vecs = ({}, {}) if args.force else load_existing()

    # A vector only means something in its own model's space. Appending to an
    # artifact built by a different model would silently mix two spaces.
    prev_model = header.get("embed_model")
    if vecs and prev_model and prev_model != model:
        print(f"TAG MISMATCH: artifact was built with {prev_model!r}, "
              f"env says {model!r}.")
        print("  If these are the same model under different tags "
              "(embeddinggemma:300m == embeddinggemma:latest), it's safe.")
        print("  If they are genuinely different models, the stored vectors are "
              "in a different space — re-run with --force to rebuild.")
        print("Refusing to mix. Fix the env var, or pass --force.")
        sys.exit(2)

    todo = [c for c in chunks if c["id"] not in vecs]
    print(f"{len(chunks)} chunks · {len(chunks) - len(todo)} already embedded · "
          f"{len(todo)} to embed")
    print(f"backend={backend} · model={model} · document-side prefix")
    if header.get("dim"):
        print(f"existing dim={header['dim']}")

    if args.status:
        sys.exit(0)
    if not todo:
        print("\nnothing to do — every chunk is embedded.")
        sys.exit(0)

    print()
    total_batches = (len(todo) + args.batch - 1) // args.batch
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        n = i // args.batch + 1
        try:
            out = embed_documents([c["text"] for c in batch])
        except Exception as exc:  # noqa: BLE001
            print(f"[batch {n}/{total_batches}] FAILED — "
                  f"{type(exc).__name__}: {exc}")
            print("stopping; re-run to resume (embedded chunks are skipped).")
            break
        for c, v in zip(batch, out):
            vecs[c["id"]] = [round(float(x), ROUND_DP) for x in v]
        save(vecs, backend, model, len(chunks))  # persist per batch
        print(f"[batch {n}/{total_batches}] {len(vecs)}/{len(chunks)} embedded",
              flush=True)

    save(vecs, backend, model, len(chunks))
    missing = [c for c in chunks if c["id"] not in vecs]
    dim = len(next(iter(vecs.values()))) if vecs else 0
    print()
    print("=" * 60)
    print(f"embedded: {len(vecs)}/{len(chunks)} · dim={dim}  ->  "
          f"{OUT_PATH.relative_to(_PROJECT_ROOT)}")
    if missing:
        print(f"MISSING {len(missing)} — re-run to finish.")
    else:
        print("all chunks embedded. the freeze chain is complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
