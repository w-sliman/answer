"""EmbeddingGemma client, called via Ollama's /api/embed endpoint.

We deliberately bypass langchain-ollama's OllamaEmbeddings class to keep full
control over EmbeddingGemma's asymmetric task prompts. Those prefixes wrap
the input before embedding:

- Documents: `title: none | text: <chunk>`
- Queries:   `task: search result | query: <question>`

Skipping (or swapping) the prefix silently degrades recall by ~10-20% — no
error, just worse retrieval. Wrapping both sides in this module is the single
chokepoint that makes "wrong prefix" structurally impossible at call sites.

Endpoint URL and model name come from env so the ephemeral ngrok Colab tunnel
can change between sessions without code edits:

    OLLAMA_BASE_URL   — same tunnel as the chat model
    OLLAMA_EMBED_MODEL — typically `embeddinggemma:300m` or `embeddinggemma`
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import httpx

from .config import EMBED_DOCUMENT_PREFIX_TEMPLATE, EMBED_QUERY_PREFIX_TEMPLATE

# Conservative timeout — embedding 100+ chunks at once over the ngrok tunnel
# can take a while. Don't bail just because the first run takes 30s.
_EMBED_TIMEOUT_S = 120.0

# The ephemeral ngrok/Colab tunnel occasionally drops a connection or has a
# transient DNS blip (seen in the wild: httpx.ConnectError mid-pipeline-run,
# killing the whole question with no fallback). Same retry treatment as
# answer.search.search() for the same reason: one flaky network call
# shouldn't be a hard failure.
_EMBED_RETRIES = 3
_EMBED_RETRY_DELAY_S = 8


def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "ollama").strip().lower()


@lru_cache(maxsize=1)
def _client_config() -> tuple[str, str, str]:
    """Returns (backend, base_url, model) for the configured embedding backend.

    - ollama: OLLAMA_BASE_URL (+ /api/embed) and OLLAMA_EMBED_MODEL.
    - openai: OPENAI_BASE_URL (+ /embeddings) and OPENAI_EMBED_MODEL — e.g. a
      llama.cpp llama-server launched with --embedding. OPENAI_BASE_URL already
      ends in /v1, so the endpoint is `{base}/embeddings`.
    """
    backend = _backend()
    if backend == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_EMBED_MODEL")
        if not base_url or not model:
            raise RuntimeError(
                "OPENAI_BASE_URL and OPENAI_EMBED_MODEL must be set when "
                "LLM_BACKEND=openai (see .env.example)."
            )
        return backend, base_url.rstrip("/"), model

    base_url = os.environ.get("OLLAMA_BASE_URL")
    model = os.environ.get("OLLAMA_EMBED_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OLLAMA_BASE_URL and OLLAMA_EMBED_MODEL must be set in the "
            "environment. OLLAMA_BASE_URL is the same Colab+ngrok tunnel as "
            "the chat model; OLLAMA_EMBED_MODEL is typically "
            "'embeddinggemma:300m'."
        )
    return backend, base_url.rstrip("/"), model


def _post_embed(inputs: list[str]) -> list[list[float]]:
    """Single POST to /api/embed. Returns one embedding vector per input.

    Ollama's /api/embed accepts a string or a list of strings under the
    `input` key. We always send a list (even for single items) so the
    response shape is uniform.

    Retries up to `_EMBED_RETRIES` times with a `_EMBED_RETRY_DELAY_S` pause
    on a transient `httpx.TransportError` (connection errors, DNS blips,
    timeouts). Raises the last exception if every attempt fails.
    """
    if not inputs:
        return []
    backend, base_url, model = _client_config()
    if backend == "openai":
        endpoint = f"{base_url}/embeddings"      # base_url ends in /v1
    else:
        endpoint = f"{base_url}/api/embed"
    payload = {"model": model, "input": inputs}

    last_exc: httpx.TransportError | None = None
    for attempt in range(_EMBED_RETRIES):
        try:
            with httpx.Client(timeout=_EMBED_TIMEOUT_S) as client:
                r = client.post(endpoint, json=payload)
                r.raise_for_status()
                data = r.json()
            break
        except httpx.TransportError as e:
            last_exc = e
            if attempt < _EMBED_RETRIES - 1:
                time.sleep(_EMBED_RETRY_DELAY_S)
    else:
        raise last_exc

    if backend == "openai":
        # OpenAI shape: {"data": [{"embedding": [...], "index": i}, ...]}.
        # Sort by index defensively; llama.cpp should already return in order.
        items = data.get("data")
        if items is None:
            raise RuntimeError(
                f"OpenAI /embeddings returned no `data` field. "
                f"Response keys: {list(data.keys())}"
            )
        embeddings = [
            it["embedding"] for it in sorted(items, key=lambda it: it.get("index", 0))
        ]
    else:
        # Ollama returns {"embeddings": [[...], [...]]} for /api/embed (plural).
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(
                f"Ollama /api/embed returned no `embeddings` field. "
                f"Response keys: {list(data.keys())}"
            )
    if len(embeddings) != len(inputs):
        raise RuntimeError(
            f"Embedding count mismatch: requested {len(inputs)}, got "
            f"{len(embeddings)}. Likely an Ollama API version skew."
        )
    return embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed N chunk texts with the document-side task prefix.

    Use this for everything that lands in the vector store: page chunks,
    cached corpus content, anything that will be *retrieved against* later.
    """
    prefixed = [EMBED_DOCUMENT_PREFIX_TEMPLATE.format(text=t) for t in texts]
    return _post_embed(prefixed)


def embed_query(text: str) -> list[float]:
    """Embed one query string with the query-side task prefix.

    Use this for the retrieval query — the thing we compare against indexed
    chunks. Single-input convenience; calls the batched endpoint with one
    element so there's only one code path.
    """
    prefixed = EMBED_QUERY_PREFIX_TEMPLATE.format(text=text)
    vecs = _post_embed([prefixed])
    return vecs[0]
