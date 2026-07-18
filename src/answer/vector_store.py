"""Chroma facade for per-job persistent vector storage.

One Chroma collection per job. For v1.0 Answer (one question = one job)
that's one collection per question; later, the competitive-analysis system
will accumulate a job's chunks across many questions in the same collection.
The schema is identical either way; only the collection-creation policy
differs.

We hold embeddings ourselves (via `embedding.py`) rather than letting Chroma
own an embedding function, because EmbeddingGemma needs asymmetric task
prefixes that Chroma's default embedders don't know about. Letting Chroma
embed would silently apply the wrong prompting.

Key design choices:

- **Skip-if-exists upserts.** Chunk IDs are deterministic over content, so
  re-running the same fetch over the same content is a free no-op (no
  re-embedding). This is what makes the persistence model pay off as soon
  as a job has more than one question.
- **`embeddings` is excluded from query results by default.** We only need
  vectors at index time. At retrieval time we get back ids, documents,
  metadatas, and distances — the embedding bytes would be wasted IO.
- **The chunk_id -> URL map is recoverable from metadata.** No sidecar
  needed; the answer renderer reads `metadata["url"]` to resolve a cited
  chunk back to its source URL.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from .chunking import Chunk
from .config import VECTOR_STORE_PATH
from .embedding import embed_documents


# Module-level client — file-backed, safe to share across calls in one process.
# Chroma uses SQLite + parquet under the hood; no daemon, no second process.
_CLIENT: chromadb.api.ClientAPI | None = None


def _client() -> chromadb.api.ClientAPI:
    global _CLIENT
    if _CLIENT is None:
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _CLIENT


def _collection_name(job_id: str) -> str:
    """Chroma's collection-name rules: 3–63 chars, alphanumeric + `_-`,
    must start and end alphanumeric. We prefix with `job_` so a UUID
    (`-` and digits) becomes a valid name and the namespace is obvious.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
    return f"job_{safe}"[:63]


def get_collection(job_id: str) -> Collection:
    """Return (or create) the Chroma collection for this job.

    `metadata={"hnsw:space": "cosine"}` makes cosine distance the default
    similarity metric, which matches the symmetric setup our reranker and
    MMR expect. The default in Chroma is L2 — fine for normalized
    embeddings, but cosine is unambiguous and EmbeddingGemma's outputs
    aren't guaranteed to be unit-normed.
    """
    return _client().get_or_create_collection(
        name=_collection_name(job_id),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(job_id: str) -> None:
    """Delete and recreate a job's collection. Useful between eval runs
    when we want a clean slate, or when chunks have stale embeddings after
    an embedding-model change.
    """
    name = _collection_name(job_id)
    client = _client()
    try:
        client.delete_collection(name=name)
    except Exception:
        # Doesn't exist — fine, the get_or_create on the next call handles it.
        pass


def upsert_chunks(collection: Collection, chunks: list[Chunk]) -> int:
    """Embed and upsert any chunks whose IDs aren't already in `collection`.

    Returns the number of NEW chunks embedded. Returns 0 if everything was
    already cached — that's the persistence win.

    We check existence in one `collection.get(ids=...)` call (Chroma returns
    the IDs it found), diff against the requested IDs, embed only the diff,
    then upsert. `add` would raise on duplicates; `upsert` would re-embed
    them — neither is what we want. Diff-then-add is the right pattern.
    """
    if not chunks:
        return 0

    requested_ids = [c.id for c in chunks]
    existing = set(collection.get(ids=requested_ids, include=[])["ids"])
    new_chunks = [c for c in chunks if c.id not in existing]
    if not new_chunks:
        return 0

    embeddings = embed_documents([c.text for c in new_chunks])
    collection.add(
        ids=[c.id for c in new_chunks],
        embeddings=embeddings,
        documents=[c.text for c in new_chunks],
        metadatas=[
            {"url": c.url, "domain": c.domain, "position": c.position}
            for c in new_chunks
        ],
    )
    return len(new_chunks)


@dataclass
class RetrievalHit:
    """One hit back from the vector store. The score is cosine *similarity*
    (1 - distance), so higher = more similar — matches the convention the
    reranker and MMR use.

    `rerank_score` is populated by the rerank node (Phase 2+) and stays
    None when reranking is bypassed. Downstream MMR prefers rerank_score
    when present and falls back to retrieval_score otherwise — same
    interface either way.
    """

    chunk_id: str
    text: str
    url: str
    domain: str
    position: int
    retrieval_score: float
    embedding: list[float]
    rerank_score: float | None = None


def similarity_search(
    collection: Collection,
    query_embedding: list[float],
    k: int,
) -> list[RetrievalHit]:
    """Top-k cosine search against the collection.

    We request `embeddings` back because downstream MMR (Phase 2) needs
    them to compute diversity. They're not free over the wire, but the
    extra payload is small (k≈40 × 768 floats ≈ 120KB) and avoiding a
    second round-trip to fetch them later is worth it.
    """
    if k <= 0:
        return []
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]
    embs = raw["embeddings"][0]
    hits: list[RetrievalHit] = []
    for cid, doc, meta, dist, emb in zip(ids, docs, metas, dists, embs):
        # Chroma with `hnsw:space=cosine` returns cosine *distance* in [0, 2].
        # Convert to similarity in roughly [-1, 1] for downstream consistency.
        similarity = 1.0 - float(dist)
        hits.append(
            RetrievalHit(
                chunk_id=cid,
                text=doc,
                url=str(meta.get("url", "")),
                domain=str(meta.get("domain", "")),
                position=int(meta.get("position", 0)),
                retrieval_score=similarity,
                embedding=list(emb),
            )
        )
    return hits
