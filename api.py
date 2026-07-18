"""FastAPI serving layer for Project Answer — the live pipeline over HTTP/SSE.

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000
    # then open http://127.0.0.1:8000  (serves web/dist if it's been built)

Endpoints:
    GET  /health   -> {"ok": true}
    POST /ask      -> Server-Sent Events. Streams the pipeline's native
                      structured events (stage_start / stage_data / stage_end)
                      and status lines AS THEY HAPPEN, then a final `answer`
                      frame (the cited answer), then `done`.
    GET  /*        -> the built React frontend (web/dist), if present.

This exposes the pipeline's structured event channel over SSE so a browser can
watch stages run: the async pipeline runs in a daemon worker thread (its own event
loop via asyncio.run), and its event/status callbacks are bridged into the
request's event loop through `call_soon_threadsafe`. That keeps the SSE
generator responsive even while a node does blocking CPU work (cross-encoder,
Chroma), instead of stalling the whole stream.

SINGLE concurrent run. The pipeline's event/status handlers are module-global
(see set_event_handler docstring), so an asyncio.Lock rejects overlapping
requests with 409. Fine for a demo; a multi-tenant deployment would need
per-request handler scoping (contextvars).
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from answer.pipeline import answer_question, set_event_handler, set_status_handler

load_dotenv()

app = FastAPI(title="Answer — agentic web search + RAG")

# Permissive CORS: lets the Vite dev server (localhost:5173) hit this API while
# developing. In the built/served setup the frontend is same-origin (mounted
# below), so this only matters for local dev. It's a single-user demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One run at a time — the pipeline's handlers are module-global.
_run_lock = asyncio.Lock()


class AskBody(BaseModel):
    question: str


def _sse(event: str, data: Any) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _answer_payload(state: dict) -> dict:
    """Serialize the final pipeline state into the frame the browser needs:
    the cited answer plus the run-level metrics (iterations, grounding)."""
    answer = state.get("final_answer")
    if answer is None:
        return {"error": "no answer produced"}

    attrib = state.get("attribution")
    n_unsupported = (
        sum(1 for c in attrib.claims if not c.supported) if attrib is not None else None
    )
    sufficiency = state.get("sufficiency")
    disclaim = bool(sufficiency is not None and not sufficiency.sufficient)

    return {
        "answer": answer.model_dump(),  # text, citations, placements, claims
        "disclaim": disclaim,
        "iteration": state.get("iteration"),
        "answer_attempts": state.get("answer_attempts"),
        "n_unsupported": n_unsupported,
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/ask")
async def ask(body: AskBody) -> StreamingResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if _run_lock.locked():
        raise HTTPException(
            status_code=409, detail="a run is already in progress (single-user demo)"
        )

    async def stream():
        async with _run_lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def push(item: tuple[str, Any]) -> None:
                # Called from the worker thread → hop onto the request loop.
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def worker() -> None:
                # Install handlers, run the pipeline in this thread's own event
                # loop, always uninstall on the way out.
                set_event_handler(lambda ev: push(("event", ev)))
                set_status_handler(lambda line: push(("status", line)))
                try:
                    state = asyncio.run(answer_question(question))
                    push(("done", state))
                except Exception as exc:  # noqa: BLE001 — surfaced to the client
                    push(("error", exc))
                finally:
                    set_event_handler(None)
                    set_status_handler(None)

            threading.Thread(target=worker, daemon=True).start()

            yield _sse("start", {"question": question})
            while True:
                kind, payload = await queue.get()
                if kind == "event":
                    yield _sse("event", payload)
                elif kind == "status":
                    yield _sse("status", {"line": payload})
                elif kind == "done":
                    yield _sse("answer", _answer_payload(payload))
                    yield _sse("done", {})
                    return
                elif kind == "error":
                    yield _sse(
                        "error",
                        {"message": f"{type(payload).__name__}: {payload}"},
                    )
                    return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built frontend, if it exists. Mounted LAST so the explicit API
# routes above win; everything else falls through to the static SPA. Build it
# with `npm run build` in web/ (outputs web/dist). Absent in a fresh checkout —
# the API still works headless (POST /ask) without it.
_DIST = Path(__file__).resolve().parent / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="site")
