"""LLM client. Wraps langchain-ollama's ChatOllama OR langchain-openai's ChatOpenAI.

The backend is selected by the `LLM_BACKEND` env var:

- ``"ollama"`` (default): ChatOllama pointed at OLLAMA_BASE_URL / OLLAMA_MODEL —
  the original Colab/local Ollama path (Gemma 4 E2B).
- ``"openai"``: ChatOpenAI pointed at an OpenAI-compatible server
  (OPENAI_BASE_URL / OPENAI_MODEL) — e.g. a llama.cpp ``llama-server``, used to
  serve the larger Gemma variants (12B/26B/31B) that don't run as E2B.

Both paths stay available so the same pipeline can be A/B'd across backends by
flipping one env var. Endpoint/model still come from the environment so an
ephemeral tunnel URL can change without code edits.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "ollama").strip().lower()


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.2, reasoning: bool = False) -> BaseChatModel:
    """Build (or reuse) a chat model for the configured backend.

    `reasoning=True` enables the model's thinking pass before producing the
    final structured output. Use it for nodes where deliberation is worth the
    ~2-3x latency (link picking, final answer); skip it for mechanical
    transforms (query generation). The parsed structured output shape is
    unchanged either way.

    - Ollama: native `reasoning` flag; thinking lands in the response's
      `additional_kwargs["reasoning_content"]`. Honors OLLAMA_NUM_CTX.
    - OpenAI: maps to Gemma's chat-template switch via
      `extra_body={"chat_template_kwargs": {"enable_thinking": ...}}`; context
      length is governed server-side (llama-server's `-c`), so OLLAMA_NUM_CTX
      does not apply.
    """
    if _backend() == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL")
        if not base_url or not model:
            raise RuntimeError(
                "OPENAI_BASE_URL and OPENAI_MODEL must be set when "
                "LLM_BACKEND=openai (see .env.example)."
            )
        return ChatOpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            model=model,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": reasoning}},
        )

    # default: ollama (unchanged behavior)
    base_url = os.environ.get("OLLAMA_BASE_URL")
    model = os.environ.get("OLLAMA_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "OLLAMA_BASE_URL and OLLAMA_MODEL must be set in the environment "
            "(see .env.example). The ngrok URL changes per Colab session."
        )
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        reasoning=reasoning,
    )


def with_schema(llm: BaseChatModel, schema) -> Runnable:
    """Bind a Pydantic schema for structured output, choosing the method per backend.

    llama.cpp's OpenAI server does grammar-constrained JSON reliably via
    `method="json_schema"`, whereas its function/tool-calling support depends on
    the chat template — so we force json_schema on the OpenAI path. Ollama keeps
    its default native structured output, preserving prior behavior exactly.

    Use this at call sites instead of `llm.with_structured_output(schema)` so the
    pipeline stays backend-agnostic.
    """
    if _backend() == "openai":
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)
