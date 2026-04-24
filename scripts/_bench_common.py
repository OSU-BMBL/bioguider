"""Shared helpers for the Phase 1-5 benchmark runner scripts.

These scripts consume the BMBLx LiteLLM proxy (``OPENAI_BASE_URL``) listed
in ``.env.example``. The model roster mirrors ``tests/test_litellm_compat.py``.

Usage sketch:

    from scripts._bench_common import make_llm, LITELLM_MODELS
    llm = make_llm("kimi-k2.5")

Everything in this module is pure plumbing — no benchmark logic lives here.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Horizontal benchmark roster (A/B/C/D/E as drawn on the whiteboard).
# Must match tests/test_litellm_compat.py so the compat suite guards the
# same models the benchmark actually uses.
LITELLM_MODELS = [
    "kimi-k2.5",       # A — default
    "glm-5",           # B
    "gpt-oss-120b",    # C
    "gpt-5.4",         # D
    "gpt-4o",          # E (reference closed-source)
]

DEFAULT_MODEL = LITELLM_MODELS[0]


def load_env() -> None:
    """Load .env and verify LiteLLM credentials are present."""
    load_dotenv()
    missing = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.stderr.write(
            "ERROR: missing env vars for LiteLLM proxy: "
            + ", ".join(missing)
            + ".\nSee .env.example for the expected keys.\n"
        )
        sys.exit(2)


def make_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    thinking: Optional[bool] = None,
    max_retries: int = 3,
) -> ChatOpenAI:
    """Construct a ChatOpenAI pointed at the BMBLx LiteLLM proxy.

    Args:
        model: One of LITELLM_MODELS (or any other name the proxy serves).
        temperature: 0.0 for benchmark determinism.
        thinking: When True and the model supports it (Kimi-k2.5), enable
            thinking/reasoning mode via the model-specific extra-body knob.
            Used for Phase 4 (thinking-vs-general).
        max_retries: LangChain-level retry count for transient 429s.
    """
    kwargs = dict(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=temperature,
        max_retries=max_retries,
    )
    if thinking is True:
        # LiteLLM forwards extra_body to the backend; Kimi recognises this flag.
        # For other models it's a no-op. Refine once Qin confirms the exact key.
        kwargs["extra_body"] = {"thinking": True}
    return ChatOpenAI(**kwargs)


def make_step_callback(tag: str = "bench") -> Callable[[str, str], None]:
    """Return a minimal stdout step callback for the managers."""

    def _cb(step_name: str, message: str) -> None:
        sys.stdout.write(f"[{tag}][{step_name}] {message}\n")
        sys.stdout.flush()

    return _cb
