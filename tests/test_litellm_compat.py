"""
LiteLLM proxy compatibility tests.

Skipped entirely when OPENAI_BASE_URL is unset (CI safety).
Run with:
    OPENAI_BASE_URL=https://... OPENAI_API_KEY=sk-... pytest tests/test_litellm_compat.py -v
"""
import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODELS = ["gpt-5.4", "kimi-k2.5", "glm-5", "gpt-oss-120b", "gpt-4o"]

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_BASE_URL"),
    reason="OPENAI_BASE_URL not set — skipping live LiteLLM compat tests",
)


@pytest.mark.parametrize("model", MODELS)
def test_model_round_trip(model):
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    response = llm.invoke("Say 'ok'")
    assert response.content, f"Empty response from {model}"

    # AC8: non-zero token usage guards silent zero-accounting on kimi/glm
    usage = response.response_metadata.get("token_usage", {})
    total = usage.get("total_tokens", 0)
    assert total > 0, (
        f"Zero total_tokens reported for {model} — "
        "possible silent accounting failure under OpenAICallbackHandler"
    )
