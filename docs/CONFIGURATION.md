# Configuration reference

BioGuider reads all runtime configuration from a `.env` file at the repo root
(loaded via `python-dotenv` / `pydantic-settings`). Copy `.env.example` to `.env`
and fill in the values for the path you intend to use.

```bash
cp .env.example .env
# then edit .env
```

> ⚠️ `.env` holds live API keys and is gitignored. **Never `git add` it.** If this
> repo has ever been shared, rotate the keys.

There are two chat-completion paths and one embedding path. They are independent —
in particular, **RAG embeddings always go to Azure regardless of the chat path**.

---

## Chat completion — pick one path

### Path A: LiteLLM proxy (recommended)

When `OPENAI_BASE_URL` is set, chat requests are routed through the LiteLLM proxy and
the Azure variables below are ignored entirely.

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_BASE_URL` | Proxy endpoint. Its presence selects this path. | `https://bmblx.bmi.osumc.edu/ai/v1` |
| `OPENAI_API_KEY` | Virtual key for the proxy. | `sk-<bioguider-virtual-key>` |
| `OPENAI_MODEL` | Model name to request from the proxy. | `gpt-5.4` |

### Path B: Azure OpenAI (legacy)

Used only when `OPENAI_BASE_URL` is **unset**. This is the path referenced as the
"default Azure OpenAI path" in the project docs.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_TYPE` | Set to `azure`. |
| `OPENAI_API_KEY` | Azure OpenAI key. |
| `AZURE_OPENAI_ENDPOINT` | `https://<your-resource>.openai.azure.com` |
| `OPENAI_DEPLOYMENT_NAME` | Your chat deployment name. |
| `OPENAI_MODEL` | Model name. |
| `OPENAI_API_VERSION` | API version, e.g. `2024-08-01-preview`. |

---

## Embeddings (RAG) — always Azure

RAG indexing does **not** honor `OPENAI_BASE_URL`; it always calls Azure embeddings.

| Variable | Purpose |
|----------|---------|
| `OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME` | Azure embedding deployment, e.g. `text-embedding-3-small`. |

> **256-dim constraint.** FAISS vectors are hard-coded to **256 dimensions** in
> `bioguider/rag/rag.py`. The embedding deployment must be configured to emit 256-dim
> vectors, or `FAISSRetriever` init fails. If you use a model with a different native
> dimension, configure the deployment's output dimensionality to 256.

---

## Token limits

| Variable | Purpose |
|----------|---------|
| `OPENAI_MAX_INPUT_TOKENS` | Max prompt tokens; drives truncation. e.g. `128000`. |
| `OPENAI_MAX_OUTPUT_TOKENS` | Max completion tokens. |

---

## Alternative providers

The chat LLM handle is built by `get_configured_llm(provider=None)`
(`bioguider/agents/agent_utils.py`). When `provider` is `None` it reads `LLM_PROVIDER`
(default `azure`). Supported values: `azure`, `kimi`, `minimax`, `gpt-oss`.

| Variable | Selects / configures |
|----------|----------------------|
| `LLM_PROVIDER` | Which provider `get_configured_llm()` uses when no explicit `provider=` is passed. |
| `KIMI_API_KEY`, `KIMI_MODEL`, `KIMI_BASE_URL`, `KIMI_MAX_OUTPUT_TOKENS` | Kimi (OpenAI-shaped endpoint). |
| `MINIMAX_API_KEY`, `MINIMAX_MODEL`, `MINIMAX_BASE_URL`, `MINIMAX_MAX_OUTPUT_TOKENS` | MiniMax (OpenAI-shaped endpoint). |
| `GPT_OSS_API_KEY`, `GPT_OSS_MODEL`, `GPT_OSS_BASE_URL`, `GPT_OSS_MAX_OUTPUT_TOKENS` | gpt-oss (OpenAI-shaped endpoint). |

The Kimi / MiniMax / gpt-oss endpoints are OpenAI-shaped, so their `*_BASE_URL` is
passed through and the model is routed through `ChatOpenAI`.

Additional keys accepted for LangChain adapter paths (commented in `.env.example`):

| Variable | Provider |
|----------|----------|
| `CLAUDE_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |

> `DeepSeekConversation.chat` (`bioguider/conversation.py`) swallows exceptions and
> returns them stringified — don't assume a successful return type on that path.

---

## Paths and debugging

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_FOLDER` | Base directory for clones, FAISS indices, and sqlite caches. | `data/` |
| `BIOGUIDER_DEBUG` | Enable ad-hoc debug dumps. | off |
| `BIOGUIDER_DEBUG_DIR` | Where debug dumps are written. | `bioguider_debug/` |

---

## Settings construction (important)

Do **not** instantiate `Setting()` directly — `ProjectSettings.target_repo` defaults to
an empty string that fails `DirectoryPath` validation at runtime. Always go through:

```python
from bioguider.settings import SettingsManager

SettingsManager.initialize_with_params(
    target_repo=Path("data/.adalflow/repos/<author>_<repo>"),
    markdown_docs_name="...",
    hierarchy_name="...",
    ignore_list=[...],
    language="python",
    max_thread_count=4,
    model="gpt-5.4",
    temperature=0.0,
    request_timeout=600,
    openai_base_url="https://bmblx.bmi.osumc.edu/ai/v1",
)
```
