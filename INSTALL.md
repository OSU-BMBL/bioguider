# Installing BioGuider

This guide walks through a from-scratch local install of the BioGuider Python
package. For what BioGuider does and how to run it, see
[README.md](README.md); for the full environment-variable reference, see
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

BioGuider is a library, not an application — there is no CLI entry point or web
server. You install it, configure a `.env`, and drive it programmatically
through the managers in `bioguider/managers/` (or via `pytest`). The companion
web UI lives in a separate repo (`bioguider-app`) and is not covered here.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.11** | `pyproject.toml` pins `^3.11`. 3.12+ is untested. |
| [Poetry](https://python-poetry.org/) | 1.8+ or 2.x | Source-of-truth installer. |
| Git | any recent | BioGuider clones target repos at runtime. |
| `libmagic` | system lib | Required by `python-magic` (file-type detection). See step 5. |
| A C toolchain | — | `faiss-cpu` ships wheels for most platforms; a compiler is the fallback. |

**libmagic** is the one non-Python dependency that trips people up:

```bash
# Debian / Ubuntu
sudo apt-get install libmagic1
# RHEL / Rocky / CentOS
sudo dnf install file-libs
# macOS
brew install libmagic
```

---

## 2. Get the code

```bash
git clone <your-fork-or-origin-url> bioguider
cd bioguider
```

All subsequent commands are run **from the repo root**. This matters — some
tests hard-code relative paths (see step 6).

---

## 3. Install dependencies

### Recommended: Poetry

```bash
poetry install          # creates the venv, installs runtime + dev deps
```

This reads `pyproject.toml` / `poetry.lock` and installs everything, including
the dev group (`pytest`, `bump2version`). Activate the environment with
`poetry shell`, or prefix commands with `poetry run`.

### Alternative: pip + requirements.txt

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is provided for non-Poetry setups, but **`pyproject.toml` is
the source of truth** — if the two ever disagree, trust Poetry.

---

## 4. Create the `logs/` directory

```bash
mkdir -p logs
```

`tests/conftest.py` opens `./logs/test.log` **unconditionally** at import time,
so `pytest` fails immediately if `logs/` does not exist. Create it once, up
front.

---

## 5. Configure `.env`

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

There are two supported LLM paths. Pick one.

### Path A — LiteLLM proxy (recommended)

Set `OPENAI_BASE_URL` to your proxy and provide a key. When `OPENAI_BASE_URL`
is set, the Azure block is **ignored entirely**:

```dotenv
OPENAI_BASE_URL=https://<your-proxy>/ai/v1
OPENAI_API_KEY=sk-<your-virtual-key>
OPENAI_MODEL=gpt-5.4
OPENAI_MAX_INPUT_TOKENS=128000
OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
```

### Path B — Azure OpenAI (legacy)

Leave `OPENAI_BASE_URL` **unset** and provide the Azure block:

```dotenv
OPENAI_API_TYPE=azure
OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
OPENAI_DEPLOYMENT_NAME=<chat-deployment>
OPENAI_MODEL=<model-name>
OPENAI_API_VERSION=2024-08-01-preview
OPENAI_MAX_INPUT_TOKENS=128000
OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME=<embedding-deployment>
```

Optional alternative providers (used via `LLM_PROVIDER` / a `provider=` argument
to `get_configured_llm`): `CLAUDE_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`,
plus the `KIMI_*`, `MINIMAX_*`, and `GPT_OSS_*` groups. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md) for every variable.

> **Embeddings are always Azure.** `OPENAI_BASE_URL` routes chat completions
> only — the RAG embedding path uses the Azure/OpenAI embedding deployment
> regardless. You must supply a working `OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME`
> even on the proxy path.

> ⚠️ **Never `git add` your `.env`.** It holds live API keys and is gitignored.
> If this repo was ever shared, rotate the keys.

---

## 6. Verify the install

Run the fast unit tests from the repo root:

```bash
poetry run pytest tests/          # or: pytest tests/  (inside poetry shell)
```

These do **not** call live LLMs and should pass offline. If they collect and
run, your install is sound.

> Do **not** start with `system_tests/`. Those are real integration tests: they
> clone public repos (Seurat, scanpy, …), call live LLMs, and cost money. Run
> them one file at a time and only when you mean to.

Lint (optional):

```bash
poetry run ruff check bioguider/
```

---

## 7. Two gotchas to know before your first run

These aren't install failures — they're the first two runtime errors people hit:

1. **Build settings via the manager, not the class.** Use
   `SettingsManager.initialize_with_params(...)`, **not** `Setting()` directly.
   `ProjectSettings.target_repo` defaults to an empty string that fails
   `DirectoryPath` validation the moment you construct settings by hand.

2. **Embeddings must be 256-dimensional.** RAG FAISS vectors are hard-coded to
   **256 dimensions** in `bioguider/rag/rag.py`. Your embedding deployment must
   produce 256-dim vectors, or queries fail at `FAISSRetriever` init. (Models
   like `text-embedding-3-small` support a `dimensions` parameter — configure
   the deployment accordingly.)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ./logs/test.log` on `pytest` | `logs/` missing | `mkdir -p logs` (step 4) |
| `ImportError: failed to find libmagic` | `python-magic` can't find the C lib | Install `libmagic` (step 1) |
| `DirectoryPath` validation error at startup | `Setting()` constructed directly | Use `SettingsManager.initialize_with_params(...)` |
| Retriever init fails on dimension mismatch | Embedding deployment ≠ 256-dim | Reconfigure embedding deployment to 256 dims |
| Tests fail only for `root_path` fixtures | Those tests hard-code a server path | Expected off that host — skip them |
| Wheel build errors for `faiss-cpu` | No prebuilt wheel for your platform | Install a C toolchain, or use a supported Python/OS |

---

## Next steps

- **[README.md](README.md)** — what BioGuider does and three runnable usage examples.
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — every environment variable.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the component map and extension points.
