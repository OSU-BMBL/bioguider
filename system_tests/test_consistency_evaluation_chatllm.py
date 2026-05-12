"""System test for ConsistencyEvaluationTask against a small synthetic repo.

The documentation below shows a code snippet that uses ``ChatLLM`` in a way that
is *inconsistent* with the actual source code shipped in the (synthetic) repo:

  * the doc does ``import ChatLLM`` then ``ChatLLM(key=key)`` — but ``ChatLLM`` is
    a class in ``chatllm.py``, not a module, and its constructor parameter is
    ``api_key`` not ``key``;
  * the doc calls ``c.chat('balahbalah')`` — the method exists but the parameter
    is named ``message``;
  * the trailing "supported LLM include GPT-5, Sonnet-4.6, ..." sentence is pure
    narrative and is not checkable against code.

The test is a smoke test (it calls a real LLM): it asserts a well-formed result
comes back and logs the assessment / development items so a human can eyeball
whether the mismatches were caught.
"""

import logging
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from bioguider.agents.consistency_evaluation_task import (
    ConsistencyEvaluationResult,
    ConsistencyEvaluationTask,
)
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.code_structure_builder import CodeStructureBuilder

logger = logging.getLogger(__name__)


DOCUMENTATION = textwrap.dedent("""\
    Here is how to use ChatLLM to communicate with open-source LLM:

    ```python
    import os
    key = os.environ.get('KEY', '')
    base_url = os.environ.get('BASE_URL', '')

    def test_chat():
        import ChatLLM
        c = ChatLLM(key=key, base_url=base_url)
        res = c.chat('balahbalah')
        print(res.content)
    ```

    The supported LLM include GPT-5, Sonnet-4.6, Gemini-2.5-Pro.
    """)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "chatllm_repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "chatllm.py").write_text(textwrap.dedent('''
        """A thin client for talking to hosted LLMs."""


        class ChatResponse:
            """The reply returned by :meth:`ChatLLM.chat`."""

            def __init__(self, content: str):
                self.content = content


        class ChatLLM:
            """Client for chatting with a hosted LLM."""

            def __init__(self, api_key: str, base_url: str = ""):
                self.api_key = api_key
                self.base_url = base_url

            def chat(self, message: str) -> "ChatResponse":
                """Send ``message`` to the model and return the reply."""
                return ChatResponse(content="...")
    '''), encoding="utf-8")
    return repo


@pytest.fixture(scope="module")
def chatllm_db(data_folder, tmp_path_factory):
    repo = _make_repo(tmp_path_factory.mktemp("chatllm"))
    db = CodeStructureDb("chatllm_test", "chatllm_test", data_folder)
    builder = CodeStructureBuilder(
        repo_path=repo,
        gitignore_path=Path(repo, ".gitignore"),
        code_structure_db=db,
    )
    builder.build_code_structure()
    # sanity: the synthetic repo's symbols made it into the DB
    names = set(db.select_all_names())
    assert {"ChatLLM", "ChatResponse", "chat"} <= names, names
    yield db
    db_dir = os.path.join(data_folder, "databases")
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except OSError as e:  # pragma: no cover - best effort cleanup
            logger.warning("Could not clean up %s: %s", db_dir, e)

@pytest.mark.skip()
def test_consistency_evaluation_chatllm(llm, step_callback, chatllm_db):
    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=chatllm_db,
        step_callback=step_callback,
    )
    result = task.evaluate(
        domain="user guide/API documentation",
        documentation=DOCUMENTATION,
    )

    assert isinstance(result, ConsistencyEvaluationResult)
    assert 0 <= result.score <= 100
    assert isinstance(result.assessment, str) and result.assessment.strip()
    assert isinstance(result.development, list)
    assert isinstance(result.strengths, list)

    logger.info("ConsistencyEvaluationTask score=%s", result.score)
    logger.info("assessment=%s", result.assessment)
    logger.info("development=%s", result.development)
    logger.info("strengths=%s", result.strengths)


# ---------------------------------------------------------------------------
# Nested-function case: a function defined inside another function
# ---------------------------------------------------------------------------

NESTED_DOCUMENTATION = textwrap.dedent("""\
    To process a batch of raw measurements, call ``process_batch(values)``. It
    relies on a small helper, ``normalize(x)``, that is defined **inside**
    ``process_batch`` and scales each value into the ``[0, 1]`` range:

    ```python
    def process_batch(values):
        def normalize(x):
            return x / 100.0
        return [normalize(v) for v in values]
    ```

    Note that ``normalize`` is not importable on its own — it only exists within
    ``process_batch``.
    """)


def _make_nested_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "nested_repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "pipeline.py").write_text(textwrap.dedent('''
        """Tiny data-processing pipeline used to exercise nested-function handling."""


        def process_batch(values):
            """Normalize every value in ``values`` into the ``[0, 1]`` range."""

            def normalize(x):
                """Scale a single raw measurement into ``[0, 1]``."""
                return x / 100.0

            return [normalize(v) for v in values]
    '''), encoding="utf-8")
    return repo


@pytest.fixture(scope="module")
def nested_db(data_folder, tmp_path_factory):
    repo = _make_nested_repo(tmp_path_factory.mktemp("nested"))
    db = CodeStructureDb("nested_test", "nested_test", data_folder)
    builder = CodeStructureBuilder(
        repo_path=repo,
        gitignore_path=Path(repo, ".gitignore"),
        code_structure_db=db,
    )
    builder.build_code_structure()
    # sanity: the nested helper is recorded with its enclosing function as parent
    names = set(db.select_all_names())
    assert {"process_batch", "normalize"} <= names, names
    normalize_rows = db.select_by_name_like("normalize")
    assert any(r["name"] == "normalize" and r["parent"] == "process_batch"
               for r in normalize_rows), normalize_rows
    yield db
    db_dir = os.path.join(data_folder, "databases")
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except OSError as e:  # pragma: no cover - best effort cleanup
            logger.warning("Could not clean up %s: %s", db_dir, e)


def test_consistency_evaluation_nested_function(llm, step_callback, nested_db):
    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=nested_db,
        step_callback=step_callback,
    )
    result = task.evaluate(
        domain="user guide/API documentation",
        documentation=NESTED_DOCUMENTATION,
    )

    assert isinstance(result, ConsistencyEvaluationResult)
    assert 0 <= result.score <= 100
    assert isinstance(result.assessment, str) and result.assessment.strip()
    assert isinstance(result.development, list)
    assert isinstance(result.strengths, list)

    logger.info("nested ConsistencyEvaluationTask score=%s", result.score)
    logger.info("assessment=%s", result.assessment)
    logger.info("development=%s", result.development)
    logger.info("strengths=%s", result.strengths)
