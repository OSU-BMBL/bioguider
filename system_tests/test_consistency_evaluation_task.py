
import os
from pathlib import Path
import shutil
import pytest
import logging
import json

from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationTask
from bioguider.agents.consistency_query_step import (
    _find_by_anagram,
    _find_by_near_match,
    _find_by_substring,
    _find_fuzzy_candidates,
)
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.rag.rag import RAG
from bioguider.utils.code_structure_builder import CodeStructureBuilder
from bioguider.agents.consistency_collection_step import ConsistencyCollectionResult

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(data_folder):
    """Cleanup function that runs after all tests in this module complete."""
    yield
    db_path = os.path.join(data_folder, "databases")
    if os.path.exists(db_path):
        logger.info(f"Cleaning up database directory: {db_path}")
        try:
            shutil.rmtree(db_path)
            logger.info("Database directory cleaned up")
        except Exception as e:
            logger.warning(f"Warning: Could not clean up database directory: {e}")


# ---------------------------------------------------------------------------
# Shared fixture: build CodeStructureDb once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seurat_db(data_folder):
    """Clone the Seurat repo and build a CodeStructureDb from it."""
    repo_url = "https://github.com/satijalab/seurat"
    rag = RAG()
    rag.initialize_db_manager()
    rag.initialize_repo(repo_url_or_path=repo_url)

    code_structure_db = CodeStructureDb("test", "test", data_folder)
    code_structure_builder = CodeStructureBuilder(
        repo_path=rag.repo_dir,
        gitignore_path=Path(rag.repo_dir, ".gitignore"),
        code_structure_db=code_structure_db,
    )
    code_structure_builder.build_code_structure()
    return code_structure_db, rag.repo_dir


# ---------------------------------------------------------------------------
# Original smoke test (kept as-is)
# ---------------------------------------------------------------------------

def test_ConsistencyEvaluationTask(llm, step_callback, seurat_db):
    schema = ConsistencyCollectionResult.model_json_schema()
    logger.info(json.dumps(schema, indent=2))

    code_structure_db, repo_dir = seurat_db

    with open(Path(repo_dir, "vignettes/de_vignette.Rmd"), "r") as f:
        user_guide_api_documentation = f.read()

    task = ConsistencyEvaluationTask(
        llm=llm,
        code_structure_db=code_structure_db,
        step_callback=step_callback,
    )
    state = task.evaluate(
        domain="user guide/API documentation",
        documentation=user_guide_api_documentation,
    )
    assert state is not None


# ---------------------------------------------------------------------------
# Fuzzy matching — DB method tests (no LLM)
# ---------------------------------------------------------------------------

class TestFuzzyMatchingDbMethods:
    """Verify that the new CodeStructureDb methods return sensible data."""

    def test_select_all_names_returns_nonempty_list(self, seurat_db):
        db, _ = seurat_db
        names = db.select_all_names()
        assert isinstance(names, list)
        assert len(names) > 0, "Expected at least one name in the Seurat db"

    def test_select_all_names_are_strings(self, seurat_db):
        db, _ = seurat_db
        names = db.select_all_names()
        assert all(isinstance(n, str) for n in names)

    def test_select_by_name_like_finds_partial_name(self, seurat_db):
        db, _ = seurat_db
        names = db.select_all_names()
        if not names:
            pytest.skip("Empty database")
        # Take first name with len >= 4 and search for its first 4 chars
        long_names = [n for n in names if len(n) >= 4]
        if not long_names:
            pytest.skip("No names long enough")
        target = long_names[0]
        partial = target[:4]
        rows = db.select_by_name_like(partial)
        assert len(rows) > 0
        assert all(partial.lower() in r["name"].lower() for r in rows)

    def test_select_by_name_like_returns_dicts_with_expected_keys(self, seurat_db):
        db, _ = seurat_db
        names = db.select_all_names()
        if not names:
            pytest.skip("Empty database")
        rows = db.select_by_name_like(names[0][:3])
        if not rows:
            pytest.skip("LIKE returned no rows")
        expected_keys = {"id", "name", "path", "doc_string", "params"}
        assert expected_keys.issubset(rows[0].keys())


# ---------------------------------------------------------------------------
# Fuzzy matching — pure function tests against real DB names
# ---------------------------------------------------------------------------

class TestFuzzyFunctionsWithRealNames:
    """Use real names from the DB to test each fuzzy tier in isolation."""

    def _get_long_names(self, db, min_len=8):
        return [n for n in db.select_all_names() if len(n) >= min_len]

    def test_find_by_anagram_finds_transposed_name(self, seurat_db):
        db, _ = seurat_db
        long_names = self._get_long_names(db)
        if not long_names:
            pytest.skip("No long names in db")
        target = long_names[0]
        # Swap first and last character to form an anagram
        if target[0] == target[-1]:
            pytest.skip("First and last chars identical; anagram == original")
        anagram = target[-1] + target[1:-1] + target[0]
        all_names = db.select_all_names()
        found = _find_by_anagram(all_names, anagram)
        assert target in found, (
            f"Expected '{target}' in anagram results for '{anagram}', got {found}"
        )

    def test_find_by_near_match_finds_one_char_typo(self, seurat_db):
        db, _ = seurat_db
        long_names = self._get_long_names(db, min_len=6)
        if not long_names:
            pytest.skip("No long names in db")
        target = long_names[0]
        # Replace the last character with a clearly foreign letter
        replacement = "z" if target[-1] != "z" else "q"
        typo = target[:-1] + replacement
        if typo == target:
            pytest.skip("Substitution produced no change")
        all_names = db.select_all_names()
        found = _find_by_near_match(all_names, typo)
        assert target in found, (
            f"Expected '{target}' in near-match results for '{typo}', got {found}"
        )

    def test_find_by_substring_finds_partial_name(self, seurat_db):
        db, _ = seurat_db
        # Use min_len=10 so stripping first+last char gives coverage (n-2)/n >= 0.80,
        # safely above the 0.70 filter threshold.
        long_names = self._get_long_names(db, min_len=10)
        if not long_names:
            pytest.skip("No names with length >= 10 in db")
        target = long_names[0]
        partial = target[1:-1]  # strip first and last char
        rows = _find_by_substring(db, partial)
        matched_names = [r["name"] for r in rows]
        assert target in matched_names, (
            f"Expected '{target}' in LIKE results for '{partial}'"
        )

    def test_find_fuzzy_candidates_returns_empty_for_gibberish(self, seurat_db):
        db, _ = seurat_db
        all_names = db.select_all_names()
        result = _find_fuzzy_candidates(db, "xyzzy_completely_unknown_zqwx", all_names)
        assert result == [], "Unresolvable name should be skipped, not returned as a sentinel"

    def test_find_fuzzy_candidates_resolves_near_match_typo(self, seurat_db):
        db, _ = seurat_db
        long_names = self._get_long_names(db, min_len=8)
        if not long_names:
            pytest.skip("No long names in db")
        target = long_names[0]
        replacement = "z" if target[-1] != "z" else "q"
        typo = target[:-1] + replacement
        if typo == target:
            pytest.skip("Substitution produced no change")
        all_names = db.select_all_names()
        results = _find_fuzzy_candidates(db, typo, all_names)
        matched_db_names = [r["name"] for r in results]
        assert target in matched_db_names, (
            f"Expected fuzzy candidates to include '{target}' for typo '{typo}'"
        )
        # All returned rows must be tagged
        assert all(r.get("possible_name_mismatch") is True for r in results)
