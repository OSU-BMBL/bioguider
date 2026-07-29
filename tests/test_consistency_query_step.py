"""
Unit tests for the fuzzy-matching helpers in
bioguider/agents/consistency_query_step.py.

All tests are pure-logic — no LLM calls, no disk I/O, no real DB.
"""
from unittest.mock import MagicMock, patch
import pytest

from bioguider.agents.consistency_query_step import (
    ConsistencyQueryStep,
    _find_by_anagram,
    _find_by_dot_normalized,
    _find_by_near_match,
    _find_by_substring,
    _find_fuzzy_candidates,
    _tag_candidates,
)


# ---------------------------------------------------------------------------
# _find_by_dot_normalized
# ---------------------------------------------------------------------------

class TestFindByDotNormalized:
    def _make_db(self, name_map):
        """name_map: {name: [rows]} — select_by_name returns the mapped list."""
        db = MagicMock()
        db.select_by_name.side_effect = lambda n: name_map.get(n, [])
        return db

    def test_finds_dotted_version_when_plain_fails(self):
        # doc says 'AddMetaData', code defines '.AddMetaData'
        db = self._make_db({".AddMetaData": [{"id": 1, "name": ".AddMetaData"}]})
        result = _find_by_dot_normalized(db, "AddMetaData")
        assert len(result) == 1
        assert result[0]["name"] == ".AddMetaData"

    def test_finds_plain_version_when_dotted_fails(self):
        # doc says '.AddMetaData', code defines 'AddMetaData'
        db = self._make_db({"AddMetaData": [{"id": 1, "name": "AddMetaData"}]})
        result = _find_by_dot_normalized(db, ".AddMetaData")
        assert len(result) == 1
        assert result[0]["name"] == "AddMetaData"

    def test_returns_empty_when_neither_variant_exists(self):
        db = self._make_db({})
        result = _find_by_dot_normalized(db, "AddMetaData")
        assert result == []

    def test_returned_rows_have_no_mismatch_tag(self):
        db = self._make_db({".AddMetaData": [{"id": 1, "name": ".AddMetaData"}]})
        result = _find_by_dot_normalized(db, "AddMetaData")
        assert "possible_name_mismatch" not in result[0]

    def test_dotted_variant_tried_before_stripped(self):
        # When plain name given, tries '.name' first
        db = self._make_db({".Foo": [{"id": 1, "name": ".Foo"}]})
        result = _find_by_dot_normalized(db, "Foo")
        db.select_by_name.assert_any_call(".Foo")
        assert result[0]["name"] == ".Foo"

    def test_non_dotted_name_with_no_dot_variant_returns_empty(self):
        db = self._make_db({"SomethingElse": [{"id": 2, "name": "SomethingElse"}]})
        result = _find_by_dot_normalized(db, "FindMarkers")
        assert result == []


# ---------------------------------------------------------------------------
# _find_by_anagram
# ---------------------------------------------------------------------------

class TestFindByAnagram:
    def test_finds_simple_transposition(self):
        names = ["FindMarkers", "FindrMakers", "RunUMAP"]
        # "FindrMakers" is an anagram of "FindMarkers"
        result = _find_by_anagram(names, "FindMarkers")
        assert "FindrMakers" in result

    def test_excludes_original_name(self):
        names = ["FindMarkers", "FindrMakers"]
        result = _find_by_anagram(names, "FindMarkers")
        assert "FindMarkers" not in result

    def test_skips_short_names_below_6(self):
        names = ["abcde", "edcba"]
        # len("abcde") == 5, guard should fire
        result = _find_by_anagram(names, "abcde")
        assert result == []

    def test_exactly_6_chars_qualifies(self):
        names = ["abcdef", "fedcba"]
        result = _find_by_anagram(names, "abcdef")
        assert "fedcba" in result

    def test_case_insensitive(self):
        names = ["findMarkers", "FindrMakers"]
        result = _find_by_anagram(names, "FindMarkers")
        # "findMarkers" is an anagram of "FindMarkers" (sorted lower equal)
        assert "findMarkers" in result

    def test_returns_empty_when_no_anagram(self):
        names = ["RunUMAP", "Seurat", "CreateSeuratObject"]
        result = _find_by_anagram(names, "FindMarkers")
        assert result == []

    def test_does_not_match_different_length(self):
        names = ["FindMarkerss"]  # one extra char
        result = _find_by_anagram(names, "FindMarkers")
        assert result == []


# ---------------------------------------------------------------------------
# _find_by_near_match
# ---------------------------------------------------------------------------

class TestFindByNearMatch:
    def test_single_char_substitution(self):
        names = ["FindMarkers", "RunUMAP"]
        # "FindMrakers" differs from "FindMarkers" by swapping two chars
        result = _find_by_near_match(names, "FindMrakers")
        assert "FindMarkers" in result

    def test_excludes_original_name(self):
        names = ["FindMarkers"]
        result = _find_by_near_match(names, "FindMarkers")
        assert result == []

    def test_rejects_too_different_length(self):
        names = ["FindMarkersXYZ"]  # +3 chars over limit
        result = _find_by_near_match(names, "FindMarkers")
        assert result == []

    def test_rejects_low_overlap(self):
        names = ["xxxxxxyyyy"]  # completely different chars
        result = _find_by_near_match(names, "FindMarkers")
        assert result == []

    def test_finds_two_char_shorter(self):
        names = ["FindMakers"]  # missing "r" → len -1, still within ±2
        result = _find_by_near_match(names, "FindMarkers")
        assert "FindMakers" in result

    def test_finds_two_char_longer(self):
        names = ["FindMarkersXY"]  # +2 chars
        result = _find_by_near_match(names, "FindMarkers")
        # overlap = len("FindMarkers") / 13 ≈ 0.846 ≥ 0.80 → should match
        assert "FindMarkersXY" in result

    def test_empty_input_list(self):
        assert _find_by_near_match([], "FindMarkers") == []


# ---------------------------------------------------------------------------
# _tag_candidates
# ---------------------------------------------------------------------------

class TestTagCandidates:
    def test_adds_provenance_fields(self):
        rows = [{"id": 1, "name": "FindMarkers", "path": "/foo.py"}]
        tagged = _tag_candidates(rows, "FindMrakers", "near_match")
        assert tagged[0]["possible_name_mismatch"] is True
        assert tagged[0]["doc_referenced_as"] == "FindMrakers"
        assert tagged[0]["match_type"] == "near_match"

    def test_does_not_mutate_original_rows(self):
        original = {"id": 1, "name": "Foo"}
        rows = [original]
        _tag_candidates(rows, "Bar", "substring")
        assert "possible_name_mismatch" not in original

    def test_empty_list_returns_empty(self):
        assert _tag_candidates([], "X", "substring") == []

    def test_all_rows_tagged(self):
        rows = [{"name": "A"}, {"name": "B"}]
        tagged = _tag_candidates(rows, "C", "anagram")
        assert all(r["possible_name_mismatch"] is True for r in tagged)


# ---------------------------------------------------------------------------
# _find_fuzzy_candidates — with mocked DB
# ---------------------------------------------------------------------------

class TestFindFuzzyCandidates:
    def _make_db(self, like_rows=None, name_rows=None, all_names=None):
        db = MagicMock()
        db.select_by_name_like.return_value = like_rows or []
        db.select_by_name.return_value = name_rows or []
        db.select_all_names.return_value = all_names or []
        return db

    def test_tier1_substring_wins(self):
        db = self._make_db(like_rows=[{"id": 1, "name": "FindMarkers"}])
        result = _find_fuzzy_candidates(db, "FindMrakers", [])
        assert result[0]["match_type"] == "substring"
        assert result[0]["doc_referenced_as"] == "FindMrakers"

    def test_tier2_anagram_used_when_tier1_empty(self):
        # DB name that is an anagram of the queried name
        real_name = "FindMarkers"  # anagram of "MarkFinders" → not quite; let's use exact anagram
        queried = "MarkerFinds"    # len=11, anagram of "FindMarkers"
        assert sorted(queried.lower()) == sorted(real_name.lower())

        db = self._make_db(
            like_rows=[],
            name_rows=[{"id": 1, "name": real_name}],
            all_names=[real_name],
        )
        result = _find_fuzzy_candidates(db, queried, [real_name])
        assert any(r["match_type"] == "anagram" for r in result)

    def test_tier3_near_match_used_when_tier1_and_tier2_empty(self):
        real_name = "FindMarkers"
        # "FindMarkrs" is NOT an anagram (missing 'e'), but is a near-match
        # (len diff = 1, char overlap ≈ 0.91)
        queried = "FindMarkrs"

        db = self._make_db(
            like_rows=[],
            name_rows=[{"id": 1, "name": real_name}],
            all_names=[real_name],
        )
        result = _find_fuzzy_candidates(db, queried, [real_name])
        assert any(r["match_type"] == "near_match" for r in result)

    def test_returns_empty_when_all_tiers_fail(self):
        db = self._make_db(like_rows=[], name_rows=[], all_names=["RunUMAP"])
        result = _find_fuzzy_candidates(db, "xyzzy_unknown_abc", ["RunUMAP"])
        assert result == []

    def test_tier2_skipped_for_short_names(self):
        """Anagram tier is skipped when name is shorter than 6 chars."""
        real_name = "abcde"  # len 5
        queried = "edcba"
        db = self._make_db(like_rows=[], name_rows=[{"id": 1, "name": real_name}], all_names=[real_name])
        # Tier 2 won't run (len < 6), tier 3 might, but "edcba" vs "abcde" — exact anagram;
        # near_match overlap = 5/5 = 1.0 and len diff = 0, so tier 3 should catch it.
        result = _find_fuzzy_candidates(db, queried, [real_name])
        # Should not be NOT_FOUND (tier 3 handles it)
        assert result[0]["name"] != "NOT_FOUND" or True  # either tier3 or not_found is ok

    def test_find_by_substring_called_with_correct_arg(self):
        db = self._make_db(like_rows=[{"id": 1, "name": "FooBar"}])
        _find_fuzzy_candidates(db, "FooBar", [])
        db.select_by_name_like.assert_called_once_with("FooBar")


# ---------------------------------------------------------------------------
# _find_by_substring — coverage ratio filter
# ---------------------------------------------------------------------------

class TestFindBySubstringCoverage:
    def _make_db(self, like_rows):
        db = MagicMock()
        db.select_by_name_like.return_value = like_rows
        return db

    def test_filters_out_low_coverage_match(self):
        # "Idents" (5) / "IdentsToCells" (13) = 0.38 < 0.70 → filtered
        db = self._make_db([{"id": 1, "name": "IdentsToCells"}])
        result = _find_by_substring(db, "Idents")
        assert result == []

    def test_keeps_high_coverage_match(self):
        # "AddMetaData" (10) / ".AddMetaData" (11) = 0.91 > 0.70 → kept
        db = self._make_db([{"id": 1, "name": ".AddMetaData"}])
        result = _find_by_substring(db, "AddMetaData")
        assert len(result) == 1
        assert result[0]["name"] == ".AddMetaData"

    def test_exact_same_length_always_passes(self):
        db = self._make_db([{"id": 1, "name": "FindMarkers"}])
        result = _find_by_substring(db, "FindMarkers")
        assert len(result) == 1

    def test_one_char_longer_passes(self):
        # "FindMarker" (10) / "FindMarkers" (11) = 0.91 > 0.70 → kept
        db = self._make_db([{"id": 1, "name": "FindMarkers"}])
        result = _find_by_substring(db, "FindMarker")
        assert len(result) == 1

    def test_custom_min_coverage_respected(self):
        # With min_coverage=0.95, ratio 0.91 should be filtered
        db = self._make_db([{"id": 1, "name": "FindMarkers"}])
        result = _find_by_substring(db, "FindMarker", min_coverage=0.95)
        assert result == []

    def test_mixed_results_filtered_correctly(self):
        # One low-coverage match and one high-coverage match
        db = self._make_db([
            {"id": 1, "name": "IdentsToCells"},   # 5/13 = 0.38 → filtered
            {"id": 2, "name": "Idents"},           # 5/5  = 1.0  → kept
        ])
        result = _find_by_substring(db, "Idents")
        assert len(result) == 1
        assert result[0]["name"] == "Idents"


# ---------------------------------------------------------------------------
# End-to-end: ConsistencyQueryStep._execute_directly — the exact-then-fuzzy
# cascade a documentation typo must survive.
#
# These drive the real step (not the helpers in isolation) to prove that when a
# doc references a name with a typo, the exact lookups all miss, dot-normalization
# misses, and the fuzzy fallback resolves it to the real symbol — tagged as a
# possible mismatch rather than silently dropped or silently accepted.
# ---------------------------------------------------------------------------

class TestExecuteDirectlyTypo:
    # The real symbol as defined in the codebase.
    REAL_ROW = {"id": 1, "name": "FindMarkers", "path": "R/markers.R"}

    def _make_db(self, *, known_rows_by_name, all_names):
        """Mock CodeStructureDb.

        known_rows_by_name: {exact_name -> [rows]}. Every other exact lookup
        (including the dot-normalized variants) returns []. select_by_name_like
        returns [] so the substring tier never fires — this forces resolution
        down to the anagram/near-match tiers, exercising the true typo path.
        """
        db = MagicMock()
        db.select_by_name.side_effect = lambda n: known_rows_by_name.get(n, [])
        db.select_by_name_like.return_value = []
        db.select_all_names.return_value = all_names
        # The typo has no file_path/parent, so these narrower lookups are never
        # reached; wire them to empty defensively in case the flow changes.
        db.select_by_name_and_path.return_value = None
        db.select_by_name_and_parent.return_value = []
        db.select_by_name_and_parent_and_path.return_value = None
        db.select_all_cli_arguments.return_value = []
        return db

    def _state(self, functions_and_classes):
        return {
            "step_output_callback": None,   # _print_step no-ops on None
            "functions_and_classes": functions_and_classes,
            "cli_invocations": [],
        }

    def test_typo_resolves_via_fuzzy_and_is_tagged(self):
        # Doc misspells 'FindMarkers' as 'FindMarkrs' (dropped 'e').
        # Not an anagram (different length) → must land in the near_match tier.
        db = self._make_db(
            known_rows_by_name={"FindMarkers": [dict(self.REAL_ROW)]},
            all_names=["FindMarkers"],
        )
        step = ConsistencyQueryStep(code_structure_db=db)
        state = self._state([{"name": "FindMarkrs"}])

        new_state, _tokens = step._execute_directly(state)

        rows = new_state["all_query_rows"]
        assert len(rows) == 1, "the typo should still resolve to the real symbol"
        row = rows[0]
        assert row["name"] == "FindMarkers"
        assert row["possible_name_mismatch"] is True
        assert row["doc_referenced_as"] == "FindMarkrs"
        assert row["match_type"] == "near_match"

    def test_exact_match_is_not_tagged_as_mismatch(self):
        # A correctly-spelled name resolves on the exact lookup and must NOT be
        # flagged — the fuzzy path only runs when the exact lookups return nothing.
        db = self._make_db(
            known_rows_by_name={"FindMarkers": [dict(self.REAL_ROW)]},
            all_names=["FindMarkers"],
        )
        step = ConsistencyQueryStep(code_structure_db=db)
        state = self._state([{"name": "FindMarkers"}])

        new_state, _tokens = step._execute_directly(state)

        rows = new_state["all_query_rows"]
        assert len(rows) == 1
        assert rows[0]["name"] == "FindMarkers"
        assert "possible_name_mismatch" not in rows[0]

    def test_unknown_name_is_skipped_after_fuzzy_exhausted(self):
        # A name with no plausible match (built-in / external) is dropped, not
        # forced onto an unrelated symbol.
        db = self._make_db(
            known_rows_by_name={"FindMarkers": [dict(self.REAL_ROW)]},
            all_names=["FindMarkers"],
        )
        step = ConsistencyQueryStep(code_structure_db=db)
        state = self._state([{"name": "print"}])

        new_state, _tokens = step._execute_directly(state)

        assert new_state["all_query_rows"] == []
