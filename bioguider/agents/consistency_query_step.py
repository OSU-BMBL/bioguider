
from collections import Counter
from typing import List, Dict, Any, Optional

from bioguider.agents.common_step import CommonStep
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE


# ---------------------------------------------------------------------------
# Fuzzy-matching helpers — encapsulated so they can be disabled independently
# ---------------------------------------------------------------------------

def _find_by_dot_normalized(db: CodeStructureDb, name: str) -> List[Dict[str, Any]]:
    """Pre-fuzzy step: handle leading-dot naming conventions (e.g. R internals).

    In R, '.Foo' is the private implementation of the public 'Foo'.
    When the exact lookup for 'AddMetaData' fails, this tries '.AddMetaData',
    and vice-versa.  Returned rows carry NO mismatch tag — a dot difference
    is a naming convention, not a documentation error.
    """
    # Case 1: doc references 'Foo', code defines '.Foo'
    dotted = "." + name
    rows = db.select_by_name(dotted)
    if rows:
        return rows

    # Case 2: doc references '.Foo', code defines 'Foo'
    if name.startswith("."):
        stripped = name[1:]
        rows = db.select_by_name(stripped)
        if rows:
            return rows

    return []


def _find_by_substring(
    db: CodeStructureDb,
    name: str,
    min_coverage: float = 0.70,
) -> List[Dict[str, Any]]:
    """Tier-1: SQL LIKE substring search filtered by coverage ratio.

    Only returns a match when len(queried) / len(matched) >= min_coverage,
    preventing short names (e.g. 'Idents', 5 chars) from spuriously matching
    much longer names (e.g. 'IdentsToCells', 13 chars → ratio 0.38).
    """
    rows = db.select_by_name_like(name)
    return [
        r for r in rows
        if len(r["name"]) > 0 and len(name) / len(r["name"]) >= min_coverage
    ]


def _find_by_anagram(all_names: List[str], name: str) -> List[str]:
    """Tier-2: return DB names that are anagrams of *name* (len >= 6 guard)."""
    if len(name) < 6:
        return []
    name_sorted = sorted(name.lower())
    return [
        n for n in all_names
        if n != name and len(n) >= 6 and sorted(n.lower()) == name_sorted
    ]


def _find_by_near_match(all_names: List[str], name: str) -> List[str]:
    """Tier-3: return DB names within ±2 chars of *name* with ≥80 % char overlap."""
    name_lower = name.lower()
    name_counter = Counter(name_lower)
    results = []
    for candidate in all_names:
        if candidate == name:
            continue
        if abs(len(candidate) - len(name)) > 2:
            continue
        cand_counter = Counter(candidate.lower())
        overlap = sum((name_counter & cand_counter).values())
        max_len = max(len(name), len(candidate))
        if max_len > 0 and overlap / max_len >= 0.80:
            results.append(candidate)
    return results


def _tag_candidates(
    rows: List[Dict[str, Any]],
    doc_name: str,
    match_type: str,
) -> List[Dict[str, Any]]:
    """Attach fuzzy-match provenance fields to each candidate row."""
    tagged = []
    for row in rows:
        r = dict(row)
        r["possible_name_mismatch"] = True
        r["doc_referenced_as"] = doc_name
        r["match_type"] = match_type
        tagged.append(r)
    return tagged


def _find_fuzzy_candidates(
    db: CodeStructureDb,
    name: str,
    all_db_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Orchestrate three-tier fuzzy fallback when exact DB lookup returns nothing.

    Tier 1 — substring LIKE
    Tier 2 — anagram (same characters, different order; len ≥ 6)
    Tier 3 — near-match (±2 length, ≥80 % character overlap)

    Returns tagged rows from the first tier that produces matches, or an
    empty list when all tiers fail (caller should skip the name — it is
    likely a built-in or external-library function).
    """
    # Tier 1
    rows = _find_by_substring(db, name)
    if rows:
        return _tag_candidates(rows, name, "substring")

    # Tier 2
    if len(name) >= 6:
        candidate_names = _find_by_anagram(all_db_names, name)
        if candidate_names:
            rows = []
            for cname in candidate_names:
                rows.extend(db.select_by_name(cname))
            if rows:
                return _tag_candidates(rows, name, "anagram")

    # Tier 3
    candidate_names = _find_by_near_match(all_db_names, name)
    if candidate_names:
        rows = []
        for cname in candidate_names:
            rows.extend(db.select_by_name(cname))
        if rows:
            return _tag_candidates(rows, name, "near_match")

    # All tiers exhausted — treat as built-in / external, skip silently
    return []


# ---------------------------------------------------------------------------
# Step class
# ---------------------------------------------------------------------------

class ConsistencyQueryStep(CommonStep):
    def __init__(self, code_structure_db: CodeStructureDb):
        super().__init__()
        self.step_name = "Consistency Query Step"
        self.code_structure_db = code_structure_db

    def _execute_directly(self, state: ConsistencyEvaluationState):
        functions_and_classes = state["functions_and_classes"]
        # Fetch all names once to avoid a full-table-scan per lookup
        all_db_names: List[str] = self.code_structure_db.select_all_names()

        all_rows: list[any] = []
        for function_or_class in functions_and_classes:
            function_or_class_name = function_or_class["name"] if "name" in function_or_class else "N/A"
            function_or_class_file_path = function_or_class["file_path"] if "file_path" in function_or_class else "N/A"
            function_or_class_parameters = function_or_class["parameters"] if "parameters" in function_or_class else "N/A"
            function_or_class_parent = function_or_class["parent"] if "parent" in function_or_class else "N/A"
            self._print_step(state, step_output=(
                f"Consistency Query Step: \n{function_or_class_name},\n"
                f" {function_or_class_file_path},\n"
                f" {function_or_class_parameters},\n"
                f" {function_or_class_parent}"
            ))
            file_path = None
            parent = None
            name = None
            if "file_path" in function_or_class and function_or_class["file_path"] != "N/A":
                file_path = function_or_class["file_path"]
            if "parent" in function_or_class and function_or_class["parent"] != "N/A":
                parent = function_or_class["parent"]
            if "name" in function_or_class and function_or_class["name"] != "N/A":
                name = function_or_class["name"]

            rows: list[any] | None = None
            if name is None:
                if file_path is not None:
                    rows = self.code_structure_db.select_by_path(file_path)
                elif parent is not None:
                    rows = self.code_structure_db.select_by_parent(parent)
            else:
                if file_path is not None and parent is not None:
                    rows = self.code_structure_db.select_by_name_and_parent_and_path(name, parent, file_path)
                    rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name_and_path(name, file_path)
                        rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name_and_parent(name, parent)
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                elif file_path is not None:
                    rows = self.code_structure_db.select_by_name_and_path(name, file_path)
                    rows = rows if rows is None else [rows]
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                elif parent is not None:
                    rows = self.code_structure_db.select_by_name_and_parent(name, parent)
                    if rows is None or len(rows) == 0:
                        rows = self.code_structure_db.select_by_name(name)
                else:
                    rows = self.code_structure_db.select_by_name(name)

            if rows is None or len(rows) == 0:
                rows = _find_by_dot_normalized(self.code_structure_db, name)
                if rows:
                    self._print_step(
                        state,
                        step_output=f"'{name}' resolved via dot-normalization",
                    )

            if rows is None or len(rows) == 0:
                self._print_step(
                    state,
                    step_output=f"No exact match for '{name}'; trying fuzzy fallback",
                )
                rows = _find_fuzzy_candidates(self.code_structure_db, name, all_db_names)
                if not rows:
                    self._print_step(
                        state,
                        step_output=f"'{name}' not found in codebase after fuzzy search; skipping (likely built-in or external)",
                    )
                    continue

            all_rows.extend(rows)

        state["all_query_rows"] = all_rows

        return state, {**DEFAULT_TOKEN_USAGE}
