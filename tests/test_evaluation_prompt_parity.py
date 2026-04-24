"""Tests for D5 — evaluation prompt extracted to a reusable .txt file.

Both BioGuider and the Claude-Code comparison run must consume the same
prompt text. This test locks down parity:

1. The ``.txt`` template loads cleanly through ``PromptLoader``.
2. ``EVALUATION_README_SYSTEM_PROMPT`` (module-level constant) and
   ``get_evaluation_prompt("readme")`` both resolve to the file content.
3. All ``.format`` placeholders that the evaluation pipeline relies on are
   intact (catches accidental edits that would break ``.format(**kwargs)``).
4. Byte-length snapshot guards against silent edits.
"""

from pathlib import Path

import pytest

from bioguider.agents.evaluation_task import EVALUATION_README_SYSTEM_PROMPT
from bioguider.generation.prompts import (
    get_evaluation_prompt,
    load_prompt,
)


EXPECTED_PLACEHOLDERS = {
    "{readme_path}",
    "{readme_content}",
    "{flesch_reading_ease}",
    "{flesch_kincaid_grade}",
    "{gunning_fog_index}",
    "{smog_index}",
}

EXPECTED_SECTION_MARKERS = (
    "### **Step 1:",
    "### **Evaluation Criteria**",
    "**1. Project Clarity & Purpose**",
    "**2. Installation Instructions**",
    "**3. Usage Instructions**",
    "**4. Contributing Guidelines**",
    "**5. License Information**",
    "**6. Readability Analysis**",
    "### Final Report Format",
    "**FinalAnswer**",
)

# Byte-length snapshot. If this changes, you intentionally edited the prompt —
# update this constant in the same commit.
EXPECTED_BYTE_LEN = 5882


class TestPromptFileExists:
    def test_txt_file_present(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "bioguider"
            / "generation"
            / "prompts"
            / "evaluation_readme.txt"
        )
        assert path.exists(), f"Missing extracted prompt: {path}"

    def test_load_via_prompt_loader(self):
        text = load_prompt("evaluation_readme")
        assert isinstance(text, str)
        assert len(text) > 100


class TestParity:
    def test_constant_equals_file(self):
        assert EVALUATION_README_SYSTEM_PROMPT == load_prompt("evaluation_readme")

    def test_public_accessor_matches(self):
        assert get_evaluation_prompt("readme") == EVALUATION_README_SYSTEM_PROMPT

    def test_unknown_accessor_category_raises(self):
        with pytest.raises(ValueError, match="Unknown evaluation prompt category"):
            get_evaluation_prompt("nope")


class TestStructuralInvariants:
    def test_all_placeholders_present(self):
        for ph in EXPECTED_PLACEHOLDERS:
            assert ph in EVALUATION_README_SYSTEM_PROMPT, f"Missing placeholder: {ph}"

    def test_placeholders_format_successfully(self):
        # Any accidental brace typo would blow up here.
        filled = EVALUATION_README_SYSTEM_PROMPT.format(
            readme_path="/tmp/README.md",
            readme_content="# Example\n",
            flesch_reading_ease=55.0,
            flesch_kincaid_grade=9.2,
            gunning_fog_index=10.8,
            smog_index=11.1,
        )
        assert "/tmp/README.md" in filled
        assert "# Example" in filled

    def test_section_markers_present(self):
        for marker in EXPECTED_SECTION_MARKERS:
            assert marker in EVALUATION_README_SYSTEM_PROMPT, f"Missing marker: {marker!r}"

    def test_byte_length_snapshot(self):
        actual = len(EVALUATION_README_SYSTEM_PROMPT)
        assert actual == EXPECTED_BYTE_LEN, (
            f"Prompt byte-length changed ({actual} vs snapshot {EXPECTED_BYTE_LEN}). "
            "If this edit was intentional, update EXPECTED_BYTE_LEN in the same commit."
        )
