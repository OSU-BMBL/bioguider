"""Unit tests for bioguider.analysis.doc_stats."""

import json
import math
from pathlib import Path

import pytest

from bioguider.analysis.doc_stats import (
    _word_count,
    apply_length_weighting,
    save_doc_stats,
)


def test_word_count():
    assert _word_count("hello world") == 2
    assert _word_count("  spaced  out  ") == 2
    assert _word_count("") == 0
    assert _word_count("single") == 1


def test_length_weighting_increases_long_doc_score():
    stats = [
        {"software": "short", "total_words": 100, "readme_score": 90,
         "installation_score": 0, "userguide_score": 0, "tutorial_score": 0},
        {"software": "long", "total_words": 10000, "readme_score": 70,
         "installation_score": 0, "userguide_score": 0, "tutorial_score": 0},
    ]
    result = apply_length_weighting(stats, alpha=0.5)
    short_adj = result[0]["readme_score_adjusted"]
    long_adj = result[1]["readme_score_adjusted"]
    assert long_adj > result[1]["readme_score"], "Long doc score should increase"
    assert short_adj > 0


def test_length_weighting_zero_words():
    stats = [{"software": "empty", "total_words": 0, "readme_score": 50,
              "installation_score": 0, "userguide_score": 0, "tutorial_score": 0}]
    result = apply_length_weighting(stats)
    assert result[0]["readme_score_adjusted"] == 50.0


def test_save_doc_stats_creates_csv(tmp_path):
    stats = [
        {"software": "test_pkg", "repo_key": "test/pkg",
         "num_userguides": 2, "num_tutorials": 5, "total_words": 3000,
         "readme_score": 85, "installation_score": 70,
         "userguide_score": 60, "tutorial_score": 55},
    ]
    out = tmp_path / "stats.csv"
    save_doc_stats(stats, out)
    assert out.exists()
    content = out.read_text()
    assert "test_pkg" in content
    assert "3000" in content


def test_save_doc_stats_empty_list(tmp_path):
    out = tmp_path / "empty.csv"
    save_doc_stats([], out)
