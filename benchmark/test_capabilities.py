"""Capability benchmark entry point: tool-calling + structured-output.

Usage
-----
    # default model set
    pytest benchmark/test_capabilities.py -v -s

    # pick models (comma-separated, names from benchmark.shared.MODELS)
    CAPABILITY_MODELS="gpt-4o,gpt-oss,kimi-k2.5" pytest benchmark/test_capabilities.py -s

    # tune concurrency / structured-output method
    CAPABILITY_MAX_WORKERS=2 CAPABILITY_STRUCT_METHOD=json_schema \
        pytest benchmark/test_capabilities.py -s

Writes per-run artifacts to ``outputs/capability_bench/run_<ts>/``:
    capability_results.json   full scorecards + per-task detail
    capability_summary.csv    one row per model
and prints a leaderboard table to stdout.
"""
from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from benchmark.capabilities.runner import run_model

# gpt-5.4 is excluded by default (throttled proxy deployment — see
# benchmark/shared.py MODELS notes); add it back via CAPABILITY_MODELS.
# glm-5 (the old deployment) is dead (HTTP 410); glm-5.1 replaces it.
DEFAULT_MODELS = "gpt-4o,gpt-oss,kimi-k2.5,glm-5.1"


def _models() -> List[str]:
    raw = os.environ.get("CAPABILITY_MODELS", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _modes() -> List[str]:
    """CAPABILITY_MODE: native (default) | prompt | both."""
    mode = os.environ.get("CAPABILITY_MODE", "native").strip().lower()
    return ["native", "prompt"] if mode == "both" else [mode]


def _print_table(cards: List[Dict]) -> None:
    cards = sorted(cards, key=lambda c: c["overall"], reverse=True)
    hdr = (
        f"{'model':<22}{'overall':>8}{'tool.sel':>9}{'tool.arg':>9}"
        f"{'abstain':>9}{'schema':>8}{'field':>8}{'exact':>8}{'sec':>7}"
    )
    print("\n" + "=" * len(hdr))
    print("CAPABILITY BENCHMARK — tool calling + structured output")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for c in cards:
        t, s = c["tool"], c["struct"]
        print(
            f"{c.get('label', c['model']):<22}{c['overall']:>8.3f}{t['selection_rate']:>9.3f}"
            f"{t['args_rate']:>9.3f}{t['abstention_rate']:>9.3f}"
            f"{s['schema_valid_rate']:>8.3f}{s['field_accuracy']:>8.3f}"
            f"{s['exact_match_rate']:>8.3f}{c['duration_s']:>7.1f}"
        )
    print("=" * len(hdr) + "\n")


def _write_csv(path: Path, cards: List[Dict]) -> None:
    cols = [
        "model", "mode", "overall", "duration_s",
        "tool_selection_rate", "tool_args_rate", "tool_valid_call_rate", "tool_abstention_rate",
        "struct_schema_valid_rate", "struct_field_accuracy", "struct_exact_match_rate",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in sorted(cards, key=lambda x: x["overall"], reverse=True):
            t, s = c["tool"], c["struct"]
            w.writerow([
                c["model"], c.get("mode", "native"), c["overall"], c["duration_s"],
                t["selection_rate"], t["args_rate"], t["valid_call_rate"], t["abstention_rate"],
                s["schema_valid_rate"], s["field_accuracy"], s["exact_match_rate"],
            ])


def test_capabilities():
    models = _models()
    modes = _modes()
    combos = [(m, mode) for m in models for mode in modes]
    max_workers = int(os.environ.get("CAPABILITY_MAX_WORKERS", "3"))

    out_dir = Path("outputs/capability_bench") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(combos))) as pool:
        futs = {pool.submit(run_model, m, mode): (m, mode) for m, mode in combos}
        for fut in as_completed(futs):
            m, mode = futs[fut]
            try:
                cards.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {m}[{mode}]: {type(e).__name__}: {e}")

    assert cards, "no model produced a scorecard"

    (out_dir / "capability_results.json").write_text(json.dumps(cards, indent=2))
    _write_csv(out_dir / "capability_summary.csv", cards)
    _print_table(cards)
    print(f"artifacts: {out_dir}")
