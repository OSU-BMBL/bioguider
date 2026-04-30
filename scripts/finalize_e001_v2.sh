#!/usr/bin/env bash
# Finalize Task #4 once E001-v2 (test_multi_file_full_matrix) finishes.
#
# Runs:
#   1. Post-hoc protection violations on the latest multi_file_stress run
#   2. Re-runs the R figure script (now picks up E001-v2's AGGREGATE_TABLE.csv)
#   3. Reminds you to update the E001-v2 section of docs/EXPERIMENT_LOG.md
#
# Usage:
#   bash scripts/finalize_e001_v2.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LATEST_MULTI=$(ls -td outputs/multi_file_stress/run_2026* 2>/dev/null | head -1)
if [ -z "$LATEST_MULTI" ]; then
  echo "ERROR: no multi_file_stress run found"
  exit 1
fi
echo "Latest E001-v2 run: $LATEST_MULTI"

if [ ! -f "$LATEST_MULTI/_aggregate/AGGREGATE_TABLE.csv" ]; then
  echo "ERROR: AGGREGATE_TABLE.csv missing in $LATEST_MULTI/_aggregate/"
  echo "Step 4 may still be running or failed. Check the test log."
  exit 1
fi

echo
echo "=== 1. Post-hoc protection violations ==="
poetry run python scripts/compute_post_hoc_protection.py "$LATEST_MULTI"

echo
echo "=== 2. Regenerating figures ==="
Rscript scripts/benchmark_figures.R

echo
echo "=== 3. Headline numbers ==="
poetry run python - "$LATEST_MULTI" <<'PY'
import csv, sys
from collections import defaultdict

run = sys.argv[1]
agg = f"{run}/_aggregate/AGGREGATE_TABLE.csv"
with open(agg) as f:
    rows = list(csv.DictReader(f))

by_model = defaultdict(list)
for r in rows:
    by_model[r["model"]].append(float(r["f1_score_scorable"]))

print(f"\nModel ranking by mean F1_scorable across {len(rows)} cells:")
ranked = sorted(by_model.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
for i, (m, vs) in enumerate(ranked, 1):
    print(f"  {i}. {m:30s}  mean F1_scorable = {sum(vs)/len(vs):.4f}  (n={len(vs)})")
PY

echo
echo "=== Done. Now update docs/EXPERIMENT_LOG.md E001-v2 section ==="
echo "Aggregate CSV: $LATEST_MULTI/_aggregate/AGGREGATE_TABLE.csv"
echo "Protection CSV: $LATEST_MULTI/_aggregate/AGGREGATE_PROTECTION.csv"
echo "Figures: outputs/figures/"
