#!/usr/bin/env bash
# Show the corrupted→fixed delta for one (file, level, model) cell.
# Useful for qualitative inspection: "did this model actually fix the
# prose_code_pkg_version injection, or did it regurgitate the corruption?"
#
# Usage:
#   scripts/bench_diff.sh <run_dir> <file_stem> <level> <model>
# Example:
#   scripts/bench_diff.sh outputs/multi_file_stress/run_20260424_022419 \
#     de_vignette 20 kimi-k2.5
set -eu
RUN_DIR="${1:?missing run_dir}"
STEM="${2:?missing file_stem}"
LEVEL="${3:?missing level}"
MODEL="${4:?missing model}"

CORR="$RUN_DIR/$STEM/$STEM.level_$LEVEL.corrupted.Rmd"
FIXED="$RUN_DIR/$STEM/$STEM.level_$LEVEL.${MODEL}_bioguider.fixed.Rmd"
ORIG="$RUN_DIR/$STEM/$STEM.original.Rmd"

for f in "$CORR" "$FIXED" "$ORIG"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f" >&2
    exit 2
  fi
done

MANIFEST="$RUN_DIR/$STEM/$STEM.level_$LEVEL.manifest.json"
echo "=== $STEM @ level=$LEVEL model=$MODEL ==="
echo
if [ -f "$MANIFEST" ]; then
  echo "--- injected errors (manifest) ---"
  # Pass the path through an env var so a crafted stem/level/model param
  # can't inject Python via string interpolation into the -c payload.
  MANIFEST="$MANIFEST" poetry run python -c '
import os, json
from collections import Counter
m = json.load(open(os.environ["MANIFEST"]))
errs = m.get("errors", [])
print(f"  total: {len(errs)}")
cats = Counter(e.get("category", "?") for e in errs)
for c, n in cats.most_common():
    print(f"    {c}: {n}")
' 2>/dev/null
fi
echo
echo "--- corrupted vs fixed (unified diff, first 80 lines) ---"
diff -u "$CORR" "$FIXED" | head -80
echo
echo "--- original vs fixed (unified diff, first 80 lines) ---"
diff -u "$ORIG" "$FIXED" | head -80
