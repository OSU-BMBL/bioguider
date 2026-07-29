#!/bin/bash
# Simple-prompt token/time sweep for the 4 working LLMs (gpt-5.4 excluded —
# it 429s even on a single call). Writes one run dir per (rep, level) under
# outputs/simpleprompt_tokentime/. Designed to be launched fully detached:
#
#   nohup setsid bash benchmark/run_simpleprompt_sweep.sh > logs/simpleprompt_sweep.out 2>&1 &
#
# so it survives shell/harness teardown. Plot afterwards with:
#   python benchmark/plot_pipeline_tokentime.py --base outputs/simpleprompt_tokentime \
#       --strategy simple --out outputs/simpleprompt_tokentime/simpleprompt_tokentime_4llm.png

REPO=/bmbl_data/shaohong/projects/github/bioguider
PY=/home/shaohong/.conda/envs/bioguider/bin/python
cd "$REPO" || exit 2

MODELS="${MODELS:-gpt-4o,kimi-k2.5,glm-5.1,gpt-oss}"
REPS="${REPS:-5}"
LEVELS="${LEVELS:-40 100 150 200}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/simpleprompt_sweep_${TS}.log"
echo "master log: $LOG  models=$MODELS reps=$REPS levels=$LEVELS"

for rep in $(seq 1 "$REPS"); do
  for level in $LEVELS; do
    echo "[$(date '+%H:%M:%S')] START rep=$rep level=$level" >> "$LOG"
    PIPELINE_OUTPUT_BASE="outputs/simpleprompt_tokentime" \
    PHAROKKA_MODELS="$MODELS" \
    PHAROKKA_STRATEGIES=simple \
    PHAROKKA_ERROR_LEVEL="$level" \
    PHAROKKA_MAX_WORKERS=4 \
    "$PY" -m pytest \
      benchmark/test_pharokka_pipeline.py::test_pipeline_vs_prompt_pharokka -s -q \
      >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE rep=$rep level=$level rc=$?" >> "$LOG"
  done
done
echo "[$(date '+%H:%M:%S')] SIMPLE SWEEP COMPLETE" >> "$LOG"
