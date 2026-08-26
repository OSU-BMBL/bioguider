#!/usr/bin/env bash
# One-liner status: files_done, total LLM 200s, pytest alive, duration-so-far.
RUN_DIR=${1:-$(ls -td outputs/multi_file_stress/run_* 2>/dev/null | head -1)}
LOG=$(cat /tmp/bench_log.txt 2>/dev/null || echo "")
if [ -z "$RUN_DIR" ]; then echo "no run dir"; exit 1; fi
files=$(find "$RUN_DIR" -maxdepth 2 -name STRESS_TEST_RESULTS.json 2>/dev/null | wc -l | tr -d ' ')
total=$(grep -c "200 OK" "$LOG" 2>/dev/null || echo 0)
alive=$(pgrep -f "pytest.*test_multi_file_full_matrix" 2>/dev/null | wc -l | tr -d ' ')
errors=$(grep -cE "401 Unauthorized|Traceback|Killed|AssertionError" "$LOG" 2>/dev/null || echo 0)
started=$(stat -f%m "$LOG" 2>/dev/null || echo 0)
now=$(date +%s)
age=$((now - started))
age_min=$((age / 60))
echo "run=$(basename $RUN_DIR) files=$files/10 llm_200=$total pytest_alive=$alive errors=$errors age=${age_min}m"
