#!/usr/bin/env bash
# Launch the multi-file benchmark in background with hygienic env.
# Protects against stale shell OPENAI_API_KEY beating .env (dotenv load_dotenv()
# does not override existing env vars — that footgun cost us an hour of 401 debugging).
#
# Usage:
#   scripts/bench_launch.sh [test_name]
# Default test: test_multi_file_full_matrix

TEST="${1:-test_multi_file_full_matrix}"
mkdir -p logs outputs/multi_file_stress
LOG="logs/multi_file_stress_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > /tmp/bench_log.txt
echo "LOG=$LOG"
echo "TEST=$TEST"

# Clear any stale shell vars that could override .env.
unset OPENAI_API_KEY OPENAI_API_KEY_OLD AZURE_OPENAI_API_KEY

# Quick proxy probe — abort early if key is wrong.
KEY=$(grep -E '^OPENAI_API_KEY=' .env | sed 's/^OPENAI_API_KEY=//')
BASE=$(grep -E '^OPENAI_BASE_URL=' .env | sed 's/^OPENAI_BASE_URL=//')
if [ -z "$KEY" ] || [ -z "$BASE" ]; then
  echo "ERROR: .env missing OPENAI_API_KEY or OPENAI_BASE_URL" >&2
  exit 2
fi
HTTP=$(curl -sm 5 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "$BASE/models")
if [ "$HTTP" != "200" ]; then
  echo "ERROR: proxy probe returned HTTP $HTTP (expected 200)" >&2
  exit 2
fi
echo "proxy probe OK (HTTP $HTTP)"

# Hand off.
poetry run pytest "system_tests/test_single_file_stress.py::$TEST" -v -s 2>&1 | tee "$LOG"
