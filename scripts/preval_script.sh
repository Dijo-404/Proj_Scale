#!/usr/bin/env bash

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
INFERENCE_TIMEOUT=1200

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
export PING_URL

PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

PYTHON_BIN="python"
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

OPENENV_CMD="openenv"
if ! command -v "$OPENENV_CMD" >/dev/null 2>&1; then
  if [ -x "$REPO_DIR/.venv/bin/openenv" ]; then
    OPENENV_CMD="$REPO_DIR/.venv/bin/openenv"
  else
    OPENENV_CMD=""
  fi
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
log "Python:   $PYTHON_BIN"
printf "\n"

log "${BOLD}Step 1/7: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check network connectivity and confirm the Space is in Running state."
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Ensure the URL is correct and your Space is healthy."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/7: Mandatory submission requirements${NC} ..."

if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "inference.py is missing from repo root"
  stop_at "Step 2"
fi

if [ ! -f "$REPO_DIR/openenv.yaml" ]; then
  fail "openenv.yaml is missing"
  stop_at "Step 2"
fi

if [ -z "${API_BASE_URL:-}" ]; then
  fail "API_BASE_URL is not defined in current environment"
  hint "Export API_BASE_URL before running this validator."
  stop_at "Step 2"
fi

if [ -z "${MODEL_NAME:-}" ]; then
  fail "MODEL_NAME is not defined in current environment"
  hint "Export MODEL_NAME before running this validator."
  stop_at "Step 2"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  fail "HF_TOKEN is not defined in current environment"
  hint "Export HF_TOKEN before running this validator."
  stop_at "Step 2"
fi

if ! grep -R --line-number --quiet "from openai import OpenAI" "$REPO_DIR"/inference*.py; then
  fail "OpenAI client import not found in inference modules"
  stop_at "Step 2"
fi

pass "inference.py/openenv.yaml present, required env vars defined, OpenAI client detected"

log "${BOLD}Step 3/7: OpenEnv spec validation${NC} ..."

if [ -z "$OPENENV_CMD" ]; then
  fail "openenv command not found"
  hint "Install openenv-core in your active environment."
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && "$OPENENV_CMD" validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/7: Docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 4"
fi

DOCKER_CONTEXT="$REPO_DIR"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -40
  stop_at "Step 4"
fi

log "${BOLD}Step 5/7: Baseline run reproduces and emits strict logs${NC} ..."

OUT_STDOUT=$(portable_mktemp "validate-infer-out")
OUT_STDERR=$(portable_mktemp "validate-infer-err")
CLEANUP_FILES+=("$OUT_STDOUT" "$OUT_STDERR")

RUN_OK=false
if run_with_timeout "$INFERENCE_TIMEOUT" \
  env API_BASE_URL="$API_BASE_URL" MODEL_NAME="$MODEL_NAME" HF_TOKEN="$HF_TOKEN" FORCE_HEURISTIC=1 \
  "$PYTHON_BIN" "$REPO_DIR/inference.py" \
  --env-base-url "$PING_URL" \
  --task easy_access_recovery \
  --success-threshold 0.0 \
  --max-steps 12 \
  >"$OUT_STDOUT" 2>"$OUT_STDERR"; then
  RUN_OK=true
fi

if [ "$RUN_OK" != true ]; then
  fail "inference.py execution failed or timed out (<20 min required)"
  printf "%s\n" "--- stderr ---"
  tail -40 "$OUT_STDERR"
  stop_at "Step 5"
fi

if "$PYTHON_BIN" - "$OUT_STDOUT" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]

if not lines:
    raise SystemExit("no stdout output from inference run")

allowed_prefixes = ("[START]", "[STEP]", "[END]")
if any(not line.startswith(allowed_prefixes) for line in lines):
    raise SystemExit("stdout includes non-contract lines (must be START/STEP/END only)")

start_re = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
step_re = re.compile(r"^\[STEP\] step=(\d+) action=(.+) reward=(\d+\.\d{2}) done=(true|false) error=(.*)$")
end_re = re.compile(
    r"^\[END\] success=(true|false) steps=(?P<steps>\d+) "
    r"score=(?P<score>(?:0(?:\.\d+)?|1(?:\.0+)?)) "
    r"rewards=(?P<rewards>\d+\.\d{2}(,\d+\.\d{2})*)?$"
)

if sum(1 for line in lines if line.startswith("[START]")) != 1:
    raise SystemExit("expected exactly one [START] line")
if sum(1 for line in lines if line.startswith("[END]")) != 1:
    raise SystemExit("expected exactly one [END] line")
if not lines[0].startswith("[START]"):
    raise SystemExit("first stdout line must be [START]")
if not lines[-1].startswith("[END]"):
    raise SystemExit("last stdout line must be [END]")

if not start_re.match(lines[0]):
    raise SystemExit("[START] line format invalid")

for line in lines[1:-1]:
    m = step_re.match(line)
    if not m:
      raise SystemExit(f"invalid [STEP] line format: {line}")

m_end = end_re.match(lines[-1])
if not m_end:
    raise SystemExit("[END] line format invalid")

steps = int(m_end.group("steps"))
rewards_str = m_end.group("rewards")
if rewards_str:
    rewards = rewards_str.split(",")
    if len(rewards) != steps:
        raise SystemExit(f"steps ({steps}) != number of rewards ({len(rewards)})")

print("stdout format OK")
PY
then
  pass "inference.py run completed and strict stdout contract validated"
else
  fail "inference.py stdout does not match required format"
  printf "%s\n" "--- stdout ---"
  cat "$OUT_STDOUT"
  stop_at "Step 5"
fi

log "${BOLD}Step 6/7: Task count and grader behavior${NC} ..."

if "$PYTHON_BIN" - "$REPO_DIR" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo))

from graders import TASK_GRADERS
from tasks import available_tasks, get_task

names = tuple(available_tasks())
if len(names) < 3:
    raise SystemExit(f"Expected at least 3 tasks, found {len(names)}")

# Verify each task has at least one registered grader
missing = [name for name in names if not TASK_GRADERS.get(name)]
if missing:
    raise SystemExit(f"Tasks missing graders: {missing}")

# Score sanity: totals are strict (0,1) and non-constant on realistic state change
for name in names:
    task = get_task(name)
    grader = TASK_GRADERS[name]

    zero_tickets = {
        t.ticket_id: {
            "priority": "low",
            "category": "access",
            "team": "tier1",
            "status": "new",
            "last_reply": None,
        }
        for t in task.tickets
    }

    perfect_tickets = {
        ticket_id: {
            "priority": goal.priority,
            "category": goal.category,
            "team": goal.team,
            "status": goal.status,
            "last_reply": " ".join(goal.reply_rule.required_keywords)
            + " extra context to satisfy length",
        }
        for ticket_id, goal in task.goals.items()
    }

    z_breakdown = grader(zero_tickets, [])
    p_breakdown = grader(perfect_tickets, [])

    z = float(z_breakdown.get("total", 0.0))
    p = float(p_breakdown.get("total", 0.0))

    if not (0.0 < z < 1.0):
        raise SystemExit(f"Task {name} zero-state total must be in (0,1), got {z}")
    if not (0.0 < p < 1.0):
        raise SystemExit(f"Task {name} perfect-state total must be in (0,1), got {p}")
    if abs(z - p) < 1e-9:
        raise SystemExit(f"Task {name} grader total appears constant ({z})")

print("grader checks OK")
PY
then
  pass "At least 3 tasks present and graders pass strict behavior checks"
else
  fail "Task/grader integrity checks failed"
  stop_at "Step 6"
fi

log "${BOLD}Step 7/7: Repo hygiene checks${NC} ..."

if ! grep -R --line-number --quiet "openenv" "$REPO_DIR/README.md"; then
  fail "README.md must include the openenv tag/reference"
  stop_at "Step 7"
fi

TOKEN_REGEX="(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})"
TOKEN_HIT=false

if git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git -C "$REPO_DIR" grep --line-number -E "$TOKEN_REGEX" -- . ":(exclude)scripts/preval_script.sh" >/dev/null 2>&1; then
    TOKEN_HIT=true
  fi
else
  if grep -R --line-number --binary-files=without-match -E "$TOKEN_REGEX" \
    --exclude="preval_script.sh" \
    --exclude-dir=".git" \
    --exclude-dir=".venv" \
    --exclude-dir="__pycache__" \
    "$REPO_DIR" >/dev/null 2>&1; then
    TOKEN_HIT=true
  fi
fi

if [ "$TOKEN_HIT" = true ]; then
  fail "Potential hardcoded API tokens found in repository"
  hint "Move secrets to environment variables and remove committed tokens."
  stop_at "Step 7"
fi

pass "README includes openenv reference and no obvious hardcoded token patterns"

printf "\n"
printf "${GREEN}${BOLD}Validation complete: %d/7 checks passed.${NC}\n" "$PASS"
printf "${GREEN}Proj_Scale appears submission-ready for the OpenEnv track.${NC}\n"
printf "\n"
