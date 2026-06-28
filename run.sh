#!/usr/bin/env bash
#
# run.sh — run the Quant Engine pipeline with full logging.
#
# - Logs everything (stdout + stderr) to logs/run_<timestamp>.log AND the console.
# - Each step prints [ OK ] or [FAIL]; on failure it stops and tells you exactly
#   which step failed, the exit code, and where the log is.
# - Captures provenance (git branch/commit, python version) at the top of the log.
#
# Usage:
#   ./run.sh                       # run the default entry (research/verify_pipeline.py)
#   ./run.sh --install             # install Python deps first, then run
#   ./run.sh --entry research/foo.py   # run a different Python entry script
#   ./run.sh --python python3.11   # use a specific interpreter
#   ./run.sh --check-only          # only verify environment/deps, don't run
#
# Exit code is 0 on success, non-zero (the failing step's code) on any failure.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"

# ---------------------------------------------------------------------------
# Defaults / args
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
ENTRY="research/verify_pipeline.py"
DO_INSTALL=0
CHECK_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --install)    DO_INSTALL=1; shift ;;
    --entry)      ENTRY="$2"; shift 2 ;;
    --python)     PYTHON="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 22
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging: send all output to console AND the log file
# ---------------------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

if [ -t 1 ]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'; BLU=$'\033[0;34m'; NC=$'\033[0m'
else
  RED=''; GRN=''; YEL=''; BLU=''; NC=''
fi

log()  { echo "${BLU}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo "${GRN}[ OK ]${NC} $*"; }
warn() { echo "${YEL}[WARN]${NC} $*"; }
err()  { echo "${RED}[FAIL]${NC} $*"; }

# Any unexpected error (not wrapped by run_step) lands here.
trap 'rc=$?; err "Unexpected error on line ${LINENO} (exit ${rc}): ${BASH_COMMAND}"; \
      err "Full log: ${LOG_FILE}"; exit "${rc}"' ERR

# run_step "name" cmd args... — runs a step, captures its exit code, reports it,
# and aborts the whole run (with that code) if it fails.
run_step() {
  local name="$1"; shift
  log "START: ${name}"
  local rc=0
  "$@" || rc=$?
  if [ "${rc}" -eq 0 ]; then
    ok "${name}"
  else
    err "${name}  (exit ${rc})"
    err "Pipeline stopped. Full log: ${LOG_FILE}"
    exit "${rc}"
  fi
}

# ---------------------------------------------------------------------------
# Header / provenance
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Quant Engine run  —  ${TS}"
echo " dir:    ${SCRIPT_DIR}"
echo " log:    ${LOG_FILE}"
echo " python: ${PYTHON}"
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
  echo " git:    $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
fi
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1 — interpreter present
# ---------------------------------------------------------------------------
check_python() {
  command -v "$PYTHON" >/dev/null 2>&1 || { echo "interpreter '$PYTHON' not found"; return 1; }
  "$PYTHON" --version
}
run_step "Check Python interpreter" check_python

# ---------------------------------------------------------------------------
# Step 2 — dependencies (install if asked; otherwise warn-only)
# ---------------------------------------------------------------------------
REQS=(pandas numpy yfinance matplotlib pyarrow)

install_deps() {
  if [ -f "requirements.txt" ]; then
    "$PYTHON" -m pip install -r requirements.txt
  else
    "$PYTHON" -m pip install "${REQS[@]}"
  fi
}

check_deps() {
  "$PYTHON" - <<'PY'
import importlib, sys
mods = {"pandas":"pandas","numpy":"numpy","yfinance":"yfinance",
        "matplotlib":"matplotlib","pyarrow":"pyarrow"}
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("missing packages: " + ", ".join(missing))
    sys.exit(1)
print("all required packages present")
PY
}

if [ "$DO_INSTALL" -eq 1 ]; then
  run_step "Install dependencies" install_deps
fi

# Dependency check is a WARNING (not fatal) unless --check-only, so the run can
# still proceed and surface the real error from Python if something's off.
if check_deps; then
  ok "Dependency check"
else
  warn "Some Python packages are missing. Re-run with: ./run.sh --install"
  [ "$CHECK_ONLY" -eq 1 ] && { err "Environment check failed."; exit 1; }
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  ok "Environment check complete (--check-only)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 3 — run the pipeline entry
# ---------------------------------------------------------------------------
[ -f "$ENTRY" ] || { err "Entry script not found: $ENTRY"; exit 1; }

run_entry() {
  # Run from the entry's own directory so relative paths (../engines, ../reports)
  # resolve correctly regardless of where run.sh was invoked from.
  ( cd "$(dirname "$ENTRY")" && "$PYTHON" "$(basename "$ENTRY")" )
}
run_step "Run pipeline: ${ENTRY}" run_entry

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "============================================================"
ok "Run complete."
echo " reports:     ${SCRIPT_DIR}/reports/   (open index.html)"
echo " full log:    ${LOG_FILE}"
echo "============================================================"
