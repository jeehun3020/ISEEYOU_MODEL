#!/bin/zsh
set -euo pipefail

ROOT_DIR="/Users/jeongjihoon/Desktop/Capstone"
cd "$ROOT_DIR"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PATH="logs/protocol_audit_bundle_${STAMP}.log"

run_step() {
  local name="$1"
  shift
  echo "[START] ${name}" | tee -a "$LOG_PATH"
  "$@" 2>&1 | tee -a "$LOG_PATH"
  echo "[DONE] ${name}" | tee -a "$LOG_PATH"
}

run_step build_video_manifest \
  python scripts/build_video_manifest.py \
    --config configs/protocol_youtube_audit.yaml

run_step audit_video_manifest \
  python scripts/audit_video_manifest.py \
    --config configs/protocol_youtube_audit.yaml

run_step shortcut_audit \
  python scripts/run_protocol_shortcut_audit.py \
    --config configs/protocol_youtube_audit.yaml

echo "[INFO] protocol audit bundle complete: ${LOG_PATH}" | tee -a "$LOG_PATH"
