#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/jeongjihoon/Desktop/Capstone"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

CELEB="$ROOT/Celeb_V2"
GENIMG="$ROOT/gen_img"
LOG_FILE="$LOG_DIR/resume_manifest_after_move_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] waiting for SSD move to finish" | tee -a "$LOG_FILE"

while true; do
  celeb_ready=0
  genimg_ready=0

  if [[ -L "$CELEB" ]]; then
    celeb_ready=1
  fi
  if [[ -L "$GENIMG" ]]; then
    genimg_ready=1
  fi

  if [[ "$celeb_ready" == "1" && "$genimg_ready" == "1" ]]; then
    break
  fi

  echo "[INFO] still waiting: Celeb_V2 symlink=$celeb_ready gen_img symlink=$genimg_ready" | tee -a "$LOG_FILE"
  sleep 30
done

echo "[INFO] SSD move finished, starting combined manifest build" | tee -a "$LOG_FILE"

cd "$ROOT"
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python scripts/build_video_manifest.py \
  --config configs/protocol_youtube_dataset_plus_local_baseline.yaml \
  2>&1 | tee -a "$LOG_FILE"
