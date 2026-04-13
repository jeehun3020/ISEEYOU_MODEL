#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/jeongjihoon/Desktop/Capstone"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CONFIG="configs/protocol_youtube_dataset_plus_local_baseline.yaml"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/resume_plus_local_experiment_after_manifest_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] waiting for manifest build to finish" | tee -a "$LOG_FILE"
while pgrep -f "build_video_manifest.py --config $CONFIG" >/dev/null; do
  echo "[INFO] manifest still building..." | tee -a "$LOG_FILE"
  sleep 30
done

echo "[INFO] starting audit + baseline experiment for $CONFIG" | tee -a "$LOG_FILE"
python scripts/audit_video_manifest.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
python scripts/run_protocol_shortcut_audit.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
python train.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
python eval.py --config "$CONFIG" --split test 2>&1 | tee -a "$LOG_FILE"
python scripts/eval_lowfpr_calibration.py --config "$CONFIG" --split test 2>&1 | tee -a "$LOG_FILE"
echo "[INFO] plus-local baseline experiment complete" | tee -a "$LOG_FILE"
