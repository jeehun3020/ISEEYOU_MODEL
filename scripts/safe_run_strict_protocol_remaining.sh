#!/bin/zsh
set -euo pipefail

ROOT="/Users/jeongjihoon/Desktop/Capstone"
cd "$ROOT"

source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/strict_protocol_remaining_${timestamp}.log"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START ${name}" | tee -a "$master_log"
  set +e
  "$@" 2>&1 | tee -a "logs/${name}_${timestamp}.log" | tee -a "$master_log"
  local cmd_status=$?
  set -e
  if [[ $cmd_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAIL ${name} exit=${cmd_status}" | tee -a "$master_log"
    exit "$cmd_status"
  fi
  echo "[$(date '+%F %T')] DONE ${name}" | tee -a "$master_log"
}

run_step generated_bulk_download \
  python scripts/download_shorts_dataset.py \
  --csv data/youtube/generated_youtube_downloaded_manifest_newonly_20260326.csv \
  --output-root data/raw/youtube_external/generated_bulk_20260326 \
  --report-path data/youtube_external_manifests/generated_bulk_download_report_20260326.json \
  --limit-total 600 \
  --max-per-source 12 \
  --socket-timeout 20 \
  --retries 2

run_step strict_protocol_prepare \
  python prepare_data.py \
  --config configs/baseline_binary_allvideo_textmask_protocol.yaml

run_step strict_protocol_train \
  python train.py \
  --config configs/baseline_binary_allvideo_textmask_protocol.yaml

run_step strict_protocol_eval_trained \
  python eval.py \
  --config configs/baseline_binary_allvideo_textmask_protocol.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary_textmask_protocol/best.pt \
  --split test

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$master_log"
