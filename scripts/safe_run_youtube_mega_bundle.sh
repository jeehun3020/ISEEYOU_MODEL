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
master_log="logs/youtube_mega_bundle_${timestamp}.log"

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

run_step mega_real_download \
  python scripts/download_shorts_dataset.py \
  --csv data/youtube/mega_youtube_real_queue_20260331.csv \
  --output-root /Volumes/SAMSUNG/CapstoneData/youtube_raw/mega_real_20260331 \
  --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/mega_real_download_report_20260331.json \
  --limit-total 2500 \
  --max-per-source 20 \
  --socket-timeout 20 \
  --retries 2

run_step mega_generated_download \
  python scripts/download_shorts_dataset.py \
  --csv data/youtube/mega_youtube_generated_queue_20260331.csv \
  --output-root /Volumes/SAMSUNG/CapstoneData/youtube_raw/mega_generated_20260331 \
  --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/mega_generated_download_report_20260331.json \
  --limit-total 3000 \
  --max-per-source 20 \
  --socket-timeout 20 \
  --retries 2

run_step mega_prepare \
  python prepare_data.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega.yaml

run_step mega_train \
  python train.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega.yaml

run_step mega_eval \
  python eval.py \
  --config configs/baseline_binary_allvideo_textmask_protocol_youtube_mega.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary_textmask_protocol_youtube_mega/best.pt \
  --split test

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$master_log"
