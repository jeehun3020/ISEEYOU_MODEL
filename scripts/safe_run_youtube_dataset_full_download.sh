#!/bin/zsh
set -euo pipefail

ROOT="/Users/jeongjihoon/Desktop/Capstone"
DATA_ROOT="/Volumes/SAMSUNG/CapstoneData"

cd "$ROOT"
source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="logs/youtube_dataset_full_download_${timestamp}.log"

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

run_step build_download_ready \
  python scripts/build_download_ready_from_youtube_dataset.py \
  --input-csv "${DATA_ROOT}/youtube_dataset/all_videos_with_split.csv" \
  --output-dir "${DATA_ROOT}/youtube_manifests"

run_step download_train \
  caffeinate -dimsu python scripts/download_shorts_dataset.py \
  --csv "${DATA_ROOT}/youtube_manifests/youtube_dataset_train_download_ready.csv" \
  --output-root "${DATA_ROOT}/youtube_downloads/youtube_dataset_split/train" \
  --report-path "${DATA_ROOT}/youtube_manifests/youtube_dataset_train_download_report.json" \
  --limit-total 0 \
  --max-per-source 0 \
  --socket-timeout 20 \
  --retries 2

run_step download_val \
  caffeinate -dimsu python scripts/download_shorts_dataset.py \
  --csv "${DATA_ROOT}/youtube_manifests/youtube_dataset_val_download_ready.csv" \
  --output-root "${DATA_ROOT}/youtube_downloads/youtube_dataset_split/val" \
  --report-path "${DATA_ROOT}/youtube_manifests/youtube_dataset_val_download_report.json" \
  --limit-total 0 \
  --max-per-source 0 \
  --socket-timeout 20 \
  --retries 2

run_step download_test \
  caffeinate -dimsu python scripts/download_shorts_dataset.py \
  --csv "${DATA_ROOT}/youtube_manifests/youtube_dataset_test_download_ready.csv" \
  --output-root "${DATA_ROOT}/youtube_downloads/youtube_dataset_split/test" \
  --report-path "${DATA_ROOT}/youtube_manifests/youtube_dataset_test_download_report.json" \
  --limit-total 0 \
  --max-per-source 0 \
  --socket-timeout 20 \
  --retries 2

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$master_log"
