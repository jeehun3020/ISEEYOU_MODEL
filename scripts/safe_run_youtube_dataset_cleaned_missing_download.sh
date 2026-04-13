#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
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

run_step() {
  local name="$1"
  shift
  echo "[INFO] START $name $(date '+%Y-%m-%d %H:%M:%S')"
  "$@"
  echo "[INFO] DONE  $name $(date '+%Y-%m-%d %H:%M:%S')"
}

run_step download_train_cleaned_missing \
  python scripts/download_shorts_dataset.py \
    --csv data/youtube/final_experiment_manifest_youtube_redownloadable_20260407_train_missing_only.csv \
    --output-root /Volumes/SAMSUNG/CapstoneData/youtube_downloads/youtube_dataset_split/train \
    --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/youtube_dataset_train_cleaned_missing_download_report_20260407.json \
    --socket-timeout 20 \
    --retries 2

run_step download_val_cleaned_missing \
  python scripts/download_shorts_dataset.py \
    --csv data/youtube/final_experiment_manifest_youtube_redownloadable_20260407_val_missing_only.csv \
    --output-root /Volumes/SAMSUNG/CapstoneData/youtube_downloads/youtube_dataset_split/val \
    --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/youtube_dataset_val_cleaned_missing_download_report_20260407.json \
    --socket-timeout 20 \
    --retries 2

run_step download_test_cleaned_missing \
  python scripts/download_shorts_dataset.py \
    --csv data/youtube/final_experiment_manifest_youtube_redownloadable_20260407_test_missing_only.csv \
    --output-root /Volumes/SAMSUNG/CapstoneData/youtube_downloads/youtube_dataset_split/test \
    --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/youtube_dataset_test_cleaned_missing_download_report_20260407.json \
    --socket-timeout 20 \
    --retries 2

echo "[INFO] ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
