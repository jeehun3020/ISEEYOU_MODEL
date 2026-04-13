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
log_path="logs/youtube_generated_resume_${timestamp}.log"

echo "[$(date '+%F %T')] START mega_generated_resume" | tee -a "$log_path"

caffeinate -dimsu python scripts/download_shorts_dataset.py \
  --csv data/youtube/mega_youtube_generated_queue_20260331.csv \
  --output-root /Volumes/SAMSUNG/CapstoneData/youtube_raw/mega_generated_20260331 \
  --report-path /Volumes/SAMSUNG/CapstoneData/youtube_manifests/mega_generated_download_report_20260331.json \
  --limit-total 3000 \
  --max-per-source 20 \
  --socket-timeout 20 \
  --retries 2 \
  2>&1 | tee -a "$log_path"

echo "[$(date '+%F %T')] DONE mega_generated_resume" | tee -a "$log_path"
