#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

REAL_CSV="data/youtube/shorts_urls_channels_real.csv"
GEN_CSV="data/youtube/shorts_urls_channels_generated.csv"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data/raw/youtube_external/shorts_binary}"
CONFIG_PATH="${CONFIG_PATH:-configs/baseline_binary_shorts.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/checkpoints_shorts_binary/best.pt}"
REPORT_PATH="${REPORT_PATH:-data/youtube_external_manifests/shorts_binary_download_report.json}"

# TODO: Increase this after securing more distinct real channels.
LIMIT_PER_LABEL="${LIMIT_PER_LABEL:-120}"
MAX_PER_SOURCE="${MAX_PER_SOURCE:-60}"
SOCKET_TIMEOUT="${SOCKET_TIMEOUT:-20}"
RETRIES="${RETRIES:-2}"

python scripts/download_shorts_dataset.py \
  --csv "$REAL_CSV" \
  --csv "$GEN_CSV" \
  --output-root "$DOWNLOAD_ROOT" \
  --limit-per-label "$LIMIT_PER_LABEL" \
  --max-per-source "$MAX_PER_SOURCE" \
  --socket-timeout "$SOCKET_TIMEOUT" \
  --retries "$RETRIES" \
  --report-path "$REPORT_PATH"

python prepare_data.py --config "$CONFIG_PATH"
python train.py --config "$CONFIG_PATH"
python eval.py --config "$CONFIG_PATH" --split val --checkpoint "$CHECKPOINT_PATH"

echo "[DONE] Shorts binary experiment finished."
