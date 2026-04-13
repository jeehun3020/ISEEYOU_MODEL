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

CONFIG="${1:-configs/protocol_youtube_dataset_fullframe_baseline.yaml}"

python scripts/build_video_manifest.py --config "$CONFIG"
python scripts/audit_video_manifest.py --config "$CONFIG"
python scripts/run_protocol_shortcut_audit.py --config "$CONFIG"
python train.py --config "$CONFIG"
python eval.py --config "$CONFIG" --split test
python scripts/eval_lowfpr_calibration.py --config "$CONFIG" --split test

echo "[INFO] baseline protocol run complete for $CONFIG"
