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

FRAME_BASE="${1:-configs/protocol_youtube_dataset_fullframe_baseline.yaml}"
FRAME_ROBUST="${2:-configs/protocol_youtube_dataset_fullframe_robustaug.yaml}"
TEMPORAL_CFG="${3:-configs/protocol_youtube_dataset_temporalconv.yaml}"
FROZEN_CFG="${4:-configs/protocol_youtube_dataset_frozen_convnext.yaml}"
FFT_AMP_CFG="${5:-configs/protocol_youtube_dataset_fft_amplitude.yaml}"
FFT_PHASE_CFG="${6:-configs/protocol_youtube_dataset_fft_phase.yaml}"

python scripts/build_video_manifest.py --config "$FRAME_BASE"
python scripts/audit_video_manifest.py --config "$FRAME_BASE"
python scripts/run_protocol_shortcut_audit.py --config "$FRAME_BASE"

python train.py --config "$FRAME_BASE"
python eval.py --config "$FRAME_BASE" --split test
python scripts/eval_lowfpr_calibration.py --config "$FRAME_BASE" --split test

python train.py --config "$FRAME_ROBUST"
python eval.py --config "$FRAME_ROBUST" --split test

python train_temporal.py --config "$TEMPORAL_CFG"
python eval_temporal.py --config "$TEMPORAL_CFG" --split test
python eval_temporal.py --config "$TEMPORAL_CFG" --split test --order-mode shuffle
python eval_temporal.py --config "$TEMPORAL_CFG" --split test --order-mode reverse

python train.py --config "$FROZEN_CFG"
python eval.py --config "$FROZEN_CFG" --split test

python train.py --config "$FFT_AMP_CFG"
python eval.py --config "$FFT_AMP_CFG" --split test

python train.py --config "$FFT_PHASE_CFG"
python eval.py --config "$FFT_PHASE_CFG" --split test

echo "[INFO] upgrade matrix complete"
