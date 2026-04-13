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

AMP_CFG="${1:-configs/protocol_youtube_dataset_plus_local_videoonly_clean_fft_amplitude.yaml}"
PHASE_CFG="${2:-configs/protocol_youtube_dataset_plus_local_videoonly_clean_fft_phase.yaml}"

AMP_LAST="outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_fft_amplitude/last.pt"
PHASE_LAST="outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_fft_phase/last.pt"

if [[ -f "$AMP_LAST" ]]; then
  python train.py --config "$AMP_CFG" --resume-from "$AMP_LAST"
else
  python train.py --config "$AMP_CFG"
fi
python eval.py --config "$AMP_CFG" --split test
python scripts/eval_lowfpr_calibration.py --config "$AMP_CFG" --split test

if [[ -f "$PHASE_LAST" ]]; then
  python train.py --config "$PHASE_CFG" --resume-from "$PHASE_LAST"
else
  python train.py --config "$PHASE_CFG"
fi
python eval.py --config "$PHASE_CFG" --split test
python scripts/eval_lowfpr_calibration.py --config "$PHASE_CFG" --split test

if [[ -x "scripts/run_youtube_dataset_plus_local_videoonly_clean_fft_followup.sh" ]]; then
  bash scripts/run_youtube_dataset_plus_local_videoonly_clean_fft_followup.sh
fi

echo "[INFO] resumed clean fft amplitude/phase runs complete"
