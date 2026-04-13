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

echo "[INFO] Upgrade matrix candidates"
echo "1) Frame robust spatial: configs/baseline_binary_allvideo_textmask_protocol_robustaug.yaml"
echo "2) Frozen image encoder: configs/baseline_binary_allvideo_textmask_protocol_frozen_convnext.yaml"
echo "3) Temporal temporal-conv: configs/temporal_binary_allvideo_textmask_protocol_temporalconv.yaml"
echo "4) Temporal GRU: configs/temporal_binary_allvideo_textmask_protocol_gru.yaml"
echo "5) FFT amplitude: configs/baseline_binary_allvideo_fft_amplitude_protocol.yaml"
echo "6) FFT phase: configs/baseline_binary_allvideo_fft_phase_protocol.yaml"
