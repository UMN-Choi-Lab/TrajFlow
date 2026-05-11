#!/bin/bash
# ETH/UCY multi-seed study: CDE-CNF marginal, 3 seeds across all 5 leave-one-out folds.
# Original paper reports ~11 h per fold × 5 folds = 55 h per seed; 3 seeds ≈ 165 h (~7 days).
# Eval uses the paper's canonical N=20 samples on ETH/UCY.
# Serial on one GPU.
set -euo pipefail

OUT="../runs"
CKPT="../runs/checkpoints"
LOG="$OUT/run_ethucy_multiseed_master.log"
mkdir -p "$OUT" "$CKPT"

SEEDS=(11 22 33)
SITES=(eth hotel univ zara1 zara2)

cd .
export PYTHONUNBUFFERED=1

for SEED in "${SEEDS[@]}"; do
  for SITE in "${SITES[@]}"; do
    TAG="ethucy_seed${SEED}_${SITE}_cdecnf-marg"
    CK="${CKPT}/${TAG}.pt"
    RESULT="$OUT/${TAG}.json"
    if [ -f "$RESULT" ]; then
      echo "===== SKIP  ${TAG} (result JSON already exists) =====" | tee -a "$LOG"
      continue
    fi
    EXTRA_ARGS=""
    if [ -f "$CK" ]; then
      echo "===== EVAL  ${TAG} (reusing checkpoint) =====" | tee -a "$LOG"
      EXTRA_ARGS="--load_only"
    else
      echo "===== START ${TAG} $(date -Iseconds) =====" | tee -a "$LOG"
    fi
    python -u run_experiment.py \
      --dataset EthUcy --encoder CDE --flow CNF --marginal 1 --seed "$SEED" \
      --site "$SITE" \
      --checkpoint "$CK" \
      $EXTRA_ARGS \
      --out_dir "$OUT" --tag "$TAG" \
      >> "$LOG" 2>&1
    echo "===== DONE  ${TAG} $(date -Iseconds) =====" | tee -a "$LOG"
  done
done
