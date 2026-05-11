#!/bin/bash
# R1#4 statistical robustness: 5-seed CDE-CNF marginal on the default inD single-session
# 75/25 random split (site 08). 100 eval samples.
# Serial on one GPU. ~12 h per seed, ~60 h total.
set -euo pipefail

OUT="../runs"
CKPT="../runs/checkpoints"
LOG="$OUT/run_r1q4_multiseed_master.log"
mkdir -p "$OUT" "$CKPT"

SEEDS=(11 22 33 44 55)

cd .
export PYTHONUNBUFFERED=1

for SEED in "${SEEDS[@]}"; do
  TAG="r1q4_seed${SEED}_cdecnf-marg_ind08"
  CK="${CKPT}/${TAG}.pt"
  RESULT="$OUT/${TAG}.json"
  if [ -f "$RESULT" ]; then
    echo "===== SKIP  ${TAG} (result JSON already exists) =====" | tee -a "$LOG"
    continue
  fi
  EXTRA_ARGS=""
  if [ -f "$CK" ]; then
    echo "===== EVAL  ${TAG} (reusing checkpoint, skipping training) =====" | tee -a "$LOG"
    EXTRA_ARGS="--load_only"
  else
    echo "===== START ${TAG} $(date -Iseconds) =====" | tee -a "$LOG"
  fi
  python -u run_experiment.py \
    --dataset InD --encoder CDE --flow CNF --marginal 1 --seed "$SEED" \
    --checkpoint "$CK" \
    --eval_samples_override 100 \
    $EXTRA_ARGS \
    --out_dir "$OUT" --tag "$TAG" \
    >> "$LOG" 2>&1
  echo "===== DONE  ${TAG} $(date -Iseconds) =====" | tee -a "$LOG"
done
