#!/bin/bash
# Driver: run the 4 leave-one-recording-out folds for inD CDE-CNF marginal + GRU-DNF.
# - Eval samples reduced from 1000 to 100 to keep wall-clock feasible on one GPU.
# - If a checkpoint already exists for a fold, skip training and evaluate only.
# - Uses python -u + direct file redirect so epoch prints flush live.
set -euo pipefail

OUT="../runs"
CKPT="../runs/checkpoints"
LOG="$OUT/run_loro_master.log"
mkdir -p "$OUT" "$CKPT"

FOLDS=(
  "07,09,10|08"
  "08,09,10|07"
  "07,08,10|09"
  "07,08,09|10"
)

cd .
export PYTHONUNBUFFERED=1

for ENCODER in CDE GRU; do
  FLOW=$([ "$ENCODER" = "CDE" ] && echo "CNF" || echo "DNF")
  for FOLD in "${FOLDS[@]}"; do
    TRAIN_SITES="${FOLD%|*}"
    TEST_SITE="${FOLD#*|}"
    TAG="loro_${ENCODER,,}-${FLOW,,}_train-${TRAIN_SITES//,/-}_test-${TEST_SITE}"
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
      --dataset InD --encoder "$ENCODER" --flow "$FLOW" --marginal 1 --seed 0 \
      --ind_train_sites "$TRAIN_SITES" --ind_test_sites "$TEST_SITE" \
      --checkpoint "$CK" \
      --eval_samples_override 100 \
      $EXTRA_ARGS \
      --out_dir "$OUT" --tag "$TAG" \
      >> "$LOG" 2>&1
    echo "===== DONE  ${TAG} $(date -Iseconds) =====" | tee -a "$LOG"
  done
done
