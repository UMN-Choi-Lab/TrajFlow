#!/bin/bash
# Waits for the currently running run_loro.sh driver (bash PID) to exit,
# then runs (i) R1#4 inD multi-seed and (ii) ETH/UCY multi-seed in sequence.
# Safe to run at any time: if LORO is not running, it starts the downstream jobs immediately.
set -euo pipefail

LORO_BASH_PID="${1:-501749}"
LOG="../runs/run_post_loro_queue_master.log"
echo "===== QUEUE WAITING for LORO PID $LORO_BASH_PID  $(date -Iseconds) =====" | tee -a "$LOG"

# Poll until LORO driver exits (or is already gone). Check every 60s.
while kill -0 "$LORO_BASH_PID" 2>/dev/null; do
  sleep 60
done
echo "===== LORO DONE, starting multi-seed chain  $(date -Iseconds) =====" | tee -a "$LOG"

cd .

echo "===== QUEUE: R1#4 inD 5-seed  $(date -Iseconds) =====" | tee -a "$LOG"
bash run_r1q4_multiseed_ind.sh 2>&1 | tee -a "$LOG" || true
echo "===== QUEUE: R1#4 inD DONE  $(date -Iseconds) =====" | tee -a "$LOG"

echo "===== QUEUE: ETH/UCY 3-seed  $(date -Iseconds) =====" | tee -a "$LOG"
bash run_ethucy_multiseed.sh 2>&1 | tee -a "$LOG" || true
echo "===== QUEUE: ETH/UCY DONE  $(date -Iseconds) =====" | tee -a "$LOG"

echo "===== QUEUE ALL DONE  $(date -Iseconds) =====" | tee -a "$LOG"
