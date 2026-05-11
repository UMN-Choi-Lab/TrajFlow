#!/bin/bash
# Chain: wait for CDE-CNF joint inD 75/25 training to land,
# then run the joint-vs-marginal coherence evaluation.
set -e
cd .
LOG=../runs/run_r1q2_chain.log

echo "===== R1#2 CHAIN: waiting for joint training JSON =====" | tee -a "$LOG"
until [ -f ../runs/r1q2_cdecnf_joint_ind75_25.json ]; do
  sleep 180
done
echo "===== R1#2 CHAIN: joint training complete at $(date -Iseconds) =====" | tee -a "$LOG"

echo "===== R1#2 CHAIN: running joint vs marginal coherence eval =====" | tee -a "$LOG"
python3 -u run_joint_vs_marginal_coherence.py >> "$LOG" 2>&1
echo "===== R1#2 CHAIN: ALL DONE at $(date -Iseconds) =====" | tee -a "$LOG"
