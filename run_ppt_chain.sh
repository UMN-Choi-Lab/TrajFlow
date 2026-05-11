#!/bin/bash
# Autonomous chain: wait for FloMo LORO to complete, then run PPT sanity,
# then PPT LORO 4 folds. Runs in background; tail the master log to follow.
set -e
cd .
LOG=../runs/run_ppt_chain_master.log

echo "===== CHAIN: waiting for FloMo LORO 4 JSONs =====" | tee -a "$LOG"
until [ "$(ls ../runs/loro_flomo_*.json 2>/dev/null | wc -l)" -ge 4 ]; do
  sleep 60
done
echo "===== CHAIN: FloMo LORO complete at $(date -Iseconds) =====" | tee -a "$LOG"

echo "===== CHAIN: starting PPT 75/25 sanity =====" | tee -a "$LOG"
python3 -u run_ppt_75_25_sanity.py >> "$LOG" 2>&1
RMSE=$(python3 -c "import json; print(json.load(open('../runs/ppt_75_25_sanity.json'))['rmse'])")
echo "===== CHAIN: PPT sanity RMSE=${RMSE} (target ~1.16) =====" | tee -a "$LOG"

echo "===== CHAIN: starting PPT LORO 4 folds =====" | tee -a "$LOG"
python3 -u run_ppt_loro.py >> "$LOG" 2>&1
echo "===== CHAIN: PPT LORO complete at $(date -Iseconds) =====" | tee -a "$LOG"
echo "===== CHAIN: ALL DONE =====" | tee -a "$LOG"
