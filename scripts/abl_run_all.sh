#!/usr/bin/env bash
# Ablation queue (sequential: < 6 GB free RAM, decisions R29). Each run trains with --resume, then the variant is
# evaluated immediately on the test grid. Order = priority with the cuts applied (A4, A5: seed 0 first; their seed-1
# runs only at the end if time is left).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/abl_queue.log; }
[ -f results/ablations/eval_ref.csv ] || { log "eval ref"; $PY scripts/abl_eval.py --variant ref >> logs/abl_eval.log 2>&1; }
for job in "a1 a1_nomask 0" "a1 a1_nomask 1" "a2 a2_noforced 0" "a2 a2_noforced 1" "a3 a3_unit 0" "a3 a3_unit 1" \
           "a4 a4_sum 0" "a5 a5_scratch 0" "a4 a4_sum 1" "a5 a5_scratch 1"; do
  set -- $job
  [ -f "STOP_ABLATIONS" ] && { log "stop file found"; break; }
  if [ -f "runs/abl_$1_seed$3/summary.json" ]; then log "skip $1 seed $3 (done)"; continue; fi
  log "train $1 seed $3 (free RAM $(powershell -c '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)') GB)"
  $PY -m wtdsws.train --config configs/ablations/$2.yaml --seed $3 --resume > logs/abl_train_$1_seed$3.log 2>&1
  log "done $1 seed $3: $(tail -n 1 logs/abl_train_$1_seed$3.log)"
  $PY scripts/abl_eval.py --variant $1 >> logs/abl_eval.log 2>&1
  log "evaluated $1"
done
log "queue finished"
