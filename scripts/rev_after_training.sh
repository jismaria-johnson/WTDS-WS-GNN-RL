#!/bin/bash
# Revision: use each new checkpoint as soon as its training ends; timing-sensitive runs only after all training.
cd "/d/project s7/wtds-ws-gnn-rl"
PY=.venv/Scripts/python.exe
L=logs/rev_after_training.log
for s in 2 3 4; do
  until grep -q "=== seed $s exit" logs/train_seeds_234.log 2>/dev/null; do sleep 20; done
  grep -q "=== seed $s exit 0" logs/train_seeds_234.log || { echo "seed $s training FAILED" >> $L; continue; }
  echo "seed $s ready $(date -Iseconds)" >> $L
  $PY -u scripts/rev_eval_grid.py --part seeds --seeds $s >> logs/rev_grid_seeds.log 2>&1
  until grep -q "^done" logs/rev_alpha_ours01.log 2>/dev/null; do sleep 20; done
  $PY -u scripts/rev_alpha_sweep.py --part ours --seeds $s >> logs/rev_alpha_ours234.log 2>&1
  echo "seed $s evaluated $(date -Iseconds)" >> $L
done
echo "timing runs start $(date -Iseconds)" >> $L
$PY -u scripts/rev_inference_time.py --device cuda > logs/rev_infer_cuda.log 2>&1
CUDA_VISIBLE_DEVICES= $PY -u scripts/rev_inference_time.py --device cpu > logs/rev_infer_cpu.log 2>&1
until grep -q "^done" logs/rev_alpha_cpu.log 2>/dev/null; do sleep 20; done   # no CP-SAT load during GPU timing
$PY -u scripts/rev_equal_time.py --group ours --seeds 0,1,2,3,4 > logs/rev_equal_time_ours.log 2>&1
$PY -u scripts/rev_coverage.py --part block --seeds 0,1,2,3,4 > logs/rev_block5.log 2>&1
echo "all done $(date -Iseconds)" >> $L
