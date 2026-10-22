#!/bin/bash
# Revision step 0: fine-tune seeds 2, 3, 4 one after another (low RAM). Each writes checkpoints/ws_seed<s>/best.pt.
cd "/d/project s7/wtds-ws-gnn-rl"
for s in 2 3 4; do
  echo "=== seed $s start $(date -Iseconds)" >> logs/train_seeds_234.log
  .venv/Scripts/python.exe -u -m wtdsws.train --config configs/ws_finetune_seed$s.yaml --seed $s > logs/train_seed$s.log 2>&1
  echo "=== seed $s exit $? $(date -Iseconds)" >> logs/train_seeds_234.log
done
