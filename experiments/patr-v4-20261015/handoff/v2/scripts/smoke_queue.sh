#!/usr/bin/env bash
set -euo pipefail
cd /home/vipuser/patr-reproduction
for method in grpo patr tree_random dapo arpo; do
  .venv/bin/python -u train.py --method "$method" --smoke > "logs/smoke_async_${method}.log" 2>&1
done
.venv/bin/python -u train.py --method grpo --stop-after-step 1 > logs/calibrate_async_grpo.log 2>&1
