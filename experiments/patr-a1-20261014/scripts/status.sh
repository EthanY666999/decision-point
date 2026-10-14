#!/usr/bin/env bash
set -euo pipefail
cd /home/vipuser/patr-reproduction
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
.venv/bin/python scripts/report.py
cat RESULTS.md
