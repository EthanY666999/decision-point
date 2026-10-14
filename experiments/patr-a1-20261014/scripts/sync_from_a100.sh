#!/usr/bin/env bash
# Run on the local Mac. Reuse an authenticated SSH connection during SSH throttling.
set -euo pipefail
patr_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patr_workspace="$(cd "$patr_root/../.." && pwd)"
mkdir -p "$patr_workspace/work"
patr_control_path="$patr_workspace/work/patr-a100.sock"
patr_transport="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPersist=3600 -o 'ControlPath=$patr_control_path'"

for patr_attempt in 1 2 3; do
    if rsync -az --stats \
        --exclude='.venv/' --exclude='.git/' --exclude='__pycache__/' \
        --exclude='.pytest_cache/' --exclude='*.pyc' --exclude='artifacts/Qwen*/' \
        --exclude='artifacts/build-deps/' \
        --exclude='checkpoint.pt' --exclude='checkpoint.tmp' --exclude='final_model/' \
        --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' \
        -e "$patr_transport" \
        galaxy-a100:/home/vipuser/patr-reproduction/ "$patr_root/"; then
        exit 0
    fi
    if (( patr_attempt < 3 )); then
        printf 'Sync connection failed; retry %s/3 after a short backoff.\n' "$((patr_attempt + 1))" >&2
        sleep "$((patr_attempt * 5))"
    fi
done
exit 1
