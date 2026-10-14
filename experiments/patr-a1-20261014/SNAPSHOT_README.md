# A1 experiment review snapshot

This is a lightweight review snapshot from the A100, not a full machine backup. It includes source, configuration, final metrics, validation metrics, scalar training histories, key logs and audits. The methods are paper-based reimplementations with unverified author details; see REPRODUCTION_AUDIT.md.

GRPO: success 61%, Pass@4 62%. PaTR: 60%, 61%. Tree-Random: 43%, 45%. DAPO was stopped after 5/400 iterations while sampling iteration 6; recoverable checkpoint is iteration 1. ARPO did not start. Validation numbers are separate from final test numbers.

All model weights, optimizer checkpoints, full rollout traces, dependencies and the complete archive remain on the rented A100. They are intentionally not uploaded here. The original project is /home/vipuser/patr-reproduction and the complete archive is /home/vipuser/patr-archives/patr-user-stop-20261014.tar.gz (SHA-256: 2a7670bf862c4b53eaf7d15679abfd2febc834ec05ff86bc6f50d9883e474812).

No experiment algorithm was changed for this snapshot. The manifest verifies the files copied into this repository. Existing scripts describe historical training and recovery procedures, not instructions to restart the stopped queue.
