import os
os.environ['C_INCLUDE_PATH'] = '/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include/python3.12:/home/vipuser/patr-reproduction/artifacts/build-deps/usr/include'
from engine_v4 import build_models, sync_weights, inference_cache_boundary
from numerical import verify_qwen_weights
from patr_core import collect, evaluation_metrics
from pathlib import Path
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT/'config_candidate.json').read_text())
start = time.time()
actor, ref, engine, tokenizer = build_models(cfg, 42)
sync_weights(actor, engine)
verified = verify_qwen_weights(actor, engine)
vm = engine.llm_engine.model_executor.driver_worker.model_runner.model
with torch.no_grad():
    vm.model.layers[0].self_attn.qkv_proj.weight[0, 0] += 1
detected = False
try:
    verify_qwen_weights(actor, engine)
except AssertionError:
    detected = True
assert detected, 'Verifier failed to detect intentional inference weight corruption'
sync_weights(actor, engine)
tasks = json.loads((ROOT/'artifacts/data_v4.json').read_text())['train'][:1]
branches, stats, events = collect(tasks, engine, tokenizer, cfg, 'grpo', np.random.default_rng(42), evaluation=True)
assert len(branches) == 4
assert all(len(b.ids) == len(b.mask) == len(b.old_logprobs) for b in branches)
assert all(len(b.budget.tokens) <= 10240 for b in branches)
inference_cache_boundary(engine, 'after_acceptance_rollout', reset=True)
report = dict(passed=True, scope='Actual vLLM synchronization with deliberate corruption negative control and one training-task four-trajectory evaluation fixture; not R1 or formal evaluation', strict_verifier=verified, corruption_detected=detected, rollout=stats, metrics=evaluation_metrics(branches, tasks), seconds=time.time()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
(ROOT/'reports/engine_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
