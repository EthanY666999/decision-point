"""Isolated V4 engine infrastructure port; algorithm update is separate."""
import os
os.environ['VLLM_USE_V1']='0'
os.environ['VLLM_ATTENTION_BACKEND']='FLASH_ATTN'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='8'
os.environ['VLLM_NO_USAGE_STATS']='1'
os.environ['DO_NOT_TRACK']='1'
os.environ['WANDB_MODE']='offline'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


import json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from numerical import verify_qwen_weights, projected_checkpoint
from run_integrity import numerical_environment
ROOT=Path(__file__).resolve().parent
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
def build_models(cfg,seed):
    frozen_numerics=numerical_environment()
    backend=cfg.get('vllm_attention_backend','FLASH_ATTN')
    assert backend in {'FLASH_ATTN','XFORMERS'},'Unsupported inference attention backend'
    os.environ['VLLM_ATTENTION_BACKEND']=backend
    # Triton builds its Python extension on first use. Minimal host images may
    # omit Python.h; matching distro headers can be extracted without root.
    local_include=ROOT/'artifacts/build-deps/usr/include'
    if (local_include/'python3.12/Python.h').exists():
        include_paths=[str(local_include/'python3.12'),str(local_include)]
        if os.environ.get('C_INCLUDE_PATH'):include_paths.append(os.environ['C_INCLUDE_PATH'])
        os.environ['C_INCLUDE_PATH']=os.pathsep.join(include_paths)
    from vllm import LLM
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model_path=str(ROOT/cfg['model'])
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    actor=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float32,
                   attn_implementation='sdpa',local_files_only=True).to('cuda')
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.config.use_cache=False
    actor._project_forward_bf16=cfg.get('project_forward_bf16',False)
    if actor._project_forward_bf16:
        actor.model._gradient_checkpointing_func=projected_checkpoint
    ref=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
                   attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    ref.requires_grad_(False)
    engine=LLM(model=model_path,tokenizer=model_path,dtype='bfloat16',
               tensor_parallel_size=1,distributed_executor_backend='uni',
               gpu_memory_utilization=cfg['vllm_memory_fraction'],
               max_model_len=cfg['max_context_tokens'],max_num_seqs=cfg['vllm_max_sequences'],
               max_num_batched_tokens=4096,enable_chunked_prefill=True,
               enable_prefix_caching=True,disable_async_output_proc=True,
               seed=seed,enforce_eager=cfg.get('vllm_enforce_eager',False),disable_log_stats=True)
    if numerical_environment()!=frozen_numerics:
        raise RuntimeError('Model/backend initialization changed frozen numerical controls')
    return actor,ref,engine,tokenizer

def inference_cache_boundary(engine, phase, *, reset=False):
    """Fail closed at actual vLLM V0/0.9.2 idle boundaries before cache changes."""
    backend=engine.llm_engine
    state=dict(phase=phase,unfinished_requests=backend.get_num_unfinished_requests(),
               has_unfinished_requests=backend.has_unfinished_requests(),schedulers=[])
    for scheduler in backend.scheduler:
        manager=scheduler.block_manager
        queues={name:len(getattr(scheduler,name)) for name in
                ('waiting','running','swapped','_async_stopped')}
        queues.update(block_tables=len(manager.block_tables),
                      cross_block_tables=len(manager.cross_block_tables))
        used={str(device):allocator.get_num_total_blocks()-allocator.get_num_free_blocks()
              for device,allocator in manager.block_allocator._allocators.items()}
        state['schedulers'].append(dict(queues=queues,used_blocks=used))
    busy=state['unfinished_requests'] or state['has_unfinished_requests'] or any(
        any(item['queues'].values()) or any(item['used_blocks'].values()) for item in state['schedulers'])
    if busy:
        raise RuntimeError('Inference boundary is not idle: '+json.dumps(state,sort_keys=True))
    state['prefix_cache_reset']=backend.reset_prefix_cache() if reset else None
    if reset and state['prefix_cache_reset'] is not True:
        raise RuntimeError('Prefix-cache reset failed: '+json.dumps(state,sort_keys=True))
    if reset:
        print(json.dumps({'INFERENCE_CACHE_BOUNDARY':state}),flush=True)
    return state

def sync_weights(actor,engine):
    # V0/UniProc colocates the inference model. No network service or extra GPU.
    inference_cache_boundary(engine,'before_weight_sync')
    vm=engine.llm_engine.model_executor.driver_worker.model_runner.model
    with torch.no_grad():
        loaded=vm.load_weights((name,p.detach().to(torch.bfloat16)) for name,p in actor.named_parameters())
        verify_qwen_weights(actor,engine)
    inference_cache_boundary(engine,'after_weight_sync',reset=True)
    torch.cuda.synchronize()
    return len(loaded) if loaded is not None else None

def selected_hidden(model,ids,attention,mask):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if getattr(model,'_project_forward_bf16',False):
            params={name:p.to(torch.bfloat16) for name,p in model.model.named_parameters()}
            hidden=functional_call(model.model,params,(),dict(input_ids=ids,attention_mask=attention,use_cache=False)).last_hidden_state
        else:
            hidden=model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        # Preserve full-position head matmul before applying assistant mask.
        # Compacting first changes BF16 backward reduction numerics.
        return hidden[:, :-1].reshape(-1, hidden.shape[-1]), ids[:, 1:].reshape(-1)

def policy_logits(model,hidden):
    logits=model.lm_head(hidden).float()
    allowed=getattr(model,'_allowed_token_ids',None)
    if allowed is not None:
        keep=torch.zeros(logits.shape[-1],device=logits.device,dtype=torch.bool)
        keep[allowed]=True
        # Finite floor gives exactly zero probability and avoids -inf - -inf in KL.
        logits=logits.masked_fill(~keep,torch.finfo(logits.dtype).min)
    return logits

def selected_logprobs(model,ids,attention,mask,gradient=False):
    """Only allocate vocabulary logits for assistant targets, in bounded chunks."""
    with torch.autocast('cuda',dtype=torch.bfloat16):
        h,labels=selected_hidden(model,ids,attention,mask)
        parts=[]
        def head_logprob(x,y):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=policy_logits(model,x)
            return -F.cross_entropy(logits,y,reduction='none')
        for lo in range(0,len(labels),512):
            hs,ys=h[lo:lo+512],labels[lo:lo+512]
            lp=checkpoint(head_logprob,hs,ys,use_reentrant=False) if gradient else head_logprob(hs,ys)
            parts.append(lp)
        return torch.cat(parts)[mask[:, 1:].reshape(-1).bool()]
