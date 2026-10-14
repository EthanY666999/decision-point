"""Explicit GPU audit, run separately from routine unit tests."""
import json
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from train import selected_logprobs,pack
from patr_core import ROOT,Branch


def test_assistant_logprobs_and_gradients():
    path=ROOT/'artifacts/Qwen2.5-0.5B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    ids=torch.tensor([tokenizer.encode('The correct next action is Right.')],device='cuda')
    attention=torch.ones_like(ids);mask=torch.ones_like(ids);mask[:,:3]=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(ids,attention_mask=attention).logits
        full=torch.log_softmax(logits[:,:-1].float(),-1).gather(-1,ids[:,1:,None]).squeeze(-1)[mask[:,1:].bool()]
    selective=selected_logprobs(model,ids,attention,mask,True)
    torch.testing.assert_close(selective,full,atol=1e-4,rtol=1e-4)
    (-selective.mean()).backward()
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0
    print('SELECTED_LOGPROBS_AND_GRADIENTS_PASS')
