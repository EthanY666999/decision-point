import ast,json,hashlib,copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from types import SimpleNamespace
P=Path(__file__).resolve().parent
tree=ast.parse((P/'upstream_core_algos.py').read_text())
chosen={'agg_loss','compute_policy_loss','compute_grpo_outcome_advantage'}
module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in chosen],type_ignores=[])
for n in module.body:n.decorator_list=[]
ns={'torch':torch,'np':np,'defaultdict':defaultdict,'verl_F':SimpleNamespace(masked_mean=lambda v,m:(v*m).sum()/m.sum())}
exec(compile(module,'verbatim_upstream_function_nodes','exec'),ns)
report={'tolerance':{'float64':1e-10,'float32':1e-6},'scope':'CPU extracted exact upstream functions; no Qwen update or GPU training','checks':[]}
for N in [8,24,256]:
 torch.manual_seed(42+N);L=13;old=torch.randn(N,L,dtype=torch.float64);cur=(old+torch.randn(N,L,dtype=torch.float64)*.4).requires_grad_();adv=torch.randn(N,L,dtype=torch.float64);mask=(torch.rand(N,L)>.25).double();nominal=256
 # Eight virtual ranks, local nominal minibatch32; gradient averaging is explicit.
 loss=0
 for c,o,a,m in zip(cur.chunk(8),old.chunk(8),adv.chunk(8),mask.chunk(8)):
  loss+=ns['compute_policy_loss'](o,c,a,m,cliprange_low=.2,cliprange_high=.28,loss_agg_mode='seq-mean-token-sum')[0]*(len(c)/32)/8
 ratio=(cur-old).exp();local=(-torch.minimum(ratio*adv,ratio.clamp(.8,1.28)*adv)*mask).sum()/(nominal*L)
 g=torch.autograd.grad(loss,cur,retain_graph=True)[0];g2=torch.autograd.grad(local,cur)[0]
 error=max(abs(float(loss-local)),float((g-g2).abs().max()));assert error<1e-10
 report['checks'].append({'name':'padded_8rank_loss_and_logprob_gradient','N':N,'width':L,'max_error':error,'pass':True})
 # Tiny parameterized model and one AdamW update, same frozen anchor/mask/adv.
 A=torch.nn.Linear(5,L,bias=False,dtype=torch.float64);B=copy.deepcopy(A);x=torch.randn(N,5,dtype=torch.float64)
 oa=torch.optim.AdamW(A.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01);ob=torch.optim.AdamW(B.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 logits=A(x);ref=logits.detach();a=adv
 loss=0
 for c,o,v,m in zip(logits.chunk(8),ref.chunk(8),a.chunk(8),mask.chunk(8)):
  loss+=ns['compute_policy_loss'](o,c,v,m,cliprange_low=.2,cliprange_high=.28,loss_agg_mode='seq-mean-token-sum')[0]*(len(c)/32)/8
 loss.backward();new=B(x);rat=(new-ref).exp();((-torch.minimum(rat*a,rat.clamp(.8,1.28)*a)*mask).sum()/(256*L)).backward()
 err=float((A.weight.grad-B.weight.grad).abs().max());assert err<1e-10
 oa.step();ob.step();upd=float((A.weight-B.weight).abs().max());assert upd<1e-10
 report['checks'].append({'name':'tiny_model_gradient_and_AdamW','N':N,'grad_error':err,'parameter_error':upd,'pass':True})
# Counterexample: accepted PaTR groups can have non-8 leaf counts, unlike GRPO.
rewards=torch.tensor([0,0,0,1,1,1,1,1,0,0,0,0],dtype=torch.float32).reshape(-1,1)
uids=np.array(['A']*6+['B']*6);mask=torch.ones_like(rewards)
full=ns['compute_grpo_outcome_advantage'](rewards.clone(),mask,uids)[0]
rounded=8
cut=ns['compute_grpo_outcome_advantage'](rewards[:rounded].clone(),mask[:rounded],uids[:rounded])[0]
err=float((full[:rounded]-cut).abs().max())
assert err>1.
report['checks'].append({'name':'whole_group_advantage_vs_literal_8rank_pre_advantage_truncation','input_groups':[6,6],'accepted_leaves':12,'upstream_kept':8,'upstream_dropped':4,'full_group_last_retained':full[6:8,0].tolist(),'literal_upstream_last_retained':cut[6:8,0].tolist(),'max_advantage_difference':err,'pass':False,'reason':'Exact pinned trainer drops tail to world-size multiple before computing advantages; whole-group PaTR adaptation gives different advantages. No upstream PaTR-specific patch exists in selected source.'})
report['checks'][-1]['classification']='expected_8rank_behavior_difference_not_single_card_failure'
report['checks'][-1]['blocking']=False
for N in [6,12,66,78,256,270]:
 torch.manual_seed(N);L=17
 old=torch.randn(N,L,dtype=torch.float64);cur=(old+.2*torch.randn_like(old)).requires_grad_();adv=torch.randn_like(old);mask=(torch.rand(N,L)>.3).double()
 errors=[]
 for start in range(0,N,256):
  c=cur[start:start+256];o=old[start:start+256];a=adv[start:start+256];m=mask[start:start+256]
  ref=ns['compute_policy_loss'](o,c,a,m,cliprange_low=.2,cliprange_high=.28,loss_agg_mode='seq-mean-token-sum')[0]*len(c)/256
  ratio=(c-o).exp();got=(-torch.minimum(ratio*a,ratio.clamp(.8,1.28)*a)*m).sum()/(256*L)
  g1=torch.autograd.grad(ref,cur,retain_graph=True)[0];g2=torch.autograd.grad(got,cur,retain_graph=True)[0]
  error=max(abs(float(ref-got)),float((g1-g2).abs().max()));assert error<1e-10;errors.append(error)
 report['checks'].append({'name':'single_card_preserves_all_rows_and_nominal_tail_scaling','N':N,'kept':N,'max_error':max(errors),'pass':True})
report['state']='R0_CPU_SUBSET_PASSED_REMAINING_ACCEPTANCE_PENDING' 
report['not_yet_accepted']=['full Qwen fixed-batch gradient/update','data/env/tokenization/resume strict verifier']
(P/'r0_equivalence_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
