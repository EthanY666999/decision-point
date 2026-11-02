"""Normalize all probe outputs using production token-limit handling; preserve raw outputs."""
import json,sys,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patr_core import atomic_json,FrozenLakeAgent
manifest=ROOT/'runs/audit/state_manifest.json';mi=json.loads((manifest.with_name('state_manifest_identity.json')).read_text());assert hashlib.sha256(manifest.read_bytes()).hexdigest()==mi['sha256']
states=json.loads(manifest.read_text())['states'];parser=FrozenLakeAgent();models={};report={}
for path in sorted((ROOT/'runs/audit/state_probe').glob('*/summary.json')):
 rs=[json.loads(s) for s in (path.parent/'responses.jsonl').read_text().splitlines()];assert len(rs)==len(states)
 out=[]
 for row,state in zip(rs,states):
  assert row['id']==state['id'];_,parsed=parser._parse_model_response(row['response']);parsed=int(parsed)
  accepted=parsed in (1,2,3,4) and row['finish_reason']!='length'
  out.append(dict(id=state['id'],uid=state['task']['uid'],parsed_legal=parsed in (1,2,3,4),legal=accepted,optimal=accepted and parsed in state['optimal_actions'],parsed_action=parsed,train_geometry_seen=state['train_geometry_seen']))
 models[path.parent.name]=out
 def stats(rows):
  n=len(rows);legal=[r for r in rows if r['legal']]
  return dict(n=n,parsed_format_legal_pct=100*sum(r['parsed_legal'] for r in rows)/n if n else None,completed_legal_pct=100*len(legal)/n if n else None,legal_n=len(legal),optimal_given_completed_legal_pct=100*sum(r['optimal'] for r in legal)/len(legal) if legal else None,optimal_all_states_pct=100*sum(r['optimal'] for r in rows)/n if n else None)
 report[path.parent.name]=dict(all_states=stats(out),unseen_geometry=stats([r for r in out if not r['train_geometry_seen']]),cost=json.loads(path.read_text()))
comparisons={}
for left in models:
 for right in ['old_patr','sft','sft_grpo400','sft_patr400']:
  if right not in models or left==right:continue
  common=[(a,b) for a,b in zip(models[left],models[right]) if a['legal'] and b['legal']]
  comparisons[left+'_minus_'+right]=dict(common_legal_n=len(common),optimal_delta_pp=100*sum(a['optimal']-b['optimal'] for a,b in common)/len(common) if common else None)
atomic_json(ROOT/'runs/audit/state_probe_report.json',dict(models=report,paired_common_legal=comparisons,note='Primary accepted-action metrics exclude token-limit outputs, matching production apply_response. Raw per-model summaries use syntactic parsing; both are retained. Common-legal subsets may be empty or selected; no causal decision-improvement claim. Fixed-state contexts have no prior history.'))
print('Summarized',list(models))
