from pathlib import Path
import json,hashlib,shutil,tarfile,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/vipuser/patr-v1-tree-20261016');OUT=ROOT/'reports/eval500-20261017'
a=json.loads((OUT/'analysis.json').read_text());assert a['training_updates']==0 and a['replay_verified_trajectories']==6000
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
pairs=['patr minus tree_random','patr minus grpo','tree_random minus grpo'];labels=['PaTR - Random','PaTR - GRPO','Random - GRPO']
for ax,metric,title in zip(axes,['success_pp','pass4_pp'],['Success difference','Pass@4 difference']):
 for i,p in enumerate(pairs):
  d=a['comparisons'][p][metric];value=d['difference'];lo,hi=d['ci95_task_bootstrap'];glo,ghi=d['ci95_geometry_cluster_sensitivity']
  ax.errorbar(value,i+.08,xerr=[[value-lo],[hi-value]],fmt='o',color='#176b8c',capsize=4,label='Paired task bootstrap' if i==0 else None)
  ax.errorbar(value,i-.08,xerr=[[value-glo],[ghi-value]],fmt='s',ms=4,color='#ad753b',capsize=3,label='Geometry-cluster sensitivity' if i==0 else None)
 ax.axvline(0,color='#777777',lw=1,ls='--');ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set(title=title,xlabel='Percentage points; 95% intervals');ax.grid(axis='x',alpha=.15)
fig.suptitle('Supplemental 500 tasks | 4 samples/task | existing seed42 models | no training')
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
fig.subplots_adjust(left=.13,right=.98,top=.84,bottom=.22,wspace=.42)
fig.savefig(OUT/'paired_intervals.png',dpi=180);fig.savefig(OUT/'paired_intervals.svg');plt.close(fig)
index='''新增500题补充评估，只评估、不训练。
先读 REPORT.txt、results.csv、analysis.json 与 paired_intervals.png。
原100题仍是主结果：GRPO 61/62，PaTR 64.5/71，Random 64.75/68（Success/Pass@4）。本目录不替换主结果，也不将新增500题合并为600题主结果。
protocol.json、execution_identity.json、tasks.json、maps.json 在模型评估前固定。全部500个任务UID与原数据零重叠；地图几何可能重复，详见data_audit.json。
每个模型子目录包含5个固定100题批次、总计2000条原始轨迹、原始采样assistant IDs、分批与总指标、按题结果。三个模型合计6000条轨迹。
源码与模型均与已完成的V1实验一致，模型SHA与原主测试SHA见protocol.json。source/保留所用冻结代码与配置；模型权重不上传GitHub。
主bootstrap按题配对、50000次；另报告按地图几何成组重采样敏感性分析。两者均不包含训练seed之间的不确定性，不因区间包含0而宣称等效。
'''
(OUT/'README.txt').write_text(index)
original=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.lock','.tmp','.pyc'] and '__pycache__' not in p.parts]
stage=OUT/'archive';stage.mkdir(exist_ok=False)
for p in original:
 q=stage/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for n in ['train.py','patr_core.py','configs/frozenlake_05b.json','artifacts/runtime_override.json']:
 q=stage/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q)
for p in (ROOT/'vendor/rllm_source').rglob('*.py'):
 q=stage/'source/vendor/rllm_source'/p.relative_to(ROOT/'vendor/rllm_source');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=[dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
(stage/'MANIFEST.json').write_text(json.dumps(dict(created_unix=time.time(),files=manifest,training_updates=0),indent=2)+'\n')
tar=OUT/'patr-supplemental500-20261017.tar.gz'
with tarfile.open(tar,'w:gz') as t:t.add(stage,arcname='patr-supplemental500-20261017')
receipt=dict(path=str(tar),bytes=tar.stat().st_size,sha256=sha(tar),file_count=len(manifest)+1)
(OUT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
