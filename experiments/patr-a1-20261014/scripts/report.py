import csv,json,datetime,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patr_core import ROOT,atomic_json
METHODS=['grpo','dapo','arpo','tree_random','patr']
PAPER={'grpo':[66.5,67.0,513,3.4],'dapo':[71.,71.,562,4.2],'arpo':[69.7,72.,377,3.3],'tree_random':[56.3,63.,400,3.9],'patr':[75.8,78.,344,3.6]}

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def records(path):
    if not path.exists():return []
    # Resume may have retried steps since the last checkpoint: last record wins.
    result={}
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line);result[r['step']]=r
        except (json.JSONDecodeError,KeyError):continue
    return [result[k] for k in sorted(result)]


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows=[];curves={};smoke=[]
    for method in METHODS:
        path=ROOT/'runs/formal'/f'{method}_seed42'
        status=read(path/'status.json',{})
        history=records(path/'train_metrics.jsonl');curves[method]=history
        incidents=[read(p,{}) for p in (path/'recovery_history').glob('*/incident.json')]
        row=dict(method=method,status=status.get('status','pending'),steps=max(status.get('step',0),history[-1]['step'] if history else 0),
                 success_rate_pct=None,pass_at_4_pct=None,response_length=None,turns=None,
                 generated_train_tokens=sum(r.get('generated_tokens',0) for r in history),
                 recovery_discarded_generated_tokens=sum(r.get('rolled_back_generated_tokens',0) for r in incidents),
                 recovery_discarded_step_hours=sum(r.get('rolled_back_step_seconds',0) for r in incidents)/3600,
                 measured_train_hours=(history[-1].get('elapsed_seconds',0)/3600 if history else 0),
                 median_last10_step_seconds=statistics.median([r['step_seconds'] for r in history[-10:]]) if history else None)
        if row['status']=='complete':
            final=read(path/'final_test.json',{})
            row.update({k:final.get(k) for k in ('success_rate_pct','pass_at_4_pct','response_length','turns')})
        rows.append(row)
        s=read(ROOT/'runs/smoke'/f'{method}_seed42'/'status.json',{})
        smoke.append((method,s.get('status','pending')))
    output=ROOT/'reports';output.mkdir(exist_ok=True)
    atomic_json(output/'summary.json',dict(updated_at=stamp,formal=rows,smoke=smoke))
    with (output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fmt=lambda v:'—' if v is None else (f'{v:.2f}' if isinstance(v,float) else str(v))
    lines=['# 本轮实测结果', '',f'更新时间（UTC）：{stamp}', '',
      '复现类型：paper-based reimplementation；完整假设与差异见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md)。', '',
      '## 正式训练', '', '| Method | 状态 | 步数 / 400 | Success % | Pass@4 % | RLen | Turns | 已记训练小时 | 最近每步秒 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join(fmt(r[k]) for k in ('method','status','steps','success_rate_pct','pass_at_4_pct','response_length','turns','measured_train_hours','median_last10_step_seconds'))+' |')
    lines+=['','未完成 400 步时不填最终测试指标。表内训练小时不含首次模型加载、首次验证或回滚后废弃的计算；总墙钟成本另见队列日志。', '']
    for r in rows:
        if r['recovery_discarded_step_hours']:
            lines.append(f"{r['method']} 另有 {r['recovery_discarded_step_hours']:.3f} 小时、{r['recovery_discarded_generated_tokens']} 个生成 token 的已完成迭代因检查点回滚而废弃。该开销必须计入复现成本，且尚不含触发故障的未完成迭代。证据保留在 `recovery_history/`。")
    completed=[r for r in rows if r['status']=='complete']
    if completed:
        lines+=['','## 已完成方法的最终动作质量','',
                '| Method | 无效动作 % | 成功轨迹 / 400 | 失败或截断分类 |',
                '|---|---:|---:|---|']
        for r in completed:
            final=read(ROOT/'runs/formal'/f"{r['method']}_seed42"/'final_test.json',{})
            invalid=final.get('invalid_action_fraction')
            terminals=final.get('terminal_counts',{})
            failure=', '.join(f'{k}: {v}' for k,v in terminals.items() if k!='success') or '—'
            lines.append(f"| {r['method']} | {fmt(100*invalid if invalid is not None else None)} | {terminals.get('success',0)} | {failure} |")
        lines+=['','无效动作比例以实际执行动作总数为分母；一条最终成功的轨迹也可能包含无效动作。GRPO 后期出现 Goal 等非法指令，PaTR 后期出现代码块结束符不完整；原始解析器与奖励保持不变。训练行为及格式审计见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md) 和 `reports/*action_format_audit.json`。验证集成绩不替代本表的最终测试。']
    audit=read(output/'evaluation_audit.json',{})
    if audit.get('all_five_verified') and all(r['status']=='complete' for r in rows):
        lines+=['','## 最终核验与论文对照','',
                '五种方法均完成 400 步；每种方法的最终 100 题 × 4 条轨迹均已重新统计并重放环境动作。详细证据见 `reports/evaluation_audit.json`。', '',
                '| Method | 论文 Success % | 实测 Success % | 差值 pp | 论文 Pass@4 % | 实测 Pass@4 % | 差值 pp | 论文 / 实测 RLen | 论文 / 实测 Turns |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            ps,pp,pl,pt=PAPER[r['method']]
            lines.append(f"| {r['method']} | {ps:.2f} | {r['success_rate_pct']:.2f} | {r['success_rate_pct']-ps:+.2f} | {pp:.2f} | {r['pass_at_4_pct']:.2f} | {r['pass_at_4_pct']-pp:+.2f} | {pl} / {r['response_length']:.2f} | {pt:.1f} / {r['turns']:.2f} |")
        lines+=['', '论文参照：[PaTR 表 1](https://arxiv.org/html/2607.15610v1#S5.T1)。差值按实测减论文计算，不代表统计显著性。', '',
                '本轮为单 seed 的论文方法重实现。作者未确认的 scorer、展开 top-k、数据/优化器/解码细节，以及单卡训练框架差异，逐项见 [REPRODUCTION_AUDIT.md](REPRODUCTION_AUDIT.md)。不依据最终测试调参或选择检查点。', '',
                '工程中断、恢复执行模式及重复计算见 `logs/engineering_recovery.jsonl`、`reports/cuda_incident_20261013.json` 和各 run 的 `recovery_history/`。已记录训练小时与废弃迭代开销分别保留在 `reports/summary.csv`；未完成故障迭代、首次加载/验证和空闲时间另计，不将这些值冒充完整租赁计费。']
    lines+=['',
            '## 链路烟测', '', '| Method | 状态 |','|---|---|']
    lines += [f'| {m} | {s} |' for m,s in smoke]
    lines+=['','烟测为 1 步小批次，仅检查真实训练链路；不可用于方法效果排序。首次 GRPO 烟测曾评估 4 个 test 任务，后续烟测改用 validation 任务；未以该结果选择算法超参。', '',
            '## 可审计文件','',
            '- `reports/summary.csv`：可直接用于表格。',
            '- `runs/formal/<method>_seed42/train_metrics.jsonl`：逐步真实采样、训练、成本。',
            '- `runs/formal/<method>_seed42/validation_*.json`：独立验证曲线。',
            '- `runs/formal/<method>_seed42/final_test.json`：最终 100 × 4 测试。',
            '- `runs/formal/<method>_seed42/rollouts_*.jsonl.gz`：原始轨迹、mask、采样 logprob 与动作。',
            '- `artifacts/map_audit.json`：地图重复与最短路审计。',
            '- `logs/queue.log`：队列执行和错误。', '',
            '已有风险：上游生成器允许最短路为 11 的少数地图，但交互上限为 10；小地图存在跨 split 的重复几何结构。这两点保留并报告，未根据测试表现修改数据。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if any(curves.values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for m,rs in curves.items():
            if not rs:continue
            axs[0].plot([r['step'] for r in rs],[r.get('accepted_success_rate',r['success_rate']) for r in rs],label=m,alpha=.7)
            vals=[]
            for p in sorted((ROOT/'runs/formal'/f'{m}_seed42').glob('validation_*.json')):
                j=read(p,{})
                if j: vals.append((int(p.stem.split('_')[-1]),j['success_rate_pct']))
            if vals:axs[1].plot(*zip(*vals),label=m,marker='o',markersize=3)
        axs[0].set(title='Training group success (DAPO filtered)',xlabel='Training iteration',ylabel='Fraction')
        axs[1].set(title='Independent validation',xlabel='Training iteration',ylabel='Success (%)')
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'training.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
