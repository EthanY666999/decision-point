# Framework-local external baselines

All comparisons use the same FrozenLake environment, model start, four trajectories per task and evaluation protocol. They are framework adaptations, not numbers copied from official benchmarks.

| Component | Paper / official implementation | Here | Source |
|---|---|---|---|
| StarPO-S reward-std retention | Paper studies 50% / 75%; current official base uses top_p=1.0 (no filtering); no unique default established | top 50%, explicitly chosen by user 2026-11-07; deterministic task-ID tie break | [RAGEN sec 4.2](https://arxiv.org/html/2504.20073v1#S4.SS2); [official filtering](https://github.com/RAGEN-AI/RAGEN/blob/9d021fc782a4a9e8e047f13c1cd706b32f16d8f2/ragen/trainer/rollout_filter.py) |
| StarPO-S KL | removal | coefficient 0 | RAGEN sec 4.2 |
| StarPO-S clipping | lower .2, upper .28 | .2 / .28 | RAGEN sec 4.2 |
| GiGPO gamma / omega | .95 / 1 | .95 / 1 | [GiGPO appendix D.1](https://arxiv.org/html/2505.10978v1#A4.SS1) |
| GiGPO state groups | identical states within an episode group | task + exact map + player position; singletons get zero step advantage | [GiGPO sec 4](https://arxiv.org/html/2505.10978v1#S4) |

GiGPO adds discounted-return normalized step advantage to the unchanged GRPO episode advantage. Turn rows keep the original trajectory token denominator, so splitting turns does not reweight trajectories. omega=0 calls the exact frozen GRPO weighting function. No entropy bonus is added: the authorized adaptation specifies precisely three StarPO-S changes on top of GRPO4.
