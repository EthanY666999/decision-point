SFT、RL、评估三者的上下文都保留历史轮的空 think 块，与模板的规范渲染不同，这是有意为之的一致性约定。

SFT uses the shared patr_core.chat_ids / turn_suffix helpers and supervises only action tokens plus EOS. The Qwen2.5 sequence is unchanged through its final EOS; only the old untrained final newline is omitted.

2026-11-07 用户补充UCF协议：若本批合法候选的正分数支持不足以达到目标2，则按可达上限选点；零分数保留rho_min，排除格式不匹配候选；记录reachable_max、expected_forks、capacity_limited。该类冒烟的1–3实际分叉下界不再是可满足条件，单独标注原门禁例外并校验期望预算等于可达上限，不伪称通过原始预算条件。
