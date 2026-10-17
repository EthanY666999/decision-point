新增500题补充评估，只评估、不训练。
先读 REPORT.txt、results.csv、analysis.json 与 paired_intervals.png。
原100题仍是主结果：GRPO 61/62，PaTR 64.5/71，Random 64.75/68（Success/Pass@4）。本目录不替换主结果，也不将新增500题合并为600题主结果。
protocol.json、execution_identity.json、tasks.json、maps.json 在模型评估前固定。全部500个任务UID与原数据零重叠；地图几何可能重复，详见data_audit.json。
每个模型子目录包含5个固定100题批次、总计2000条原始轨迹、原始采样assistant IDs、分批与总指标、按题结果。三个模型合计6000条轨迹。
源码与模型均与已完成的V1实验一致，模型SHA与原主测试SHA见protocol.json。source/保留所用冻结代码与配置；模型权重不上传GitHub。
主bootstrap按题配对、50000次；另报告按地图几何成组重采样敏感性分析。两者均不包含训练seed之间的不确定性，不因区间包含0而宣称等效。
