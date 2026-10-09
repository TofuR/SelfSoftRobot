# 建模基准实现验证记录

验证日期：2026-09-12。研究协议见[modeling_experiments.md](modeling_experiments.md)。本记录报告代码验证范围，论文性能数据保留待填。

## 实际执行

| 项目 | 结果 |
|---|---|
| 数据恢复与分组 | 核对7条完整5 Hz原始序列、17440帧动作/骨架/真实mask；fold0为3774帧train、1286帧val、12380帧test |
| 短训练 | 21种配置 × 2个种子 = 42个完成run；神经模型2 epoch，闭式拟合/静态参考1次评价 |
| 目标帧预算 | 每个run选32个训练窗口目标；HOV先验也只拟合相同32帧；历史20、batch16、先验优化2步 |
| 验证集评价 | 每个run计8个固定验证帧；骨架、末端、真实mask，以及标注骨架管状投影诊断全部完成 |
| 选模 | 每个run使用验证节点欧氏误差选择checkpoint，恢复后完成评价；42个run的评价脚本版本一致 |
| 重复实验汇总 | 输出逐序列CSV、均值/种子标准差CSV、240项配对模型/指标诊断；仅1条验证序列，全部为 `inconclusive`，`inference_eligible=false` |
| 单元与集成测试 | **40项通过**，包含全部21种模型的前向、梯度与checkpoint重建 |
| 官方代码 | SPONGE GRU/LSTM提取类与下载的固定commit逐字比较一致，已通过训练/评价；其完整原训练程序未执行 |
| 文档检查 | `check_docs_governance.py`通过；`git diff --check`通过 |

最新实际产物：

- [分组数据清单](../../../workspace/data/derived/modeling_5hz_fold0_20260912_001/dataset_manifest.json)
- [42次运行的准确计划](../../../workspace/runs/modeling/modeling_smoke_20260912_001/sweep_plan.json)
- [HOV seed0运行记录](../../../workspace/runs/modeling/modeling_smoke_20260912_001/fold0_hov_seed0/run_manifest.json)
- [HOV seed0骨架与mask评价](../../../workspace/runs/modeling/modeling_smoke_20260912_001/fold0_hov_seed0/evaluation_val/records.json)
- [逐序列结果](../../../workspace/reports/modeling_smoke_20260912_002/per_sequence.csv)
- [汇总结果](../../../workspace/reports/modeling_smoke_20260912_002/summary.csv)
- [统计诊断](../../../workspace/reports/modeling_smoke_20260912_002/statistics.json)

这些workspace产物是本地实验文件，源码仓库保留运行程序、协议与配置。前序 `_000` smoke及中间重评输出保留为开发记录；最新42次运行已统一先验与主训练的目标帧预算。

## 测试覆盖

[模型与流程测试](../../../tests/test_modeling_benchmark.py)检查因果窗口、跨组隔离、未来动作不可见、方向保持、train-only归一化、先验与监督目标对齐、训练阶段不打开test数组、重新训练时关闭分支、Bézier接点和等弧长采样、checkpoint哈希变化拒绝、输出目录覆盖拒绝、全流程指标汇总。

[指标与统计测试](../../../tests/test_modeling_benchmark_metrics.py)检查毫米尺度、节点对应顺序、末端定义、RMS与Chamfer定义、mask空集约定、边界容差、裁剪投影与亚像素半径，以及配对覆盖、重复样本拒绝、组内种子平均、精确置换、bootstrap与Holm校正。额外集成检查验证prefixed mask指标的“越大越好”方向和验证结果的诊断属性。

```bash
python -m unittest tests.test_modeling_benchmark tests.test_modeling_benchmark_metrics -v
python scripts/maintenance/check_docs_governance.py
```

## 证据范围

本次在CPU环境执行，PyTorch 2.6.0+cu118，CUDA未用于验证。训练预算极小，只证明运行链路、选择语义和产物合同成立；不能依据这些数值给出方法排名。真实test数组没有用于本次训练/选模或性能评价；准备清单时读取了完整语料用于数据完整性核查。

后续完整实验需要：在训练/验证组完成合理调参和收敛检查，冻结主指标与主比较，运行多折/多种子完整训练，最后在冻结测试组评价。现有7条同日序列的检验分辨率和分组依赖限制已写入协议；新增跨采集日的独立序列是扩大结论范围的必要证据。完整SoftNeRF的SDF/渲染复现、连续状态rollout和独立NDI末端评价不在本次已验证范围中。
