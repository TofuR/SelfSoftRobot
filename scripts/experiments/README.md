# 实验与离线分析脚本

在项目根目录使用已配置的项目 Python 环境。基础依赖见 `requirements.txt`，分析和论文工具的补充依赖安装方式：

```bash
python -m pip install -r requirements-analysis.txt
```

补充依赖为 pandas（表格分析）、scikit-image（实机图像骨架）、python-docx 和 lxml（Word/OOXML）。数值计算、模型加载和绘图还使用基础环境中的 NumPy、SciPy、PyTorch、Matplotlib、OpenCV、Pillow 和 threadpoolctl。模型相关入口依赖完整的 `src/benchmarks/`、`src/evaluation/modeling_benchmark_metrics.py` 及模型/算子源码。

## 数据与输出前提

多数入口固定读取 `workspace/runs/training/`、`workspace/runs/analysis/`、`workspace/runs/validation/` 中带日期和实验编号的目录。需要保留相应协议、数据清单、检查点、冻结源码、逐帧预测和上游分析结果；部分清单还包含原机器的绝对路径。`workspace/` 被 Git 忽略，检出代码不会获得这些数据。运行前检查脚本中的 `STUDY`、`RUN`、`SOURCE`、`OUT` 等路径及其上游文件。

离线入口会写入分析结果、图表或报告。部分一次性脚本在导入时就执行读写，应按独立脚本使用。论文绘图和打包工具还可能依赖本地 `docs/icra2027/` 稿件与图片；该目录内容被 Git 忽略，相关文件需本地提供。

## 常用离线入口

以下文件均位于 `scripts/experiments/`，输入数据齐备后再运行。

| 用途 | 入口 | 主要前提 |
| --- | --- | --- |
| 表征与历史机制 | `analyze_modeling_representation.py`、`analyze_modeling_branch_interpretation.py`、`analyze_modeling_history_mechanisms.py`、`analyze_modeling_time_memory.py` | 冻结检查点及训练/验证/测试清单；branch 使用 representation 输出 |
| 20 次重复实验分析 | `analyze_unified20_geometry.py`、`analyze_unified20_efficiency.py`、`analyze_unified20_sampling.py` | 完整 unified20 训练产物；sampling 复用 time-memory 脚本和原始采样数据 |
| 论文补充机制 | `analyze_memory_physics.py`、`analyze_draft2_memory_reuse.py`、`analyze_draft2_spatial_reversal.py`、`analyze_path_memory_section36_v3.py` | geometry、MLP 记忆对照及冻结预测等上游结果 |
| 实机图像测量 | `analyze_real_control_image_endpoints.py` → `convert_real_control_endpoint_units.py` → `analyze_draft2_real_execution.py` | 归档实机图像、采样记录与配准矩阵；按箭头顺序准备输出 |
| 绘图 | `plot_unified20_results.py`、`plot_unified20_geometry.py`、`plot_unified20_efficiency_sampling.py` 及其余 `plot_*.py` | 已完成的专项分析结果 |
| 报告 | `report_modeling_improvement.py`、`build_modeling_*_html.py`、`build_unified20_report.py`、`build_modeling_extensions_report.py`、`build_completed_control_and_mlp_report.py` | 对应实验汇总、图目录及稿件；HTML 交付依赖见下文 |
| 方法讲解 | `build_modeling_method_walkthrough.py` + `assets/modeling_method_walkthrough.html` | 固定 seed100 HOV 检查点及验证窗口；直接生成页面 |
| Word 章节维护 | `scripts/paper/update_v3_section36_docx.py`（从项目根目录定位） | 指定旧稿/备份、`section36.md` 和章节图片；另需 Pandoc |

文件名前缀不能作为是否训练的判断依据：`analyze_modeling_plugin_convergence.py`、`analyze_modeling_sweep_hysteresis.py` 包含拟合；`evaluate_hov_initialization_only.py` 会执行参考模型的 750 次 Adam 更新及记忆读出拟合。`modeling_*study.py`、`modeling_benchmark.py`、`run_*`、`extend_modeling_plugin_repetitions.py`、`complete_internal_mlp_memory_comparison.py` 是实验/调度入口，需按各自参数和实验协议使用。

## 可选 HTML 报告 helper

`build_unified20_report.py --deliver` 以及 extensions/completed 报告的 HTML 交付需要 Node.js 和 Data Analytics 的 `deliver_portable_artifact.mjs` 及其配套依赖；这些不由 pip 安装。优先显式配置 helper 文件路径：

```bash
export SELF_SOFT_REPORT_DELIVERY=/path/to/data-analytics/skills/build-report/scripts/deliver_portable_artifact.mjs
```

未配置时，在 `${CODEX_HOME:-$HOME/.codex}/plugins/cache/*/data-analytics/*/skills/build-report/scripts/` 中查找已安装 helper。找到唯一文件时自动使用；多版本并存时列出候选，要求显式选择。显式路径无效时直接报错。路径在实际交付时解析，缺少 helper 不影响导入共享报告模块或仅生成 unified20 的 `artifact.json`；extensions/completed 入口仍遵循各自已有的交付流程。

`build_modeling_method_walkthrough.py` 使用仓库内 HTML 模板，不需要此 helper。Word 工具的 Pandoc 是另一个独立的系统依赖。
