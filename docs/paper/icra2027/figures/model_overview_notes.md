# 模型总览图

[可编辑 SVG](model_overview.svg) · [PNG 预览](../../../../workspace/reports/model_overview_20260913/model_overview.png) · [矢量 PDF](../../../../workspace/reports/model_overview_20260913/model_overview.pdf)

白底、细线和低饱和分支配色，尺寸为 180 × 82.125 mm。全图为原生 SVG：软臂表面、相机、曲线、文字和箭头均可编辑。PNG 为 2400 × 1095 像素；PDF 保留矢量。

## 中文图注

**融合路径记忆与时间记忆的视觉全身自建模框架。** (a) 单摄像头位于软臂前方，朝向臂身中部；场景以斜上方观察视角绘制。单视角图像提供训练中心线标签，压力历史驱动预测模型。(b) 当前输入产生参考形状坐标，路径记忆与时间记忆分别通过学习读出贡献历史相关的坐标修正，三者相加形成形状坐标。示意的加载—反转—保持输入下，路径响应在保持阶段维持偏移，时间响应逐渐衰减。(c) 根据局部弯曲角与区间长度，从固定基点生成连通的 15 节点平面中心线。曲线由算子递推方程在示意输入下计算；软臂形态参考控制实验图像及该实验记录的模型预测中心线。

## English caption

**Visual whole-body self-modeling with path and time memory.** (a) A single camera faces the middle of the soft arm; the scene is shown from an elevated oblique viewpoint. Images provide centerline supervision, while the pressure history drives the predictive model. (b) The current-input reference and learned path- and time-memory contributions are added in shape-coordinate space. Under an illustrative loading–reversal–hold input, the path response retains an offset during the hold, while the time response decays. (c) Local bending angles and interval lengths generate a connected, planar 15-node centerline from a fixed base. Curves are computed operator responses to the illustrative input. The arm depiction is informed by a control-experiment image and its logged model-predicted centerline.

## 数据与几何来源

- 实机外形参考：[控制实验帧 00049](../../../../workspace/runs/validation/real_robot_20260912_001/trials/t18_tip_left_clear_closed_5hz_i2/raw/cam0/00049.png)。该帧用于参考两段软臂的比例、连续连接及弯曲形态。
- 中心线来源：同一试验 [steps.jsonl](../../../../workspace/runs/validation/real_robot_20260912_001/trials/t18_tip_left_clear_closed_5hz_i2/steps.jsonl) 最后一条记录的 `prediction_px`，即控制模型记录的预测坐标。15 个坐标保存在 [control_shape.json](../../../../workspace/reports/model_overview_20260913/rendered_scene/control_shape.json)。这是模型日志几何，不是独立测量的三维真值。
- 绘制过程：将像素中心线放入一个平面，沿弧长插值，再添加示意截面并进行透视投影。外径、安装座和浅表纹理由图像估计，用于解释形态。场景相机位于 `[-25, -285, -145]`、朝向 `[-25, 0, -145]`；绘图观察点为 `[580, -1050, 700]`。这些是示意场景坐标，不表示物理标定结果。
- 记忆响应：初始状态为零，归一化示意驱动在 0、1.8、2.8、6 秒取 0、0.9、0.35、0.35，逐段线性变化，步长 0.025 秒。路径阈值为 0.02、0.5；显示 `q/r`。时间分支展示 0.6、1.0、2.0 秒三个说明性时间尺度，显示 `d=h−e`。正式模型使用六个时间尺度，图中概括展示三个。
- 路径状态由 `pₜ=clip(pₜ₋₁,eₜ−r,eₜ+r)`、`qₜ=eₜ−pₜ` 计算；时间状态由 `hₜ=exp(−Δt/τ)hₜ₋₁+[1−exp(−Δt/τ)]eₜ` 计算。图面已标明 “Illustrative operator responses”。这些是示意输入驱动的计算状态，不是直接测得的记忆信号。

## 与模型的对应

内容依据 [当前稿件](../../../icra2027/draft.md)、[模型实现](../../../../src/models/model_hereditary_geometry.py)、[路径算子](../../../../src/operators/play_bank.py)及[时间算子](../../../../src/operators/maxwell_bank.py)。

参考项与两个记忆项先在形状坐标空间相加，再生成中心线。`Wₚ`、`Wₕ` 概括学习读出；记忆模块包含单调驱动映射及状态更新。`ξ` 包含局部弯曲与两段对数长度变化；几何模块将长度坐标映射为正的区间长度 `ℓᵢ`，并累积局部弯曲角 `βᵢ`。图中的几何量标在透视投影后的同一条中心线上，角弧先在原始平面中构造再投影。外轮廓帮助识别软臂，模型输出为节点坐标。视觉监督关系在图注中交代。

## 论文绘图参考

已查看以下论文相关模型图的页面渲染，借鉴其信息组织方式，图形重新绘制。

| 论文与图号 | 参考的表达方式 |
| --- | --- |
| [Shan et al., SoftNeRF, IROS 2024](https://doi.org/10.1109/IROS58592.2024.10801344)，Fig. 2，PDF 第 3 页 | 模块颜色保持一致，以具体输入、输出图例承载方法语义。 |
| [Yu, Wang & Tan, 2026](https://arxiv.org/abs/2603.01751)，Fig. 1–2，PDF 第 2、4 页 | 驱动曲线与软臂骨架并列，几何控制点表达可解释形状。 |
| [Chen et al., Science Robotics 2022](https://doi.org/10.1126/scirobotics.abn1944)，Fig. 2，PDF 第 3 页 | 实体机器人、精简编码模块与具体几何输出形成清楚的流程。 |
| [Hu et al., Nature Machine Intelligence 2025](https://doi.org/10.1038/s42256-025-01006-w)，Fig. 1，PDF 第 2 页 | 相机布局与视觉观测明确说明自建模的数据来源。 |
| [Almanzor et al., TRO 2023](https://doi.org/10.1109/TRO.2023.3275375)，Fig. 2，PDF 第 5 页 | 用机器人输入输出实例及时间索引解释模型关系。 |
| [Tang et al., Science Advances 2026](https://doi.org/10.1126/sciadv.eae3712)，Fig. 2，PDF 第 3 页 | 分开表达不同功能分支，并通过系统实例保持物理含义。 |
| [Xu et al., TRO 2026](https://doi.org/10.1109/TRO.2026.3706564)，Fig. 1–2，PDF 第 4 页 | 透视场景同时表达相机、软臂及几何量的空间关系。 |

## 复现与导出

在项目根目录运行：

```bash
python workspace/reports/model_overview_20260913/rendered_scene/build_model_svg.py
/usr/bin/python3 workspace/reports/model_overview_20260913/rendered_scene/export_figure.py
```

[vector_scene.py](../../../../workspace/reports/model_overview_20260913/rendered_scene/vector_scene.py) 生成透视几何；[build_model_svg.py](../../../../workspace/reports/model_overview_20260913/rendered_scene/build_model_svg.py) 计算示意响应并排版；[export_figure.py](../../../../workspace/reports/model_overview_20260913/rendered_scene/export_figure.py) 根据 SVG 的 viewBox 和毫米尺寸导出。生成器使用 NumPy、SciPy；导出器使用系统 Python 的 Rsvg 与 Cairo。[生成记录](../../../../workspace/reports/model_overview_20260913/rendered_scene/model_figure_manifest.json) 保留来源与参数，[operator_probe.npz](../../../../workspace/reports/model_overview_20260913/rendered_scene/operator_probe.npz) 保留曲线数值。
