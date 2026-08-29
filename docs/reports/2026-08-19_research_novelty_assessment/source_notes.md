# 研究创新性评估：证据与口径说明

日期：2026-08-19

## 证据优先级

本评估有意不把仓库中既有的研究方向、related-work 草稿或个人构想当作文献结论。证据按以下顺序使用：

1. `docs/papers/` 中六篇已下载论文的 PDF 原文；
2. 仓库中可复查的模型输出、定量结果和代码能力；
3. 旧文档仅用于提出待验证问题，不用于支持“空白”“首次”或“创新”判断。

本地六篇论文只能构成代表性核查，不能替代系统综述。因此报告中的潜在创新均表述为“值得验证的研究空白”，不使用“首次”声明。

## 已核对的代表论文

- Almanzor et al. (2023), *Static Shape Control of Soft Continuum Robots Using Deep Visual Inverse Kinematic Models*：单相机、免标定、六路气动、全身静态形态闭环控制和部分遮挡增强；局限为准静态、每步视觉反馈，论文明确指出完全遮挡不能由该增强方法解决。
- Chen et al. (2022), *Fully body visual self-modeling of robot morphologies*：全身视觉自模型、模型内运动规划、异常检测及模型更新。
- Hu et al. (2025), *Teaching robots to build simulations of themselves*：机器人从视觉建立可查询的自身仿真，用于几何推理、规划和损坏后的模型恢复。
- Shan et al. (2024), *SoftNeRF*：以神经辐射场构造软体机器人视觉自模型，并支持几何/占用相关任务。
- Tang et al. (2026), *A general soft robotic controller ...*：离线共享表示、在线模型和策略更新，覆盖负载、风和执行器失效等变化，并在多个软体平台上进行全身整形。
- Yu et al. (2026), *Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control*：多视角免标定的 Bézier 形态表示、NODE 动力学、形态—末端混合控制、障碍避让和 self-motion；依赖持续形态反馈，严重遮挡仍是限制。

## 仓库证据

- `output/real_quant/exp_20260709_5/summary.txt`：31 节点 GTObserved 模型的 tip mean 为 1.929 px，node mean 为 3.906 px；NDI 标定底噪为 0.737 mm，tip NDI mean 为 0.765 mm。该结果使用每步真实形态锚定，不能代表长期无视觉 OpenLoop。
- `output/exp5b_hysteresis_loop/summary.txt`：真实序列中准静态 load/unload 半高宽约 1.53 mm，动态合并数据约 2.06 mm，只能证明存在路径/速率相关现象，不能单独证明某个历史编码器有效。
- `output/window_ablation/ablation_summary.json`：GTObserved 单步验证中 window=1 mean 1.216 px，window=40 mean 2.101 px。当前消融不支持“长历史必然更好”。
- `output/openloop_window_compare/compare_summary.json`：window=40 的 OpenLoop 误差随预测长度退化；window=1 的首步异常且后续下降，提示初始化或训练口径需排查，尚不能支持分数阶/GL 记忆的创新主张。
- `train_log/open_loop_transition/exp_20260714_8/eval_plan/plan_result.json`：模型内规划 mean 4.29 px，对比 do-nothing 10.91 px；但规划动作尚未在真实机器人执行，且 GT action 对比受目标时刻和 K 口径影响。
- `train_log/open_loop_transition/exp_20260714_8/transition_metrics.json`：最终评估含 NaN，不能作为完成的 OpenLoop 定量结论。
- 现有真实训练 NPZ 的动作维仍为 1；六通道采集、通道映射和训练接口是代码能力，不是已经获得的六通道双段实验结果。

## 建议补充的系统检索

数据库：IEEE Xplore、Scopus、Web of Science、Google Scholar。建议组合关键词：

- `intermittent visual feedback soft continuum robot`
- `occluded whole-body soft robot control`
- `history-conditioned soft robot dynamics`
- `hysteresis-aware inverse planning continuum robot`
- `event-triggered perception soft robot`
- `sparse observation model predictive control continuum robot`

检索后应制作纳入/排除表，并追踪引用链，重点排除已经同时覆盖“历史状态 + 观测中断 + 全身规划执行”的工作。
