# 研究方向总览 (Research Directions Overview)

> 基于 MS-SCNF 实验结果和文献分析，识别出 5 个研究方向。每个方向独立成文，但互有关联。

---

## 方向关系图

```
                    ┌─────────────────────┐
                    │  5. 时序迟滞建模     │ ← 核心创新点
                    │  (Temporal Hysteresis)│
                    └──────────┬──────────┘
                               │ 时序编码改进
                               │
    ┌──────────────┐    ┌──────┴───────┐    ┌──────────────┐
    │ 1. 形态发现   │    │  2. 纯 2D    │    │ 4. Sim-to-Real│
    │ (Morphology  │    │  自建模       │    │ (迁移学习)    │
    │  Discovery)  │    │ (Pure 2D)    │    │              │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │
           │ 空间结构          │ 训练信号          │ 数据域
           │                   │                   │
           └───────────┬───────┘                   │
                       │                           │
              ┌────────┴────────┐                  │
              │ 3. 多相机系统    │◄─────────────────┘
              │ (Multi-Camera)  │   真实世界数据
              └─────────────────┘
```

## 各方向文档

| 方向 | 文档 | 核心问题 |
|------|------|---------|
| 1 | [direction_1_morphology_discovery.md](direction_1_morphology_discovery.md) | 如何让机器人发现自身结构？ |
| 2 | [direction_2_pure_2d_self_modeling.md](direction_2_pure_2d_self_modeling.md) | 仅用 2D 图像能学到什么？ |
| 3 | [direction_3_multi_camera_setup.md](direction_3_multi_camera_setup.md) | 多视角如何帮助消除深度歧义？ |
| 4 | [direction_4_sim2real_transfer.md](direction_4_sim2real_transfer.md) | 仿真知识如何迁移到真实？ |
| 5 | [direction_5_temporal_hysteresis.md](direction_5_temporal_hysteresis.md) | 如何建模粘弹性迟滞？ |

## 优先级排序

### 论文创新贡献角度
1. **方向 5**（时序迟滞）— 最强创新点，现有工作完全空白
2. **方向 1**（形态发现）— 从硬编码到自发现的范式转变
3. **方向 2**（纯 2D）— 量化自建模能力边界

### 实验可行性角度（仿真环境）
1. **方向 5** — 可以直接在现有仿真器上验证
2. **方向 2** — 仅需修改训练 loss，不需要新数据
3. **方向 1** — 需要新模型设计，但可在仿真中验证

### 需要真实硬件
4. **方向 4**（Sim-to-Real）— 最终目标，依赖方向 2/3
5. **方向 3**（多相机）— 硬件搭建 + 标定

## 建议实施路线

```
Phase 1 (Week 1-4): 方向 5 + 方向 2
  → 验证仿真器迟滞行为
  → 实现 HA-EMA
  → 实现纯 2D 训练
  → 对比实验

Phase 2 (Week 5-8): 方向 1 + 方向 5 深化
  → 路线 A: 2D 骨架学习
  → 迟滞数据采集与评估
  → 消融实验

Phase 3 (Week 9-16): 方向 3 + 方向 4
  → 硬件搭建
  → 多视角数据采集
  → Sim-to-Real 迁移
```

## 每个方向的产出物

| 方向 | 模型文件 | 训练脚本 | 评估脚本 | Notebook |
|------|---------|---------|---------|----------|
| 1 | `model_ms_scnf_2d.py`, `model_adaptive_skeleton.py`, `model_sdf.py` | `train_ms_scnf_2d.py`, `train_adaptive_skeleton.py`, `train_sdf.py` | `eval_skeleton_discovery.py` | 09, 10, 11 |
| 2 | `model_ms_scnf_2d.py`, `model_ms_scnf_mv.py` | `train_ms_scnf_2d.py`, `train_ms_scnf_2d_curriculum.py` | `evaluate_3d.py`, `compare_views.py` | 12 |
| 3 | `multi_view_renderer.py`, `dataset_2view.py` | `train_ms_scnf_mv.py` | - | 13 |
| 4 | `model_ms_scnf_da.py`, `domain_randomization.py` | `finetune_real.py`, `train_domain_adaptive.py` | `evaluate_real.py` | 14 |
| 5 | `layers_hysteresis.py`, `model_ms_scnf_hysteresis.py` | `train_hysteresis.py`, `train_ms_scnf_hysteresis.py` | `eval_hysteresis.py`, `plot_hysteresis_loops.py` | 15, 16 |
