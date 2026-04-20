# 软体机器人形态估计 — 仿真、数据采集与训练流程

> 将论文"Teaching robots to build simulations of themselves"的 FBV-SM 方法应用于三腔道驱动软体机械臂。
> 仿真基于 PyElastica，渲染基于 PyVista。

---

## 1. 整体流程概览

```
PyElastica 物理仿真 → PyVista 渲染图像 → 数据采集 (.npz) → 神经场训练 → 形态估计
```

---

## 2. 项目目录结构

```
SelfSoftRobot/
├── elastica_env.py               # 仿真环境 + 渲染 + 物理参数常量
│
├── src/                          # 可复用模块
│   ├── models/                   # 所有模型定义
│   │   ├── model.py              #   FBV_SM（静态 NeRF 基线）
│   │   ├── model_seq.py          #   model_v1（LSTM + NeRF）
│   │   ├── model_seq_skip.py     #   model_v2（LSTM + Skip + NeRF）
│   │   ├── model_seq_skip_pinn.py#   model_v3（CNN+LSTM PINN，图像→图像）
│   │   ├── model_seq_open_loop.py#   RecurrentPhysicsModel（开环）
│   │   ├── model_v4_nerf_pinn.py #   NeRF_PINN（NeRF+PINN）
│   │   ├── model_v5_deformable.py#   DeformableSoftRobotModel（可变形 NeRF）
│   │   └── layers.py             #   通用层（PositionalEncoder, LSTM, MLP 等）
│   ├── data/
│   │   └── dataset.py            # SoftSequenceDataset + load_soft_data
│   ├── utils/
│   │   ├── camera.py             # get_rays（射线生成）
│   │   ├── rendering.py          # 渲染函数（OM_rendering, sample_stratified 等）
│   │   └── visualization.py      # generate_validation_gif
│   └── training/
│       └── rendering.py          # run_batch_rendering_nerf, run_full_rendering_nerf
│
├── scripts/
│   ├── data_collection/          # 数据采集脚本
│   │   ├── collect_data.py       #   统一入口（batch / sequence 两种模式）
│   │   ├── collect_remote.py     #   旧版批量采集（每次独立仿真）
│   │   └── collect_sequence.py   #   旧版连续采集
│   ├── training/                 # 训练脚本
│   │   ├── train_soft.py         #   基线（FBV_SM）
│   │   ├── train_soft_seq2x.py   #   v1（model_v1 + LSTM 编码）
│   │   ├── train_soft_seq2x_vis.py # v2（model_v2 + Skip + 可视化）
│   │   ├── train_soft_v3.py      #   v3（CNN+LSTM PINN，图像→图像）
│   │   ├── train_soft_v4.py      #   v4（开环 CNN+LSTM）
│   │   ├── train_soft_v4_nerf.py #   v4 NeRF-PINN
│   │   └── train_soft_v5.py      #   v5（可变形 NeRF）
│   └── visualization/            # 可视化与测试
│       ├── view_data.py          #   交互式数据浏览（滑块）
│       ├── save_gif.py           #   快速 GIF 预览
│       ├── verify_simulation_3d.py # 3D 一致性验证
│       └── test_3d_seq.py        #   3D 序列推理可视化
│
├── docs/                         # 文档
├── data/                         # 数据目录
├── train_log/                    # 训练日志与模型保存
│
└── (原始论文参考代码: env.py, train.py, func.py, predefined.py, ...)
```

---

## 3. 模型迭代历程

| 版本 | 模型类 | 核心思想 | 训练脚本 | 渲染方式 |
|------|--------|---------|---------|---------|
| 基线 | `FBV_SM` | 静态 NeRF：3D 坐标 + 动作 → 密度 | `train_soft.py` | 体渲染（射线采样） |
| v1 | `model_v1` | LSTM 编码动作序列 → 物理状态 → 空间解码 | `train_soft_seq2x.py` | 体渲染（分块） |
| v2 | `model_v2` | v1 + 当前动作 Skip Connection 直连解码器 | `train_soft_seq2x_vis.py` | 体渲染（分块） |
| v3 | `model_v3` | CNN+LSTM 图像到图像自编码 + PINN 平滑 | `train_soft_v3.py` | CNN Decoder（无射线） |
| v4 | `RecurrentPhysicsModel` | 开环：只用首帧 + 动作序列预测 | `train_soft_v4.py` | CNN Decoder（无射线） |
| v4-N | `NeRF_PINN` | 静态结构记忆 + 动态物理引擎 + PINN | `train_soft_v4_nerf.py` | 体渲染（分块） |
| v5 | `DeformableSoftRobotModel` | Canonical Space + Deformation Field | `train_soft_v5.py` | 体渲染（变形场） |

---

## 4. 仿真环境

### 4.1 物理模型

使用 PyElastica 的 **CosseratRod**（Cosserat 杆模型）模拟软体机械臂：

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_ELEMENTS` | 30 | 离散单元数 |
| `ROD_LENGTH` | 0.5 m | 杆体长度 |
| `ROD_RADIUS` | 0.015 m | 杆体半径 |
| `ROD_DENSITY` | 1000 kg/m³ | 密度 |
| `YOUNGS_MODULUS` | 1e6 Pa | 杨氏模量 |
| `POISSON_RATIO` | 0.5 | 泊松比（近似不可压缩） |
| `DAMPING_CONSTANT` | 0.1 | 阻尼系数 |
| `RAMP_UP_TIME` | 0.5 s | 扭矩渐升时间 |

杆体从原点出发，沿 Z 轴向上延伸，底端固定（`OneEndFixedBC`）。

### 4.2 驱动方式

当前为 **2 维简化版**驱动（`torque_x`, `torque_y`），通过 `SimpleDistributedTorque` 沿杆体均匀施加分布扭矩。后续将扩展为 3 腔道驱动（3 维动作空间）。

### 4.3 两种仿真模式

**静态模式**（`get_simulation_data_pair`）：每次创建全新仿真实例，适合独立数据采集。

**连续模式**（`ContinuousSoftArmEnv`）：保持仿真状态持续运行，适合时序关联数据采集。

---

## 5. 渲染管线

### 5.1 相机参数

| 参数 | 值 |
|------|-----|
| `CAMERA_EYE` | (1.5, 0.0, 0.5) |
| `CAMERA_CENTER` | (0.0, 0.0, 0.25) |
| `CAMERA_UP` | (0.0, 0.0, 1.0) |
| 默认图像尺寸 | 100×100 |

### 5.2 渲染流程

```
3D 杆体数据 → PyVista tube 渲染 → 黑底白管 → 截图 → 灰度化 → 二值化 → 0/1 图像
```

---

## 6. 数据采集

### 6.1 统一采集脚本

```bash
# 批量静态采集
python scripts/data_collection/collect_data.py --mode batch --count 100

# 连续时序采集
python scripts/data_collection/collect_data.py --mode sequence --sequences 10 --actions-per-seq 50
```

### 6.2 数据格式

每个 `.npz` 文件包含：`images` (T,H,W)、`actions` (T,D)、`focal`、`dt`、`camera_eye`、`camera_center`、`camera_up`。

---

## 7. 与论文原始方法的差异

| 方面 | 论文（刚性臂） | 当前（软体臂） |
|------|--------------|--------------|
| 仿真器 | PyBullet（刚体） | PyElastica（Cosserat 杆） |
| 自由度 | 4 个旋转关节 | 2 维分布扭矩（简化版） |
| 坐标变换 | 前 2 角度用旋转矩阵处理 | 无关节变换，射线直接查询 |
| 渲染方式 | PyBullet 内置相机 | PyVista 离屏渲染 |
| 输出 | 灰度可见性图 | 二值掩码图 |
| 动态特性 | 无 | 阻尼 + 扭矩渐升 |
| 时序依赖 | 无（单帧独立） | 有（连续仿真状态） |
