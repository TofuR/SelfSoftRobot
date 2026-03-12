# SelfSoftRobot 项目帮助文档

本文档用于快速理解项目代码结构、核心模块职责、主要函数作用和推荐执行流程。

## 1. 项目整体逻辑

项目主线可以概括为：

1. 数据采集：通过 PyBullet/Elastica 环境采集动作-观测数据。
2. 模型训练：使用体渲染（NeRF 风格）和时序模型学习软体机械臂状态。
3. 推理与可视化：输出 2D/3D 预测结果并与仿真进行对比验证。

---

## 2. 根目录 Python 文件说明

### 2.1 环境与几何工具

- `env.py`
  - 作用：PyBullet Gym 环境 `FBVSM_Env`，负责相机、关节控制、碰撞检测、观测返回。
  - 关键接口：`reset()`、`act()`、`get_obs()`。

- `elastica_env.py`
  - 作用：基于 PyElastica 的连续软体臂物理仿真环境。
  - 关键接口：`create_simulation()`、`get_simulation_data_pair()`、`ContinuousSoftArmEnv.step()`、`ContinuousSoftArmEnv.get_observation()`。

- `func.py`
  - 作用：体渲染核心工具库（射线生成、分层采样、渲染聚合、模型前向封装、坐标变换）。
  - 关键接口：`get_rays()`、`sample_stratified()`、`sample_pdf()`、`sample_hierarchical()`、`OM_rendering()`、`model_forward()`。

- `predefined.py`
  - 作用：动作列表与碰撞筛选工具，含颜色掩码和旋转矩阵函数。
  - 关键接口：`generate_action_list()`、`self_collision_check*()`。

### 2.2 数据采集脚本

- `data_collection.py`
  - 作用：按动作列表采集仿真图像与关节角数据，保存为 `npz`。
  - 关键接口：`collect_data()`、`w2c_matrix()`。

- `collect_remote.py`
  - 作用：远程/批量触发数据采集流程。
  - 关键接口：`run_debug_check()`、`run_batch_collection()`。

- `collect_sequence.py`
  - 作用：采集连续时间序列数据（随机游走动作 + 连续仿真步进）。
  - 关键接口：`generate_random_walk_actions()`、`collect_continuous_data()`。

### 2.3 训练脚本

- `train.py`
  - 作用：基础训练入口（较早版本），初始化模型并执行训练。
  - 关键接口：`init_models()`、`train()`。

- `train_soft.py`
  - 作用：软体机器人体渲染训练（早期版本）。
  - 关键接口：`soft_sample_stratified()`、`Robust_Mask_Rendering()`、`soft_model_forward()`、`load_soft_data()`、`train()`。

- `train_soft_seq2x.py`
  - 作用：序列训练（2x 版本）。
  - 关键接口：`SoftSequenceDataset`、`train_seq()`。

- `train_soft_seq2x_vis.py`
  - 作用：序列训练并定期输出可视化结果。
  - 关键接口：`SoftSequenceDataset`、`train_seq_vis()`。

- `train_soft_v3.py`
  - 作用：V3 图像序列监督训练。
  - 关键接口：`ImageSequenceDataset`、`train_v3()`、`visualize_and_save()`。

- `train_soft_v4.py`
  - 作用：V4 序列训练流程。
  - 关键接口：`ImageSequenceDataset`、`train_v4()`、`visualize_and_save()`。

- `train_soft_v4_nerf.py`
  - 作用：V4 NeRF+PINN 训练与评估。
  - 关键接口：`SoftSequenceDataset`、`train_v4_nerf()`、`evaluate_and_save()`。

- `train_soft_v5.py`
  - 作用：V5 可变形场（Deformable）训练与可视化。
  - 关键接口：`SoftSequenceDataset`、`run_full_rendering()`、`train_v5_deformable()`。

### 2.4 测试、验证与可视化

- `test_model.py`
  - 作用：单模型查询/交互测试，输出占据点云或末端估计。
  - 关键接口：`test_model()`、`query_models()`、`query_models_separated_outputs()`。

- `test_3d_seq.py`
  - 作用：序列预测结果的 3D 可视化辅助。
  - 关键接口：`ResultVisualizer`。

- `verify_simulation_3d.py`
  - 作用：3D 仿真投影验证。
  - 关键接口：`get_camera_matrix()`、`project_points()`、`run_verification()`。

- `visualize_bullet.py`
  - 作用：Bullet 环境交互与路径规划（A*/RRT）可视化。
  - 关键接口：`collision_free_planning()`、`A_star_search()`、`rrt()`、`shortcut_path()`。

- `view_data.py`
  - 作用：快速查看数据文件内容。
  - 关键接口：`view_dataset()`。

- `save_gif.py`
  - 作用：把 `npz` 序列导出为 GIF。
  - 关键接口：`save_as_gif()`。

### 2.5 兼容导出层

- `model.py`、`model_seq.py`、`model_seq_skip.py`、`model_seq_skip_pinn.py`
  - 作用：兼容导入路径的 re-export 文件，实际实现位于 `src/models/`。

---

## 3. src 目录说明

### 3.1 `src/models/`

- `layers.py`
  - 通用网络层：`PositionalEncoder`、`TemporalLSTMEncoder`、`MLPDecoder` 等。

- `model.py`
  - 基础网络：`FBV_SM`。

- `model_seq.py`
  - 序列模型 V1：时序编码 + 空间解码。

- `model_seq_skip.py`
  - 序列模型 V2（skip 思路）。

- `model_seq_skip_pinn.py`
  - 序列模型 V3（加入 PINN 平滑约束）。

- `model_seq_open_loop.py`
  - 开环递归物理模型。

- `model_v4_nerf_pinn.py`
  - NeRF+PINN 模型：静态几何 + 动态物理状态融合。

- `model_v5_deformable.py`
  - 可变形模型：Canonical 空间 + 变形场建模。

### 3.2 `src/data/`

- `dataset.py`、`preprocessing.py`
  - 当前以占位/导出为主，可扩展统一数据集与预处理流程。

### 3.3 `src/config/`、`src/training/`、`src/utils/`

- 提供包结构与后续扩展入口（当前实现较轻量）。

---

## 4. tests 目录脚本说明

- `tests/visualize_prediction.py`
  - 单帧预测渲染可视化。

- `tests/visualize_seq_prediction.py`
  - 时序预测逐帧可视化。

- `tests/visualize_prediction_3d.py`
  - 3D 点云预测与物理重建对比。

---

## 5. 关键函数流（从采集到训练）

### 5.1 采集流程

1. 生成动作（`generate_action_list` 或 `generate_random_walk_actions`）
2. 环境执行（`FBVSM_Env.step/act` 或 `ContinuousSoftArmEnv.step`）
3. 获取观测（`get_obs` / `get_observation`）
4. 保存为 `npz`

### 5.2 训练流程

1. 加载数据（`SoftSequenceDataset` / `load_soft_data`）
2. 生成射线（`get_rays_from_camera_params`）
3. 射线采样（`sample_stratified` / `soft_sample_stratified`）
4. 网络前向（`model_forward` / `forward_rendering` / `query_field`）
5. 渲染聚合（`OM_rendering` / `Robust_Mask_Rendering`）
6. 反向更新并保存可视化

---

## 6. 推荐使用顺序

1. 先采集：`collect_sequence.py` 或 `data_collection.py`
2. 训练：优先 `train_soft_v4_nerf.py` / `train_soft_v5.py`
3. 验证：`tests/visualize_seq_prediction.py`、`tests/visualize_prediction_3d.py`
4. 定位问题：`test_model.py` + `verify_simulation_3d.py`

---

## 7. 常见注意点

- 不同脚本中相机参数（`eye/center/up`、`near/far`、`focal`）需保持一致，否则 GT/预测会错位。
- 动作归一化因子（`action_norm_factor.txt`）必须与训练/推理保持一致。
- `func.py` 的渲染模式由 `output_flag` 决定，不同模型应使用匹配渲染头。
- 序列模型训练时应区分“训练随机采样射线”和“验证全图渲染”，避免显存不足。
