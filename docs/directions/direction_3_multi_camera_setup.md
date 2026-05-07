# 方向 3: 多相机真实世界采集系统 (Multi-Camera Real-World Setup)

> 核心目标：设计真实世界的多视角数据采集方案，为 sim-to-real 和纯 2D 自建模提供硬件基础。

---

## 1. 问题分析

### 1.1 为什么需要多相机

| 单视角问题 | 多视角解决方案 |
|------------|--------------|
| 深度歧义 (z 轴) | 双视角三角化 |
| 遮挡 (背面不可见) | 互补视角 |
| 尺度不确定 | 已知基线距离 → 绝对尺度 |
| 形状约束不足 | 多视角一致性 loss |

### 1.2 与现有工作的对比

| 工作 | 相机配置 | 输出 |
|------|---------|------|
| Chen 2022 | 5 × RealSense D435i (RGB-D) | 融合 3D 点云 |
| Hu 2025 (FBV-SM) | 1 × 单视角 RGB | 2D 灰度图 |
| Shan 2024 (SoftNeRF) | 多视角 RGB（仿真65视角 / 真实环绕视频） | 2D 图像 |
| **本方案** | **2 × RGB 相机 (正交)** | **2D 灰度图对** |

### 1.3 设计约束

真实软体机器人实验台的限制：
- 软体臂通常竖直安装在台面上
- 2D 驱动（x/y 轴扭矩）→ 主要弯曲在 xy 平面
- 相机需要避免遮挡驱动线缆和安装支架
- 预算有限（优先 RGB，不一定用 RGB-D）

---

## 2. 硬件设计

### 2.1 推荐方案：XY 双轴正交相机

```
          Camera 1 (正面, 看 yz 平面)
              ↓
              |
    ──────────┼────────── z 轴
              |
    [固定端]──软体臂──→ 自由端
    (base)                (tip)
              |
              ↑
          Camera 2 (侧面, 看 xz 平面)

俯视图:

         Camera 1
            ↓
     ┌──────┼──────┐
     │      │      │
     │  ─── 臂 ──→ │ ← x 轴 (弯曲方向 1)
     │      │      │
     └──────┼──────┘
     Camera 2
     (看 y 方向)
```

**相机参数建议**：
- 分辨率：640×480 或 1280×720
- 帧率：≥30 fps（与驱动频率匹配）
- 焦距：根据工作距离选择（建议 2-3 倍臂长的距离）
- 类型：工业 USB 相机（如 Logitech C920）或低成本 RGB-D（RealSense D435i）

### 2.2 备选方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| 2 × RGB | 低成本，简单 | 无深度信息 |
| 2 × RGB-D | 有深度 → 3D 点云 | 成本高，需对齐 |
| 1 × RGB + 标定板 | 可用结构光恢复 3D | 复杂，不鲁棒 |
| 3+ × RGB | 更完整的 3D 覆盖 | 遮挡、同步问题 |

**推荐**：先用 2 × RGB 验证，后续升级 RGB-D。

### 2.3 标定流程

```bash
# Step 1: 棋盘格标定（内参 + 畸变）
python scripts/calibration/calibrate_cameras.py \
    --images_dir calibration/camera1 \
    --board_size 9x6 \
    --square_size 0.025 \
    --output calibration/camera1_params.json

# Step 2: 双相机外参标定（相对位姿）
python scripts/calibration/calibrate_stereo.py \
    --images1_dir calibration/camera1 \
    --images2_dir calibration/camera2 \
    --board_size 9x6 \
    --square_size 0.025 \
    --output calibration/stereo_params.json

# Step 3: 世界坐标系对齐（与软体臂 base 对齐）
python scripts/calibration/align_world.py \
    --stereo_params calibration/stereo_params.json \
    --marker_pos "0,0,0" \
    --output calibration/world_transform.json
```

---

## 3. 数据采集系统

### 3.1 软件架构

```
collect_real.py (主控)
  ├── CameraSync (双相机同步采集)
  │     ├── Camera 1 → image1
  │     └── Camera 2 → image2
  ├── ActuatorController (驱动控制)
  │     └── send_command(torque_x, torque_y)
  └── DataWriter (数据保存)
        └── save .npz: {image_front, image_side, action, timestamp}
```

### 3.2 采集脚本

```bash
# 基本采集（随机动作序列）
python scripts/data/collect_real.py \
    --camera1 /dev/video0 \
    --camera2 /dev/video1 \
    --calibration calibration/ \
    --n_steps 500 \
    --action_range 0.3 \
    --output data/real_data/session_001

# 周期性动作采集（更平滑的轨迹）
python scripts/data/collect_real.py \
    --camera1 /dev/video0 \
    --camera2 /dev/video1 \
    --calibration calibration/ \
    --n_steps 500 \
    --mode sinusoidal \
    --freq_range 0.5 2.0 \
    --output data/real_data/session_002
```

### 3.3 数据格式

每个 `.npz` 文件包含：

```python
{
    'images_front': (T, H, W),      # 正面视角灰度图
    'images_side':  (T, H, W),      # 侧面视角灰度图
    'actions':      (T, 2),         # [torque_x, torque_y]
    'timestamps':   (T,),           # 时间戳 (秒)
    'camera_params': {
        'front': {'K': (3,3), 'dist': (5,), 'R': (3,3), 't': (3,)},
        'side':  {'K': (3,3), 'dist': (5,), 'R': (3,3), 't': (3,)},
    }
}
```

---

## 4. 模型适配

### 4.1 双视角渲染

当前 `get_rays()` 只支持单相机。需要扩展为多相机：

```python
class MultiViewRenderer:
    """多视角体渲染器。"""

    def __init__(self, camera_params_list, H, W, near=0.5, far=1.5, n_samples=64):
        self.views = []
        for cam in camera_params_list:
            rays_o, rays_d = get_rays(H, W, cam['focal'],
                                       cam['eye'], cam['center'], cam['up'])
            self.views.append({
                'rays_o': rays_o,
                'rays_d': rays_d,
                'near': near,
                'far': far,
                'n_samples': n_samples,
            })

    def render_all_views(self, model, action_window):
        """渲染所有视角。"""
        results = []
        for view in self.views:
            pts, z_vals = sample_stratified(
                view['rays_o'], view['rays_d'],
                view['near'], view['far'], view['n_samples'], perturb=False
            )
            raw = model(pts.unsqueeze(0), action_window)
            rendered, _ = OM_rendering(raw.squeeze(0))
            results.append(rendered)
        return results  # list of (N_rays,) tensors
```

### 4.2 多视角训练 Loss

```python
def multi_view_loss(model, action_window, gt_images_list, renderer):
    """
    Args:
        gt_images_list: [gt_front, gt_side] — 两个视角的 GT 图像
    """
    pred_images_list = renderer.render_all_views(model, action_window)

    total_loss = 0
    for pred, gt in zip(pred_images_list, gt_images_list):
        total_loss += F.mse_loss(pred, gt)

    # 视角一致性 loss（3D 形状在两个视角应该一致）
    # 已由共享密度场隐式保证

    return total_loss / len(gt_images_list)
```

### 4.3 仿真环境适配

在仿真中模拟双相机采集，生成训练数据：

```python
# elastica_env.py 中添加第二个相机

CAMERA_FRONT = {
    'eye': np.array([1.5, 0.0, 0.5]),
    'center': np.array([0.0, 0.0, 0.25]),
    'up': np.array([0.0, 0.0, 1.0]),
}

CAMERA_SIDE = {
    'eye': np.array([0.0, 1.5, 0.5]),
    'center': np.array([0.0, 0.0, 0.25]),
    'up': np.array([0.0, 0.0, 1.0]),
}
```

---

## 5. 实现文件清单

| 文件 | 用途 |
|------|------|
| `scripts/calibration/calibrate_cameras.py` | 单相机内参标定 |
| `scripts/calibration/calibrate_stereo.py` | 双相机外参标定 |
| `scripts/calibration/align_world.py` | 世界坐标系对齐 |
| `scripts/data/collect_real.py` | 真实世界数据采集 |
| `scripts/data/collect_2view_sim.py` | 仿真双视角数据采集 |
| `src/utils/multi_view_renderer.py` | 多视角渲染器 |
| `src/data/dataset_2view.py` | 双视角数据集 |
| `notebooks/13_multi_camera_setup.ipynb` | 相机标定和验证 |

---

## 6. 验证步骤

### 6.1 仿真验证（不需要硬件）

```bash
# Step 1: 生成仿真双视角数据
python scripts/data/collect_2view_sim.py \
    --n_episodes 200 \
    --output data/sequence_data_2view_sim

# Step 2: 训练双视角模型
python scripts/training/train_ms_scnf_mv.py \
    --data_dir data/sequence_data_2view_sim \
    --n_epochs 200 \
    --save_dir train_log/ms_scnf_2view_sim

# Step 3: 评估
python scripts/evaluation/evaluate_3d.py \
    --checkpoint train_log/ms_scnf_2view_sim/model/best_model.pt \
    --data_dir data/sequence_data_3d \
    --save_dir output/multi_view_eval
```

### 6.2 真实世界验证（需要硬件）

```bash
# Step 1: 标定相机
python scripts/calibration/calibrate_cameras.py --images_dir calibration/front
python scripts/calibration/calibrate_cameras.py --images_dir calibration/side
python scripts/calibration/calibrate_stereo.py --images1_dir calibration/front --images2_dir calibration/side

# Step 2: 采集数据
python scripts/data/collect_real.py \
    --camera1 /dev/video0 --camera2 /dev/video1 \
    --n_steps 500 --output data/real_data/session_001

# Step 3: 训练
python scripts/training/train_ms_scnf_mv.py \
    --data_dir data/real_data/session_001 \
    --n_epochs 200 \
    --save_dir train_log/ms_scnf_real

# Step 4: 可视化
python scripts/evaluation/visualize_predictions.py compare \
    --checkpoint train_log/ms_scnf_real/model/best_model.pt \
    --data_dir data/real_data/session_001
```

### 6.3 Notebook 验证

`13_multi_camera_setup.ipynb`:
```
1. 相机标定可视化
   - 棋盘格检测
   - 内参矩阵
   - 畸变系数
   - 重投影误差

2. 双视角数据检查
   - 正面/侧面图像对
   - 同步性验证（时间戳差异）
   - 动作-图像对应关系

3. 双视角渲染验证
   - 同一 3D 形态在两个视角的渲染结果
   - 视角一致性检查

4. 多视角 vs 单视角对比
   - 骨架精度
   - 深度恢复能力
   - 渲染质量
```

---

## 7. 创新点

1. **首个面向软体机器人的低成本双视角自建模系统**
   - Chen 2022 用了 5 个 RGB-D 相机（成本高、部署复杂）
   - 我们的方案仅需 2 个 RGB 相机

2. **正交平面视角设计**
   - 利用软体臂 2D 驱动（x/y）的先验 → 两个正交视角最优覆盖
   - 比随意放置相机更高效

3. **仿真→真实的渐进验证**
   - 先在仿真中验证双视角训练的有效性
   - 再迁移到真实世界

---

## 8. 时间规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| Week 1-2 | 仿真验证 | 双视角仿真数据 + 训练 + 评估 |
| Week 3-4 | 硬件搭建 | 相机采购/安装/标定 |
| Week 5-6 | 真实数据采集 | 采集脚本 + 数据质量验证 |
| Week 7-8 | 真实训练 | 在真实数据上训练并评估 |

**依赖**：方向 2 的纯 2D 训练框架应先完成，多视角是其扩展。
