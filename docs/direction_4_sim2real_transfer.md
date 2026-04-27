# 方向 4: 仿真到真实迁移 (Sim-to-Real Transfer)

> 核心问题：所有模型都在 PyElastica 仿真器上训练，但仿真的渲染与真实世界有显著差异。如何将仿真中学到的知识迁移到真实软体机器人？

---

## 1. 问题分析

### 1.1 Sim-Real 差距分析

| 维度 | 仿真 (PyElastica) | 真实世界 |
|------|-------------------|---------|
| 渲染 | 二值图像（0/1），无纹理 | 真实光照、阴影、纹理、噪声 |
| 形态 | 完美圆柱体 | 制造误差、表面粗糙度 |
| 物理模型 | 纯弹性 Cosserat 杆 | 粘弹性、蠕变、温度效应 |
| 驱动模型 | 理想扭矩输入 | 电机非线性、延迟、死区 |
| 背景 | 纯黑/纯白 | 复杂背景 |
| 传感器噪声 | 无 | 相机噪声、压缩伪影 |

### 1.2 迁移策略概述

三种主要策略：

1. **Domain Randomization (DR)**：在仿真中随机化参数，使模型对差异鲁棒
2. **Fine-tuning**：在仿真中预训练，用少量真实数据微调
3. **Domain Adaptation (DA)**：学习仿真→真实的映射（如 CycleGAN）

---

## 2. 策略 A: Domain Randomization

### 2.1 思路

在仿真训练时，对渲染和物理参数施加随机扰动，使模型学会忽略这些变化，从而在真实世界中也能工作。

### 2.2 随机化维度

```python
# 仿真渲染参数随机化
RENDER_RANDOMIZATION = {
    'background_color': (0, 255),      # 背景灰度值
    'noise_std': (0, 0.1),             # 高斯噪声强度
    'blur_kernel': (0, 3),             # 模糊核大小
    'brightness': (0.7, 1.3),          # 亮度变化
    'contrast': (0.7, 1.3),            # 对比度变化
    'rod_color': (100, 255),           # 杆体灰度值
    'shadow_intensity': (0, 0.3),      # 阴影强度
}

# 物理参数随机化
PHYSICS_RANDOMIZATION = {
    'youngs_modulus': (0.8e6, 1.2e6),  # 弹性模量 ±20%
    'damping': (0.05, 0.2),            # 阻尼系数
    'rod_radius': (0.012, 0.018),      # 杆体半径
    'rod_length': (0.48, 0.52),        # 杆体长度
}
```

### 2.3 实现方案

```python
class RandomizedRenderer:
    """带域随机化的渲染器。"""

    def __init__(self, base_env, render_rand=True, physics_rand=True):
        self.env = base_env
        self.render_rand = render_rand
        self.physics_rand = physics_rand

    def get_observation(self):
        img, action = self.env.get_observation()

        if self.render_rand:
            # 随机背景
            bg_value = np.random.randint(0, 30)
            mask = img == 0
            img[mask] = bg_value

            # 高斯噪声
            noise = np.random.normal(0, np.random.uniform(0, 0.1), img.shape)
            img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)

            # 随机亮度/对比度
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-30, 30)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        return img, action
```

### 2.4 数据采集

```bash
# 采集带域随机化的仿真数据
python scripts/data/collect.py \
    --n_episodes 500 \
    --output data/sequence_data_dr \
    --randomize_render \
    --randomize_physics
```

### 2.5 训练

```bash
# 在随机化数据上训练
python scripts/training/train_ms_scnf.py \
    --data_dir data/sequence_data_dr \
    --n_epochs 300 \
    --save_dir train_log/ms_scnf_dr
```

---

## 3. 策略 B: Fine-tuning

### 3.1 思路

先在大量仿真数据上预训练，然后用少量真实数据微调。关键是**冻结合适的层**。

### 3.2 层冻结策略

```
MS-SCNF 模型结构:
  temporal (MultiScaleEMA)     → 物理动态编码（泛化性好）
  skeleton_head (SkeletonHead) → 骨架预测（依赖物理精度）
  density (DensityField)       → 视觉外观（仿真-真实差距最大）

冻结策略:
  Fine-tune 方案 1: 只训练 density（外观适应）
  Fine-tune 方案 2: 训练 skeleton_head + density（形态适应）
  Fine-tune 方案 3: 全部训练（端到端适应）
```

### 3.3 实现方案

```python
# scripts/training/finetune_real.py

def setup_finetune(model, strategy='density_only'):
    """设置微调冻结策略。"""
    if strategy == 'density_only':
        for param in model.temporal.parameters():
            param.requires_grad = False
        for param in model.skeleton_head.parameters():
            param.requires_grad = False
    elif strategy == 'skeleton_and_density':
        for param in model.temporal.parameters():
            param.requires_grad = False
    # strategy == 'full': 全部可训练

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)")
```

### 3.4 训练

```bash
# Step 1: 在仿真上预训练
python scripts/training/train_ms_scnf.py \
    --data_dir data/sequence_data_3d \
    --save_dir train_log/ms_scnf_pretrain

# Step 2: 在真实数据上微调（只训练密度场）
python scripts/training/finetune_real.py \
    --pretrained train_log/ms_scnf_pretrain/phase2/model/best_model.pt \
    --data_dir data/real_data/session_001 \
    --strategy density_only \
    --n_epochs 50 \
    --lr 1e-4 \
    --save_dir train_log/ms_scnf_finetune_density

# Step 3: 微调骨架+密度（如果 Step 2 不够）
python scripts/training/finetune_real.py \
    --pretrained train_log/ms_scnf_pretrain/phase2/model/best_model.pt \
    --data_dir data/real_data/session_001 \
    --strategy skeleton_and_density \
    --n_epochs 50 \
    --lr 5e-5 \
    --save_dir train_log/ms_scnf_finetune_skeleton
```

---

## 4. 策略 C: Domain Adaptation (GAN-based)

### 4.1 思路

使用 CycleGAN 或 Pix2Pix 学习仿真图像 ↔ 真实图像的映射。

```
方案 1: Sim → Real 图像翻译
  仿真图像 → Generator → "真实风格"图像
  在翻译后的数据上训练模型

方案 2: 特征级域适应
  共享的时序/骨架编码器
  域判别器 (Domain Discriminator) → 对抗训练
  使特征不区分仿真/真实
```

### 4.2 模型设计

```python
class DomainAdaptiveMSCNF(MSSCNFModel):
    """带域适应的 MS-SCNF。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 域判别器：判断特征来自仿真还是真实
        self.domain_classifier = nn.Sequential(
            nn.Linear(kwargs['hidden_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def domain_adversarial_loss(self, physics_state, domain_label):
        """
        Args:
            physics_state: (B, hidden_dim) — 时序编码器输出
            domain_label: 0=仿真, 1=真实
        """
        pred_domain = self.domain_classifier(physics_state.detach())
        target = torch.full_like(pred_domain, domain_label)
        return F.binary_cross_entropy(pred_domain, target)
```

### 4.3 训练

```bash
# 需要同时有仿真和真实数据
python scripts/training/train_domain_adaptive.py \
    --sim_data data/sequence_data \
    --real_data data/real_data/session_001 \
    --n_epochs 200 \
    --lambda_domain 0.1 \
    --save_dir train_log/ms_scnf_da
```

---

## 5. 系统评估框架

### 5.1 迁移质量指标

```python
def evaluate_transfer(sim_model, real_data_loader, device):
    """评估 sim→real 迁移质量。"""

    metrics = {
        'render_psnr': [],      # 渲染质量
        'render_ssim': [],      # 结构相似性
        'skeleton_smooth': [],  # 骨架物理合理性
    }

    for batch in real_data_loader:
        action_window = batch[0].to(device)
        gt_img = batch[1]

        with torch.no_grad():
            # 渲染预测
            pred_dict = sim_model.predict_skeleton(action_window)
            # ... 计算指标

    return {k: np.mean(v) for k, v in metrics.items()}
```

### 5.2 渐进式评估

```bash
# 1. 仿真测试集 (baseline)
python scripts/evaluation/evaluate_3d.py \
    --checkpoint train_log/ms_scnf_pretrain/phase2/model/best_model.pt \
    --data_dir data/sequence_data_3d

# 2. 仿真域随机化
python scripts/evaluation/evaluate_3d.py \
    --checkpoint train_log/ms_scnf_dr/model/best_model.pt \
    --data_dir data/sequence_data_3d

# 3. 真实数据 (无微调)
python scripts/evaluation/evaluate_real.py \
    --checkpoint train_log/ms_scnf_pretrain/phase2/model/best_model.pt \
    --real_data data/real_data/session_001

# 4. 真实数据 (微调后)
python scripts/evaluation/evaluate_real.py \
    --checkpoint train_log/ms_scnf_finetune_density/model/best_model.pt \
    --real_data data/real_data/session_001
```

---

## 6. 实现文件清单

| 文件 | 用途 |
|------|------|
| `src/utils/domain_randomization.py` | 渲染/物理随机化 |
| `scripts/data/collect_dr.py` | 域随机化数据采集 |
| `scripts/training/finetune_real.py` | 真实数据微调 |
| `scripts/training/train_domain_adaptive.py` | 域适应训练 |
| `scripts/evaluation/evaluate_real.py` | 真实数据评估 |
| `src/models/model_ms_scnf_da.py` | 域适应模型 |
| `notebooks/14_sim2real_eval.ipynb` | 评估 notebook |

---

## 7. 验证 Notebook

`14_sim2real_eval.ipynb`:
```
1. Sim-Real 差距可视化
   - 仿真图像 vs 真实图像对比
   - 渲染风格差异分析

2. Domain Randomization 效果
   - 不同随机化强度的训练曲线
   - 在真实数据上的泛化能力

3. Fine-tuning 策略对比
   - density_only vs skeleton_and_density vs full
   - 微调前后的渲染质量变化
   - 微调所需数据量分析（10/50/100/500 帧）

4. 域适应效果
   - 特征分布可视化（t-SNE）
   - 域判别器准确率（越低越好 = 特征越域不变）

5. 渐进式评估
   - Pretrain → DR → Fine-tune → DA
   - 每步的性能提升
```

---

## 8. 创新点

1. **首个面向软体机器人自建模的 sim-to-real 系统**
2. **分层冻结微调策略**：利用软体机器人物理结构的层次性
3. **域随机化 + 微调的组合方案**

---

## 9. 依赖关系

- **方向 3** (多相机系统) 是数据采集的前提
- **方向 2** (纯 2D) 的训练框架是微调的基础
- 可以先在仿真中做 DR 实验（不需要真实硬件）

---

## 10. 时间规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| Week 1-2 | 仿真 DR | 域随机化实现 + 训练 + 评估 |
| Week 3-4 | 微调策略 | 在仿真→仿真变体上验证微调策略 |
| Week 5-6 | 真实数据 | 结合方向 3 采集真实数据 |
| Week 7-8 | 真实迁移 | 真实数据微调 + 域适应 |
