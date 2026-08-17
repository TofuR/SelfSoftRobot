# 归档清单 · 2026-07-28

> **为什么归档**:方法路线已收敛到**状态转移族**,特别是 `open_loop_transition`(部署主线);`gt_transition` 保留作论文消融。其余模型族(C-MSTNF / MS-SCNF / SDF / SkeletonSDF / FlowMatch / SpatialSequence 等)已完成历史作用,连同实验日志一起归档,以简化项目目录。
>
> **本轮范围**:只动 `train_log/`(gitignored,纯磁盘移动)与文档状态标记。**`src/` 代码一律不动** —— 见 §4。
>
> **回溯路径**:归档目录移动到 `train_log/_archive/<原名>/`,路径可逆;本文件是唯一权威清单。

---

## 1. 保留(仍是活的主线)

| train_log 目录 | 模型 | 定位 |
|---|---|---|
| `open_loop_transition/` | `OpenLoopTransitionModel` | **部署主线**。最好实验 `exp_20260714_8`(best_loss 0.080,全套 eval);`exp_20260716_9` 是 window=1 消融臂 |
| `gt_transition/` | `GTObservedTransitionModel` | **论文消融 + 精度上界**。最好实验 `exp_20260714_7`(best_loss 0.00077);`exp_20260716_8` 是 window=1 消融臂 |

对应代码(**不归档**):`src/models/model_state_transition.py`(基类)、`model_gt_transition.py`、`model_open_loop_transition.py`;编码器 `src/encoders/fractional_memory.py`;数据集 `src/data/dataset_spatial.py`。

---

## 2. 归档到 `train_log/_archive/`

| train_log 目录 | 对应模型 / 文件 | 归档理由 |
|---|---|---|
| `train_mstnf/` | MSTNF · `src/models/model_mstnf.py` | 单相位体渲染,已被 C-MSTNF 取代 |
| `train_cmstnf/` | C-MSTNF · `model_cmstnf.py` | D-NeRF 范式典范+变形,已被 MS-SCNF 取代 |
| `train_ms_scnf/`、`ms_scnf/` | MS-SCNF · `model_ms_scnf.py` | 仿真主线终点;已被状态转移族取代(它是前馈 action→state,不解决迟滞) |
| `train_sdf/` | TemporalSDF · `model_sdf.py` | 直接 3D SDF 监督,仿真专用 |
| `train_skeleton_sdf/` | SkeletonSDF · `model_skeleton_sdf.py` | 同上,两阶段版 |
| `train_ode_cmstnf/` | ODE-CMSTNF | 方向文档已在 `docs/archived/ode_cmstnf/` |
| `train_smooth_cmstnf/` | Smooth-CMSTNF | 方向文档已在 `docs/archived/smooth_cmstnf/` |
| `train_unified/` | UnifiedTrainer 早期通用输出 | 无独立模型语义 |
| `train_log_seq_vis/` | 序列可视化中间产物 | 非训练日志 |
| `spatialsequence/` | `SpatialSequenceModel` · `model_spatial_sequence.py` | 前馈基线,已被状态转移族取代 |
| `pcspatialsequence/` | `PCSpatialSequenceModel` · `model_pc_spatial.py` | 预测-修正两阶段;是唯一用图像作残差修正输入的模型(模型输入约定的例外) |
| `flowmatchpointcloud/` | `FlowMatchPointCloudModel` · `model_flowmatch_pointcloud.py` | 点云 flow matching 探索 |
| `state_transition/`、`state_transition_s1/` | `StateTransitionSpatialModel`(方向 13) | 纯自回归**无界** rollout 对照,实测漂移 1170×,不可用;**基类本身保留**(gt/open_loop 继承它) |

### 2.1 归档但标"可能复活" —— 多视角三条

| train_log 目录 | 为什么不深埋 |
|---|---|
| `train_multiview_cmstnf/` | **2026-07-28 决定 3D 多自由度走"双/多相机标定 + 三角化"路线**,这三个是仓库里唯一的多相机训练先例。它们用的是**体渲染多视角一致性**(不是三角化),但相机系统、标定文件格式、多视角数据集类都可复用 |
| `train_multiview_consistency_cmstnf/` | 同上(方案 B:跨视角一致性 + 重投影) |
| `train_multiview_mstnf/` | 同上 |

复用时对应代码:`src/utils/camera_system.py`(`MultiCameraSystem`)、`src/data/dataset_multiview.py`、`dataset_multiview_depth.py`、`src/rendering/view_strategy.py`(`MultiViewStrategy`)、`src/data/real/triangulation.py`、`scripts/real/calibrate_cameras.py`、`scripts/real/capture_to_npz.py --view-dirs`。

---

## 3. 文档状态重标(不是归档)

2026-07-28 的三条决策改变了若干方向的优先级,因此 `docs/directions/` **不做整体归档**,而是重标状态:

| 方向 | 原状态 | 新状态 | 原因 |
|---|---|---|---|
| [06 多视角 2D→3D 骨架](../directions/06_multi_view_2d_to_3d_skeleton.md) | ★★☆ | **★★★ 主线** | 3D 多自由度的状态来源就是它 |
| [08 单 DOF 分解与组合](../directions/08_per_dof_decomposition.md) | ★☆☆ | **★★★ 主线** | 6 通道驱动直接需要:同段腔道竞争 + 跨段耦合 + 组合泛化 |
| [15 窗口开环](../directions/15_open_loop_windowed_transition.md) | 主线 | 主线(不变) | 部署主线 |
| [14 全 GT 驱动](../directions/14_gt_observed_transition.md) | 诊断 | **论文消融** | 明确其论文角色 |
| [13 闭环状态转移](../directions/13_closed_loop_state_transition.md) | Stage0 实现 | **已被 14/15 取代**(基类文档保留) | 无界 rollout 不可用 |
| [07 从轮廓恢复形状](../directions/07_shape_from_silhouette.md) | ★★☆ | **搁置** | 与 05 竞争同一目标,05 已有 IoU 0.91 基线 |
| [09 拓扑引导残差流](../directions/09_topology_guided_residual_flow.md) | ★★☆ | **搁置** | 仿真路线遗留 |
| [11 Sim-to-Real](../directions/11_sim_to_real_transfer.md) | ★★☆ | **搁置** | 实物已直接采数训练,不再需要迁移 |

**⚠️ 一条不变量被反转**:`docs/HANDOFF.md` §7.2 不变量 #7 原文"**免标定**……`calibrate_cameras.py`/`capture_to_npz.py`/`inspect_capture.py` 是**遗留标定路线,别用于路线 B**"。2026-07-28 决定 3D 走多相机标定三角化后,这条**反转** —— 那些代码从"遗留"变为主线基础设施。

连带需要修订(尚未做,记录在此以免遗忘):
- `docs/HANDOFF.md` §7.2 不变量 #7、§2 的路线对照表
- spec `docs/superpowers/specs/2026-07-28-real-validation-task-layer-ik-design.md` §1.2(3D/多相机从非目标变中期目标)、§3.3(四坐标系:`camera_pixel → model` 不再是恒等映射,而是 per-view 像素 → 三角化 3D)、§6.3("障碍是平面近似"可升级为真 3D)、§8(采集协议要加标定与多相机)
- `pc_scale[2] = 1e-6` 的退化保护不再成立(真 3D 数据下 z 有真实量程);**旧 checkpoint 的 buffer 是 1e-6,不能与 3D 数据混用**
- `docs/papers/related_work_draft.md` §2.7 的"免标定"差异化论点要重写

**另一条决策的残留风险**:mask 源定为"在线跑 SAM2 前向流式"。训练用的是**双向分块**传播且锚帧来自启发式修复的干净帧,在线**没有这个锚帧来源**,故前向流式与双向的差异**仍未量化**;GPU 单帧延迟也未实测(采集节拍 0.2 s 是预算)。两条都记为已知缺口,不假装解决。

---

## 4. 为什么 `src/` 代码本轮不动

归档代码会碰到**四处按 model type 分派的地方**,任一处漏改都会静默打断在留的状态转移族:

| 分派点 | 风险 |
|---|---|
| `src/training/trainer_unified.py` | 按 `training_spec` 解释相位;`supervision_mode` 分派两处必须一致(HANDOFF 不变量 #15) |
| `src/training/dataset_factory.py` | 按 spec 造数据集 + collate |
| `src/utils/model_loader.py` | 按 state_dict key 自动识别模型类型;`_migrate_gru_keys` **只对 state_transition 族**做迁移,对 `spatial_sequence` 不能做(不变量 #1) |
| `src/config/args.py` | CLI `--model` 选项枚举 |

另有两个继承关系,**2026-07-28 已查清**(结论直接决定未来能归档什么):

### 4.1 `StateTransitionSpatialModel` **不**继承 `PCSpatialSequenceModel` ✅

`src/models/model_state_transition.py:60`:
```python
class StateTransitionSpatialModel(nn.Module, TemporalMixin):
```
且 docstring 第 29 行明说"**不修改 SpatialSequenceModel / PCSpatialSequenceModel,互不影响**"。

→ **`model_pc_spatial.py` 与 `model_spatial_sequence.py` 未来可以安全归档**,不会打断主线。(两者各自注册 `pc_center`/`pc_scale` buffer 是**平行实现**,不是继承。)

### 4.2 主线只用 `TemporalMixin`,但 `mixins.py` 整个文件不能动 ⚠️

`SkeletonMixin` 未被状态转移族使用(它服务 MS-SCNF / SkeletonSDF),但它与 `TemporalMixin` 同在 `src/models/mixins.py`,且该文件**模块级** import 了 `from src.heads.skeleton_heads import downsample_skeleton`(第 28 行)。

→ **`src/models/mixins.py` 与 `src/heads/` 都必须保留**,即使 `SkeletonMixin` 本身是死代码。若要归档 `src/heads/`,得先把这个 import 移进 `SkeletonMixin` 的方法体内。

### 4.3 主线的完整依赖面(实测 import 清单)

`model_state_transition.py:44-57` 的 import 决定了**哪些看似废弃的模块其实不能动**:

| 依赖 | 能否归档 |
|---|---|
| `src/encoders/{fractional_memory, multi_scale_ema, gamma_laguerre, temporal_gru, temporal_transformer, temporal_tcn}.py` | ❌ **6 个编码器全部被 import**(即使只用 fractional),`--encoder` CLI 选项也依赖它们 |
| `src/training/spec.py` | ❌ `TrainingSpec` / `PhaseSpec` |
| `src/models/mixins.py` → `src/heads/skeleton_heads.py` | ❌ 见 §4.2 |
| **`src/data/dataset_pointcloud.py`** | ❌ **意外依赖**:`model_state_transition.py:57` 有 `from src.data.dataset_pointcloud import _sample_surface`。所以尽管 `flowmatchpointcloud/` 日志归档了,这个数据集文件**不能**跟着归档 |

### 4.4 一个潜伏(非当前)缺陷

`mixins.py:97` 的 `TemporalMixin.compute_losses` 与 `:78` 的 `get_learned_decays` 都访问 `self.temporal.decays`,而 `FractionalMemory` 只有 `raw_alphas`,**没有 `decays`** → 用 fractional 编码器且 `active_losses` 含 `"smooth"` 时会 `AttributeError`。当前 `open_loop_transition` 的 `active_losses` 是 `["skeleton", "spatial_smooth"]`(不含 `smooth`),故该分支未被走到。**潜伏而非现存**,记录以免日后加 loss 时踩到。

且 3D 路线可能需要**复活** multiview / 三角化那部分代码,现在删了要再捡回来。

---

## 5. 执行清单

```bash
cd /Data5/ddf/projects/SelfSoftRobot
mkdir -p train_log/_archive
du -sh train_log/*/ | sort -rh > /tmp/train_log_sizes.txt   # 规模记录,填进 §6

for d in train_mstnf train_cmstnf train_ms_scnf ms_scnf train_sdf train_skeleton_sdf \
         train_ode_cmstnf train_smooth_cmstnf train_unified train_log_seq_vis \
         spatialsequence pcspatialsequence flowmatchpointcloud \
         state_transition state_transition_s1 \
         train_multiview_cmstnf train_multiview_consistency_cmstnf train_multiview_mstnf; do
  [ -d "train_log/$d" ] && mv "train_log/$d" "train_log/_archive/$d" && echo "moved $d"
done

ls train_log/    # 验收:_archive  gt_transition  open_loop_transition
```

**验收标准**:`train_log/` 顶层只剩 `_archive/`、`gt_transition/`、`open_loop_transition/`;`_archive/` 下 18 个目录;`git status` **无变化**(`train_log/` 已 gitignored)。

---

## 6. 归档时的实际规模(实测 2026-07-28)

`train_log/` 总计 **1.3 G**;保留的两项只占 **27 M**,归档约 **1.27 G**。

| 目录 | 大小 | 处置 |
|---|---|---|
| `train_ms_scnf/` | **424 M** | 归档 |
| `train_cmstnf/` | **248 M** | 归档 |
| `train_smooth_cmstnf/` | 125 M | 归档 |
| `train_ode_cmstnf/` | 108 M | 归档 |
| `train_log_seq_vis/` | 97 M | 归档 |
| `train_mstnf/` | 77 M | 归档 |
| `train_sdf/` | 68 M | 归档 |
| `flowmatchpointcloud/` | 24 M | 归档 |
| **`open_loop_transition/`** | **19 M** | **保留(部署主线)** |
| `train_multiview_consistency_cmstnf/` | 14 M | 归档(标可能复活) |
| `train_multiview_cmstnf/` | 9.1 M | 归档(标可能复活) |
| `ms_scnf/` | 8.8 M | 归档 |
| `train_skeleton_sdf/` | 8.0 M | 归档 |
| **`gt_transition/`** | **8.0 M** | **保留(论文消融)** |
| `pcspatialsequence/` | 6.5 M | 归档 |
| `spatialsequence/` | 5.9 M | 归档 |
| `state_transition_s1/` | 2.1 M | 归档 |
| `state_transition/` | 2.1 M | 归档 |
| `train_unified/` | 1.9 M | 归档 |
| `train_multiview_mstnf/` | 36 K | 归档(标可能复活) |

前两项(`train_ms_scnf` + `train_cmstnf` = 672 M)就占了一半以上 —— 它们是仿真主线的两代模型,归档收益最大。
