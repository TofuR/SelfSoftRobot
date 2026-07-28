# real_validation P1a(M0 感知迁移)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把在线感知(分割 + 骨架 + 背景 + 位姿注册 + 质量门控)做成 `real_validation/perception/` 下的唯一实现,让离线管线通过薄壳复用它,并交付一个**在没有相机的开发机上就能跑**的命令行感知探针。

**Architecture:** 感知实现从 `src/` 移入 `real_validation/perception/`(部署产物持有实现),`src/utils/skeleton_2d.py` 与 `src/data/real/segmentation.py` 改成薄壳 re-export(签名不变)。在线与离线由**构造**保证一致,再用冻结参考 parity 测试与 import 卫生测试把它锁死。位姿注册只做**检测**不做 warp。

**Tech Stack:** Python 3.10+、numpy、opencv-python(cv2,ORB 在主包)、scipy(`binary_fill_holes`)、unittest。**本计划全程不涉及 torch。**

**上游 spec:** [`../specs/2026-07-28-real-validation-task-layer-ik-design.md`](../specs/2026-07-28-real-validation-task-layer-ik-design.md) §12 的 P1,拆分后的**前半**(M0)。

> **范围拆分说明(2026-07-28)**:spec §12 的 P1 = M0 + M3,但两者**互不共享代码**(M0 纯 numpy/cv2 感知;M3 纯 torch/dataclass 契约与 planner),各自独立可测,且 **P2 的数据采集只被 M0 阻断**(采集必须用与在线一致的分割参数)。故拆为:
> - **P1a(本计划)** = M0,7 个任务,交付感知唯一实现 + 探针
> - **P1b(姊妹计划 `2026-07-28-real-validation-p1b-contracts-and-units.md`)** = M3,交付 B1–B5 / B7–B11 / **B14–B17** 的契约与单位修复 + T2/T3/T4a/T4b/T7/T8
>
> 两者可并行,无相互依赖。

## Global Constraints

- 分支固定 `feat/real-data-transition`。**不切分支、不新建 worktree。**
- **向后兼容是硬性要求**:不得破坏 `src/`、`scripts/`、`notebooks/` 中任何现有调用签名。移动实现只能通过薄壳 re-export。
- `real_validation/` **不得 import `src/`**(可移植契约)。反方向允许:`src/` 与 `scripts/` 可以 import `real_validation.*`。
- `real_validation/perception/` 只依赖 numpy / cv2 / scipy,**不得 import torch**。
- `real_validation/__init__.py` 必须保持只依赖标准库(现状:`models`/`session`/`io`/`preflight` 全为标准库)。新增子包不得改变这一点。
- 测试框架:`unittest`(仓库无 pytest,**计划中禁止出现 pytest 命令**)。基线:`cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core -v` → **期望 20 passed**(实测行号 47/53/61/72/78/93/105/117/127/135/142/148/157/164/179/185/220/255/278/291)。
- **现有 20 个测试的行为断言必须保持通过**;但本计划**会修改 `tests/test_real_validation_core.py` 的 `fixtures()`(第 28-43 行)** 以补齐新契约字段 —— 这是有意为之(见 Task 7 的 fail-closed 裁决),**不承诺"零测试改动"**。
- `validate_plan(plan, model, anchor, scene, safety)` 有 **7 处位置参数调用**(测试 55/58/68/75/252 行 + `session.py:159` `accept_plan` + `session.py:175` `arm`)。新参数**一律 keyword-only + 默认值**,签名前 5 位不得变。
- 纯 numpy 的感知测试**不得加进 `tests/test_real_validation_core.py`**(那个文件已依赖 torch/PyQt5),另开文件。
- **测试基线数据只能用 `.npy` / `.json` / `.png`,不能用 `.npz`** —— 根 `.gitignore` 含裸行 `*.npz`,fixtures 无法提交。
- 提交信息用 Conventional Commits,中英混排可,**禁止 `Co-Authored-By`**。
- **每次 `git commit` 前必须询问用户**(用户硬性偏好)。计划中的 "Commit" 步骤 = 先询问、获准后提交。
- 数值常量(实测,不得改动):`mm_per_px = 0.302`;`hi6[0] = 150.0 kPa`;`n_nodes = 15`;`window_size = 40`;分割参数取自 `derived/<seq>/segment_meta.json`(`sat=100, val=100, diff=25, dil=35, open_k=5, close_k=15, min_area_frac=0.003, min_h_frac=0.15`)—— **注意代码默认是 `val=120`,批产用的是 `val=100`,以 segment_meta.json 为准**。
- **`train_dt` 禁止硬写 `0.203125`** —— 仓库里没有任何文件记录这个数,它只能由生产者对 `frame_times.txt` 现算(`np.diff` 均值/标准差)。计划里只允许出现 `train_dt_nominal_s = 0.2`(来自 `meta.json` 的 `action_interval_s`)与"由生产者实测"两种表述。
- **依赖现状(实测)**:根 `requirements.txt:20` 已有 `opencv-python==4.11.0.86` → `ORB_create` / `BFMatcher` / `findHomography(RANSAC)` / `estimateAffinePartial2D` **全在主包,不需要 opencv-contrib**;但根 `requirements.txt` **缺 `scipy`**(`segmentation.py:95`、`repair_masks.py:27` 是隐式依赖),需补。
- `real_validation/requirements.txt` 现在只有 4 行(numpy/torch/PyQt5/pyqtgraph),**新增 `real_validation/requirements-perception.txt`**(opencv-python + scipy)而不是往主文件塞,并同步改 `real_validation/README.md:20-28`。
- **`real_validation/README.md:43-47` 是现存的坏指令**(它让操作员在 PC 上跑 `python -m unittest tests.test_real_validation_core`,而 PC 上只拷 `real_validation/`,没有 `tests/`)→ 本计划必须修掉。

## 依赖数据(本机已有,离线可用;全部被 gitignore)

| 路径 | 用途 |
|---|---|
| `real_capture/data/raw/seq_20260627_163921/cam0/*.png`(10214 帧 640×480) | 探针 `--source dir` 输入 |
| `real_capture/data/derived/seq_20260627_163921/bg_median.png` | 背景基准 |
| `real_capture/data/derived/seq_20260627_163921/segment_meta.json` | 真实分割参数(`val=100`) |
| `real_capture/data/derived/seq_20260627_163921/masks/*.png` | 可选 parity 输入 |
| `train_log/open_loop_transition/exp_20260714_8/config.json` | manifest 生成源 |
| `.../exp_20260714_8/eval_horizon/horizon_summary.json` | `k_safe_table` 源(`Kmax_px_5=51`、`Kmax_px_10=124`) |

**因此所有提交的测试必须能在没有这些数据时通过**(用合成数据);真实数据只作可选增强(`@unittest.skipUnless(path.exists())`)。

## File Structure

**新建(P1a)**

| 文件 | 责任 |
|---|---|
| `real_validation/perception/__init__.py` | 空文件(零副作用,避免连带 import cv2) |
| `real_validation/perception/skeleton.py` | 逐行质心 + 弧长重采样 + tip_fix(唯一实现,仅 numpy) |
| `real_validation/perception/segmentation.py` | 4 种分割方法 + 中值背景(唯一实现) |
| `real_validation/perception/background.py` | 背景加载/重建/漂移检测 |
| `real_validation/perception/registration.py` | ORB+RANSAC 位姿注册与位移量化(只检测,不 warp) |
| `real_validation/perception/quality.py` | 在线单帧质量门控(机制;数据相关阈值无默认值) |
| `real_validation/perception_probe.py` | 命令行感知探针(`--source dir|live`) |
| `real_validation/requirements-perception.txt` | opencv-python + scipy |
| `tests/test_perception_parity.py` | 冻结参考对比(T1)+ tip_fix 可观测性 |
| `tests/test_import_hygiene.py` | import 卫生锁(T9) |
| `tests/test_perception_registration.py` | 背景漂移 + 位姿注册 |
| `tests/test_perception_quality.py` | 质量门控各判据 |

**修改(P1a)**

| 文件 | 改什么 |
|---|---|
| `src/utils/skeleton_2d.py` | 三个骨架函数改薄壳;`project_3d_to_2d`/`compute_2d_skeleton_loss` **留在原处**(需 torch) |
| `src/data/real/segmentation.py` | 全部 8 个公开函数改薄壳 |
| `real_validation/__init__.py` | docstring 写死 stdlib-only 约束 |
| `real_validation/requirements.txt` | 加一行指向 requirements-perception.txt |
| `requirements.txt`(根) | 补 `scipy`(现在是隐式依赖) |
| `real_validation/README.md:20-28, 43-47` | 补感知依赖段;**修掉"在 PC 上跑 tests/"的坏指令** |
| `scripts/real/write_data_readme.py:159` | 硬编码的 `"## 骨架化(src/utils/skeleton_2d.py)"` 改指新路径(它是 `data/real_seq/README.md` 的生成器,不改则重跑会写回旧路径) |
| `src/data/real/triangulation.py:5,7,52` | 三处 docstring 里的 skeleton_2d 路径 |
| `real_capture/data/derived/<seq>/README.md:13` | 顺手修坏路径(写 `scripts/real/segment_rd/segment_batch.py`,实际是 `scripts/real/segment_batch.py`) |
| `CLAUDE.md` | "无正式测试框架"一句改为现状(4 个测试文件 + 运行命令) |

> `src/calibration/camera_params_format.py:4,8,51` **不用改** —— 它引用的是 `project_3d_to_2d`,而该函数留在 `src/`。

**P1a 明确不做**(全部归 P1b 或更后)

| 项 | 归属 |
|---|---|
| `units.py` / `obstacles.py` / `deploy_manifest.py` / `ui_state.py` / `scripts/utils/build_deploy_manifest.py` | P1b |
| B1–B5、B7–B11、B14–B17 契约与单位修复 | P1b |
| `perception/live_anchor.py` | P3(M4)—— 需要真 `pc_center/pc_scale`,即需要 checkpoint |
| 多起点批并行 | **不做**:`clip_grad_norm_` 是全局范数、cuDNN GRU 在 batch=1 与 batch=R 走不同 kernel → **无法与既有结果逐位一致**。价值取决于 P1b 的耗时基准 |

---

### Task 1: 骨架实现迁移 + 冻结参考 parity 测试

把 `extract_skeleton_2d` / `batch_extract_skeleton_2d` / `_perpendicular_tip_fix` 迁到 `real_validation/perception/skeleton.py`。parity 用**冻结在测试文件里的旧实现副本**做对照——测试自包含、可提交、永久防漂移(不依赖 gitignored 数据或 git 历史)。

**Files:**
- Create: `real_validation/perception/__init__.py`
- Create: `real_validation/perception/skeleton.py`
- Create: `tests/test_perception_parity.py`
- Modify: `src/utils/skeleton_2d.py:1-120`(改薄壳,保留 `project_3d_to_2d` / `compute_2d_skeleton_loss`)

**Interfaces:**
- Consumes: 无(首个任务)
- Produces:
  - `real_validation.perception.skeleton.extract_skeleton_2d(binary_img, n_points=31, tip_fix=False)` → `np.ndarray (n_points,2) float32`
  - `real_validation.perception.skeleton.batch_extract_skeleton_2d(images, n_points=31, tip_fix=False)` → `np.ndarray (T,n_points,2) float32`
  - `real_validation.perception.skeleton._perpendicular_tip_fix(skeleton, binary_img, n_points)` → `np.ndarray`
  - `src.utils.skeleton_2d` 继续导出上述三名 + 原有 `project_3d_to_2d` / `compute_2d_skeleton_loss`

> **迁移的四条不可协商约束**(来自向后兼容穷尽枚举,13 个 import 点):
> 1. **禁止 `from ... import *`** —— 会丢 `_perpendicular_tip_fix`(`CLAUDE.md:96`、`docs/HANDOFF.md:280`、`docs/overview/project_help.md:190` 都记录在此路径),且对签名漂移零防护。必须显式命名 import + 显式 `__all__`。
> 2. **`tip_fix` 默认必须保持 `False`** —— `scripts/real/compare_skeleton_methods.py:81,187,226,420` 有 5 处省略该参数,靠默认 False 充当 **M0 未修基线**;`src/data/dataset_multiview.py:70-72` 同样依赖 False。改成 True 会让 tip_fix 的 -71% 结论变成 fix-vs-fix 自比,**且不报错**。`masks_to_skeletons_2d` 的 `tip_fix` 默认必须保持 `True`(与前者相反,两个都要原样)。
> 3. **返回值必须保持裸 `ndarray`** —— `scripts/real/masks_to_transition_npz.py:91` 有 `T, N, _ = sk2d.shape`;`composite_frames.py:134`、`segmentation.py:169`、`viz_qc.py:117` 都是 `out[i] = ...`。诊断标志只能走 opt-in kwarg 或独立函数(Task 2 即如此)。
> 4. **`perception/__init__.py` 必须保持空** —— 反面教材 `src/data/real/__init__.py:10` eager import 全部子模块;若照抄,`masks_to_transition_npz.py`(只需 numpy+cv2)与全部仿真训练都会被迫拉 scipy/ORB。
>
> 另注:`project_3d_to_2d` / `compute_2d_skeleton_loss` **不迁移**(只服务路线 A 仿真多视角标定:`exp7_multiview_2d_skeleton.py:84`、`exp7_3d_occupancy.py:375`、`_smoke_triangulation.py:76`;路线 B 免标定不需要可微投影)。特别注意 `_smoke_triangulation.py:73-90` 把 `ImportError` 吞进 `except` 只打印"torch 不可用,跳过" —— **漏 re-export 这里会静默失活而非崩溃**,Step 7 必须显式验证。

- [ ] **Step 1: 建空的 perception 包**

创建 `real_validation/perception/__init__.py`,**只有一行 docstring,不得有任何 import**(否则 `import real_validation.perception.skeleton` 会连带拉入 cv2/scipy):

```python
"""在线感知的唯一实现。子模块按需单独 import，本文件保持零副作用。"""
```

- [ ] **Step 2: 写 parity 失败测试**

创建 `tests/test_perception_parity.py`:

```python
"""感知迁移的行为冻结测试。

_legacy_* 是迁移前 src/utils/skeleton_2d.py 与 src/data/real/segmentation.py 的
逐行副本，作为永久对照基线。任何对 real_validation/perception/ 的修改若改变输出，
本测试立即失败。合成输入保证测试自包含（仓库数据目录被 gitignore）。
"""

import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 冻结的旧实现
def _legacy_perpendicular_tip_fix(skeleton, binary_img, n_points):
    sk = skeleton.astype(np.float64)
    if n_points < 5 or np.abs(sk).max() == 0:
        return skeleton
    ys, xs = np.where(binary_img > 0.5)
    if len(xs) < 10:
        return skeleton
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    far = sk[min(max(2, int(0.25 * n_points)), n_points - 1)]
    near = sk[min(max(1, int(0.10 * n_points)), n_points - 1)]
    seg = near - far
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return skeleton
    d = seg / L
    proj = (pts - far) @ d
    w = float(binary_img.sum(1).max())
    slab = proj >= proj.max() - 0.4 * w
    if int(slab.sum()) < 3:
        return skeleton
    node0 = pts[slab].mean(0)
    sk[0] = node0
    a = sk[min(3, n_points - 1)]
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32)


def _legacy_extract_skeleton_2d(binary_img, n_points=31, tip_fix=False):
    H, W = binary_img.shape
    coords = []
    for row in range(H - 1, -1, -1):
        white_cols = np.where(binary_img[row] > 0.5)[0]
        if len(white_cols) > 0:
            coords.append([white_cols.mean(), float(row)])
    if len(coords) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    coords = np.array(coords, dtype=np.float32)
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)
    target_lens = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lens, cum_len, coords[:, 0])
    resampled[:, 1] = np.interp(target_lens, cum_len, coords[:, 1])
    if tip_fix:
        resampled = _legacy_perpendicular_tip_fix(resampled, binary_img, n_points)
    return resampled


# ---------------------------------------------------------------- 合成 mask
def synthetic_masks():
    """覆盖全部代码路径的确定性合成 mask，返回 [(name, (H,W) uint8), ...]。"""
    cases = []

    cases.append(("empty", np.zeros((64, 48), np.uint8)))

    one_row = np.zeros((64, 48), np.uint8)
    one_row[40, 20:26] = 1
    cases.append(("single_row", one_row))

    straight = np.zeros((120, 80), np.uint8)
    straight[20:110, 36:44] = 1
    cases.append(("straight_tube", straight))

    bent = np.zeros((120, 80), np.uint8)
    for row in range(20, 110):
        left = 34 + int(round(0.12 * (row - 20) ** 1.35 / 10.0))
        bent[row, left:left + 9] = 1
    cases.append(("bent_tube", bent))

    tilted_cap = bent.copy()
    for offset in range(6):
        row = 109 - offset
        tilted_cap[row, :] = 0
        left = 34 + int(round(0.12 * (row - 20) ** 1.35 / 10.0)) + offset
        tilted_cap[row, left:left + max(1, 9 - offset)] = 1
    cases.append(("tilted_cap", tilted_cap))

    tiny = np.zeros((64, 48), np.uint8)
    tiny[30:33, 20:23] = 1
    cases.append(("tiny_blob", tiny))

    edge = np.zeros((120, 80), np.uint8)
    edge[10:118, 0:7] = 1
    cases.append(("touching_edge", edge))

    return cases


class SkeletonParityTest(unittest.TestCase):
    def test_matches_frozen_reference_on_synthetic_masks(self):
        from real_validation.perception.skeleton import extract_skeleton_2d

        for name, mask in synthetic_masks():
            for n_points in (15, 31):
                for tip_fix in (False, True):
                    tag = f"{name} n={n_points} tip_fix={tip_fix}"
                    expected = _legacy_extract_skeleton_2d(mask, n_points, tip_fix=tip_fix)
                    actual = extract_skeleton_2d(mask, n_points, tip_fix=tip_fix)
                    self.assertEqual(actual.shape, expected.shape, tag)
                    self.assertEqual(actual.dtype, expected.dtype, tag)
                    self.assertTrue(np.array_equal(actual, expected), tag)

    def test_batch_matches_frozen_reference(self):
        from real_validation.perception.skeleton import batch_extract_skeleton_2d

        masks = np.stack([mask for _, mask in synthetic_masks()
                          if mask.shape == (120, 80)])
        expected = np.stack([_legacy_extract_skeleton_2d(m, 15, tip_fix=True)
                             for m in masks])
        actual = batch_extract_skeleton_2d(masks, 15, tip_fix=True)
        self.assertTrue(np.array_equal(actual, expected))

    def test_shim_reexports_same_objects(self):
        import src.utils.skeleton_2d as shim
        from real_validation.perception import skeleton as canonical

        self.assertIs(shim.extract_skeleton_2d, canonical.extract_skeleton_2d)
        self.assertIs(shim.batch_extract_skeleton_2d, canonical.batch_extract_skeleton_2d)
        self.assertIs(shim._perpendicular_tip_fix, canonical._perpendicular_tip_fix)
        self.assertTrue(callable(shim.project_3d_to_2d))
        self.assertTrue(callable(shim.compute_2d_skeleton_loss))


REAL_MASKS = REPO / "real_capture/data/derived/seq_20260627_163921/masks"


@unittest.skipUnless(REAL_MASKS.is_dir(), "真实 mask 目录不存在（已 gitignore）")
class SkeletonParityOnRealMasksTest(unittest.TestCase):
    def test_matches_frozen_reference_on_50_real_masks(self):
        import cv2
        from real_validation.perception.skeleton import extract_skeleton_2d

        files = sorted(REAL_MASKS.glob("*.png"))
        self.assertGreater(len(files), 50)
        for index in np.linspace(0, len(files) - 1, 50).astype(int):
            path = files[index]
            mask = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            expected = _legacy_extract_skeleton_2d(mask, 15, tip_fix=True)
            actual = extract_skeleton_2d(mask, 15, tip_fix=True)
            self.assertTrue(np.array_equal(actual, expected), path.name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.perception.skeleton'`

- [ ] **Step 4: 建 `real_validation/perception/skeleton.py`**

把 `src/utils/skeleton_2d.py:13-120` 的三个函数(`_perpendicular_tip_fix`、`extract_skeleton_2d`、`batch_extract_skeleton_2d`)**逐字**搬过来,含全部 docstring,一个字符都不改。文件头换成:

```python
"""2D 骨架提取（唯一实现）。

逐行质心 → 弧长均匀重采样 → 可选 tip_fix。只依赖 numpy，供在线部署与离线数据
准备共用；src/utils/skeleton_2d.py 是本模块的薄壳。

节点顺序：node0 = tip（图像底部、运动末端），node N-1 = base（图像顶部、固定基座）。
"""

import numpy as np
```

- [ ] **Step 5: 把 `src/utils/skeleton_2d.py` 改薄壳**

文件前半替换为下面内容,后半(`project_3d_to_2d` 与 `compute_2d_skeleton_loss`,原 `:123-192`)**一字不改地保留**:

```python
"""skeleton_2d.py — 2D 骨架提取与 3D→2D 投影工具。

⚠️ 骨架提取的**唯一实现**已移至 real_validation/perception/skeleton.py
（部署产物持有实现；本文件是薄壳，签名与行为完全不变）。
本文件仍保留需要 torch 的投影/loss 工具，因为 perception 包不依赖 torch。

提供:
  - 从二值图像提取 2D 中心线/骨架（re-export）
  - 将 3D 骨架点投影到相机像素坐标
  - 计算 2D 投影骨架 loss（Phase 1 监督信号）
"""

import numpy as np
import torch

from real_validation.perception.skeleton import (  # noqa: F401  薄壳 re-export
    _perpendicular_tip_fix,
    batch_extract_skeleton_2d,
    extract_skeleton_2d,
)

__all__ = [
    "_perpendicular_tip_fix",
    "batch_extract_skeleton_2d",
    "extract_skeleton_2d",
    "project_3d_to_2d",
    "compute_2d_skeleton_loss",
]
```

- [ ] **Step 6: 运行 parity 测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity -v`
Expected: PASS(3 个合成测试;本机有 mask 目录时 4 个)

- [ ] **Step 7: 确认既有调用点未被破坏**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
grep -rn "skeleton_2d" --include=*.py scripts/ src/ sam2/ tests/ | grep -v "real_validation/perception"
python -c "from src.utils.skeleton_2d import batch_extract_skeleton_2d, extract_skeleton_2d, project_3d_to_2d, compute_2d_skeleton_loss; print('shim ok')"
python -c "import numpy as np; from src.utils.skeleton_2d import extract_skeleton_2d as f; m=np.zeros((40,20),np.uint8); m[10:35,8:12]=1; print(f(m,15,tip_fix=True).shape)"
```
Expected: grep 列出的每个调用点导入的名字都在 `__all__` 里;两条 python 命令成功。

- [ ] **Step 8: 运行现有测试确认无回归**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core -v`
Expected: 22 tests OK

- [ ] **Step 9: 询问用户后提交**

先问"Task 1 完成,是否提交?",获准后:
```bash
git add real_validation/perception/__init__.py real_validation/perception/skeleton.py \
        src/utils/skeleton_2d.py tests/test_perception_parity.py
git commit -m "refactor(perception): 骨架提取迁为唯一实现 + src 薄壳 + 冻结参考 parity 测试"
```

---

### Task 2: tip_fix 可观测化(修 B13)

现状:`_perpendicular_tip_fix` 的门控是**静默跳过**(`src/utils/skeleton_2d.py:28-45` 的 4 处 `return skeleton`),调用方无从得知末端 node0 可能落在 cap 角落。在线质量门控需要这个信号。

**Files:**
- Modify: `real_validation/perception/skeleton.py`
- Modify: `tests/test_perception_parity.py`(追加测试类)

**Interfaces:**
- Consumes: Task 1 的 `extract_skeleton_2d`
- Produces:
  - 常量 `TIP_FIX_APPLIED="applied"`、`TIP_FIX_NOT_REQUESTED="not_requested"`、`TIP_FIX_SKIP_FEW_POINTS="n_points_lt_5"`、`TIP_FIX_SKIP_ZERO_SKELETON="zero_skeleton"`、`TIP_FIX_SKIP_FEW_FOREGROUND="foreground_lt_10"`、`TIP_FIX_SKIP_DEGENERATE_AXIS="local_axis_degenerate"`、`TIP_FIX_SKIP_THIN_SLAB="tip_slab_lt_3"`
  - `_perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points)` → `(np.ndarray, str)`
  - `_perpendicular_tip_fix(...)` → `np.ndarray`(行为不变)
  - `extract_skeleton_2d(binary_img, n_points=31, tip_fix=False, return_info=False)`;`return_info=True` → `(skeleton, info)`,`info` 键为 `tip_fix_requested: bool`、`tip_fix_applied: bool`、`tip_fix_reason: str`、`n_foreground_px: int`、`n_valid_rows: int`
  - `batch_extract_skeleton_2d` 签名不变

- [ ] **Step 1: 写失败测试**

在 `tests/test_perception_parity.py` 的 `if __name__` 之前追加:

```python
class TipFixObservabilityTest(unittest.TestCase):
    def _info(self, mask, n_points=15, tip_fix=True):
        from real_validation.perception.skeleton import extract_skeleton_2d
        return extract_skeleton_2d(mask, n_points, tip_fix=tip_fix, return_info=True)

    def test_applied_on_tilted_cap(self):
        _, info = self._info(dict(synthetic_masks())["tilted_cap"])
        self.assertTrue(info["tip_fix_requested"])
        self.assertTrue(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "applied")

    def test_skip_reason_too_few_points(self):
        _, info = self._info(dict(synthetic_masks())["bent_tube"], n_points=4)
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "n_points_lt_5")

    def test_skip_reason_too_few_foreground(self):
        mask = np.zeros((40, 20), np.uint8)
        mask[10:13, 8:11] = 1
        self.assertLess(int(mask.sum()), 10)
        _, info = self._info(mask)
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "foreground_lt_10")

    def test_not_requested_reports_reason(self):
        _, info = self._info(dict(synthetic_masks())["bent_tube"], tip_fix=False)
        self.assertFalse(info["tip_fix_requested"])
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "not_requested")

    def test_empty_mask_reports_zero_skeleton(self):
        _, info = self._info(dict(synthetic_masks())["empty"])
        self.assertEqual(info["tip_fix_reason"], "zero_skeleton")
        self.assertEqual(info["n_valid_rows"], 0)

    def test_return_info_false_keeps_legacy_return_type(self):
        from real_validation.perception.skeleton import extract_skeleton_2d
        result = extract_skeleton_2d(dict(synthetic_masks())["bent_tube"], 15, tip_fix=True)
        self.assertIsInstance(result, np.ndarray)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity.TipFixObservabilityTest -v`
Expected: FAIL — `TypeError: extract_skeleton_2d() got an unexpected keyword argument 'return_info'`

- [ ] **Step 3: 加常量**

在 `real_validation/perception/skeleton.py` 的 `import numpy as np` 之后插入:

```python
TIP_FIX_APPLIED = "applied"
TIP_FIX_NOT_REQUESTED = "not_requested"
TIP_FIX_SKIP_FEW_POINTS = "n_points_lt_5"
TIP_FIX_SKIP_ZERO_SKELETON = "zero_skeleton"
TIP_FIX_SKIP_FEW_FOREGROUND = "foreground_lt_10"
TIP_FIX_SKIP_DEGENERATE_AXIS = "local_axis_degenerate"
TIP_FIX_SKIP_THIN_SLAB = "tip_slab_lt_3"
```

- [ ] **Step 4: 拆出带原因的 tip_fix**

新增 `_perpendicular_tip_fix_with_reason`(把原函数体逐句搬进来,每个提前 return 带上原因常量;注意原来 `n_points < 5 or max==0` 是同一个 if,现在必须拆成两个以区分原因):

```python
def _perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points):
    """与 _perpendicular_tip_fix 相同的计算，同时返回是否生效/跳过原因。"""
    sk = skeleton.astype(np.float64)
    if n_points < 5:
        return skeleton, TIP_FIX_SKIP_FEW_POINTS
    if np.abs(sk).max() == 0:
        return skeleton, TIP_FIX_SKIP_ZERO_SKELETON
    ys, xs = np.where(binary_img > 0.5)
    if len(xs) < 10:
        return skeleton, TIP_FIX_SKIP_FEW_FOREGROUND
    pts = np.column_stack([xs.astype(float), ys.astype(float)])  # (col, row)
    far = sk[min(max(2, int(0.25 * n_points)), n_points - 1)]
    near = sk[min(max(1, int(0.10 * n_points)), n_points - 1)]
    seg = near - far
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return skeleton, TIP_FIX_SKIP_DEGENERATE_AXIS
    d = seg / L
    proj = (pts - far) @ d
    w = float(binary_img.sum(1).max())
    slab = proj >= proj.max() - 0.4 * w
    if int(slab.sum()) < 3:
        return skeleton, TIP_FIX_SKIP_THIN_SLAB
    node0 = pts[slab].mean(0)
    sk[0] = node0
    a = sk[min(3, n_points - 1)]
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32), TIP_FIX_APPLIED
```

把原 `_perpendicular_tip_fix` 的函数体换成一行委托,**docstring 全文保留**并在末尾补一句:

```python
def _perpendicular_tip_fix(skeleton, binary_img, n_points):
    """<原有 docstring 全文保留>

    行为与迁移前完全一致；需要"是否生效"信号时改用
    _perpendicular_tip_fix_with_reason。
    """
    return _perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points)[0]
```

- [ ] **Step 5: 给 `extract_skeleton_2d` 加 `return_info`**

替换 `extract_skeleton_2d` 的函数体(docstring 保留并补 `return_info` 说明):

```python
def extract_skeleton_2d(binary_img, n_points=31, tip_fix=False, return_info=False):
    """<原有 docstring 保留>

        return_info: True 时返回 (skeleton, info)；info 含 tip_fix_requested /
            tip_fix_applied / tip_fix_reason / n_foreground_px / n_valid_rows。
            默认 False，返回值与迁移前完全一致。
    """
    H, W = binary_img.shape
    n_foreground = int((binary_img > 0.5).sum())
    coords = []

    for row in range(H - 1, -1, -1):
        white_cols = np.where(binary_img[row] > 0.5)[0]
        if len(white_cols) > 0:
            center_col = white_cols.mean()
            coords.append([center_col, float(row)])

    def _wrap(skeleton, reason, n_valid_rows):
        if not return_info:
            return skeleton
        return skeleton, {
            "tip_fix_requested": bool(tip_fix),
            "tip_fix_applied": reason == TIP_FIX_APPLIED,
            "tip_fix_reason": reason,
            "n_foreground_px": n_foreground,
            "n_valid_rows": int(n_valid_rows),
        }

    if len(coords) < 2:
        return _wrap(np.zeros((n_points, 2), dtype=np.float32),
                     TIP_FIX_SKIP_ZERO_SKELETON, len(coords))

    coords = np.array(coords, dtype=np.float32)

    # 沿弧长均匀重采样
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return _wrap(np.zeros((n_points, 2), dtype=np.float32),
                     TIP_FIX_SKIP_ZERO_SKELETON, len(coords))

    target_lens = np.linspace(0, total_len, n_points)

    resampled = np.zeros((n_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lens, cum_len, coords[:, 0])
    resampled[:, 1] = np.interp(target_lens, cum_len, coords[:, 1])

    if not tip_fix:
        return _wrap(resampled, TIP_FIX_NOT_REQUESTED, len(coords))
    fixed, reason = _perpendicular_tip_fix_with_reason(resampled, binary_img, n_points)
    return _wrap(fixed, reason, len(coords))
```

- [ ] **Step 6: 运行全部感知测试**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity -v`
Expected: 全部 PASS —— parity 测试必须仍通过(`return_info=False` 路径逐位不变)

- [ ] **Step 7: 询问用户后提交**

```bash
git add real_validation/perception/skeleton.py tests/test_perception_parity.py
git commit -m "feat(perception): tip_fix 生效与跳过原因可观测(修静默跳过)"
```

---

### Task 3: 分割实现迁移 + src 薄壳

**Files:**
- Create: `real_validation/perception/segmentation.py`
- Modify: `src/data/real/segmentation.py:1-171`(全文改薄壳)
- Modify: `tests/test_perception_parity.py`(追加分割测试)
- Modify: `real_validation/requirements.txt`

**Interfaces:**
- Consumes: Task 1 的 `real_validation.perception.skeleton.batch_extract_skeleton_2d`
- Produces(`real_validation.perception.segmentation` 与 `src.data.real.segmentation` 均导出,签名与迁移前完全一致):
  `_clean(mask)`、`segment_backlight(gray, thresh=60)`、`segment_bg_subtract(gray, bg_gray, thresh=25)`、`segment_color(bgr, lower_hsv, upper_hsv)`、`build_median_background(cam_dir, n_bg=500)` → `(bg_gray, frame_paths)`、`segment_white_on_blue(bgr, bg_gray, sat=100, val=120, diff=25, dil=35, open_k=5, close_k=15, min_area_frac=0.003, min_h_frac=0.15)` → `(H,W) uint8 {0,1}`、`segment_views(...)`、`masks_to_skeletons_2d(masks, n_points=31, tip_fix=True)`

- [ ] **Step 1: 写失败测试**

在 `tests/test_perception_parity.py` 的 `TipFixObservabilityTest` 之后追加:

```python
def _legacy_clean(mask):
    import cv2
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (lbl == keep).astype(np.uint8)


def _legacy_segment_white_on_blue(bgr, bg_gray, sat=100, val=120, diff=25, dil=35,
                                  open_k=5, close_k=15,
                                  min_area_frac=0.003, min_h_frac=0.15):
    import cv2
    from scipy.ndimage import binary_fill_holes
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < sat) & (hsv[:, :, 2] > val)).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    moved = (cv2.absdiff(gray, bg_gray) > diff).astype(np.uint8)
    moved = cv2.dilate(moved, np.ones((dil, dil), np.uint8)) if dil > 1 else moved
    m = (white & moved).astype(np.uint8)
    if open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    m = binary_fill_holes(m > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros((H, W), np.uint8)
    if n > 1:
        cands = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * H * W
                 and stats[i, cv2.CC_STAT_HEIGHT] >= min_h_frac * H]
        if cands:
            cands.sort(reverse=True)
            out[lbl == cands[0][1]] = 1
    return out


def synthetic_bgr_scene():
    """合成 (bgr, bg_gray)：蓝墙背景 + 白半透明弯臂 + 一条白细管干扰。"""
    import cv2
    rng = np.random.default_rng(20260728)
    H, W = 160, 120
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:, :, 0] = 180
    bg[:, :, 1] = 70
    bg[:, :, 2] = 40
    bg = np.clip(bg.astype(np.int16) + rng.integers(-4, 5, bg.shape), 0, 255).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    frame = bg.copy()
    for row in range(24, 150):
        left = 52 + int(round(0.05 * (row - 24) ** 1.3 / 10.0))
        frame[row, left:left + 11] = (235, 235, 238)
    frame[10:150, 100:103] = (240, 240, 240)   # 白细管干扰
    return frame, bg_gray


class SegmentationParityTest(unittest.TestCase):
    def test_white_on_blue_matches_frozen_reference(self):
        from real_validation.perception.segmentation import segment_white_on_blue
        bgr, bg_gray = synthetic_bgr_scene()
        for val in (100, 120):
            expected = _legacy_segment_white_on_blue(bgr, bg_gray, val=val)
            actual = segment_white_on_blue(bgr, bg_gray, val=val)
            self.assertTrue(np.array_equal(actual, expected), f"val={val}")
            self.assertEqual(actual.dtype, np.uint8)

    def test_clean_matches_frozen_reference(self):
        from real_validation.perception.segmentation import _clean
        rng = np.random.default_rng(7)
        mask = (rng.random((80, 60)) > 0.75).astype(np.uint8)
        mask[20:60, 25:35] = 1
        self.assertTrue(np.array_equal(_clean(mask.copy()), _legacy_clean(mask.copy())))

    def test_masks_to_skeletons_uses_canonical_skeleton(self):
        from real_validation.perception.segmentation import masks_to_skeletons_2d
        from real_validation.perception.skeleton import batch_extract_skeleton_2d
        bent = dict(synthetic_masks())["bent_tube"]
        masks = bent[None, None, :, :]
        out = masks_to_skeletons_2d(masks, n_points=15, tip_fix=True)
        self.assertEqual(out.shape, (1, 1, 15, 2))
        self.assertTrue(np.array_equal(
            out[0], batch_extract_skeleton_2d(masks[0], 15, tip_fix=True)))

    def test_shim_reexports_same_objects(self):
        import src.data.real.segmentation as shim
        from real_validation.perception import segmentation as canonical
        for name in ("_clean", "segment_backlight", "segment_bg_subtract",
                     "segment_color", "build_median_background",
                     "segment_white_on_blue", "segment_views",
                     "masks_to_skeletons_2d"):
            self.assertIs(getattr(shim, name), getattr(canonical, name), name)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity.SegmentationParityTest -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.perception.segmentation'`

- [ ] **Step 3: 建 `real_validation/perception/segmentation.py`**

把 `src/data/real/segmentation.py:13-170` 的全部内容(`import glob/os/numpy`、cv2 的 try-import、8 个函数)**逐字**搬过来,只改两处:

(a) 文件头 docstring:

```python
"""彩色图 → 二值剪影（唯一实现）。

实物硅胶半透明、易高光，分割是最易出错的环节。按可用条件选方法：
  - 'backlight'      背光剪影，臂成暗块 → 亮度反相阈值（最稳，推荐）
  - 'bg_subtract'    减去参考背景 → 阈值（需先拍无臂背景）
  - 'color'          HSV 颜色阈值（臂为特定色，如染色/涂层）
  - 'white_on_blue'  白半透明硅胶臂 + 蓝静态墙背景 + 白气管场景（专用，diag 校准）

统一输出 (H,W) 二值（1=前景=臂），形态学清理 + 取最大连通区，可直接喂
real_validation.perception.skeleton.extract_skeleton_2d。
src/data/real/segmentation.py 是本模块的薄壳。

⚠️ 在线部署必须使用与训练一致的参数。真实参数在
   real_capture/data/derived/<seq>/segment_meta.json（实测 val=100，非默认 120）。
"""
```

(b) `masks_to_skeletons_2d` 内部的 import 改成相对导入:

```python
    from .skeleton import batch_extract_skeleton_2d
```

- [ ] **Step 4: 把 `src/data/real/segmentation.py` 改薄壳**

整个文件替换为:

```python
"""segmentation.py — 薄壳：实现已移至 real_validation/perception/segmentation.py。

部署产物持有实现（工作台需要在没有仓库 src/ 的 PC 上运行同一份分割代码），
本文件保持原有公开签名不变，供离线数据准备脚本继续使用。
"""

from real_validation.perception.segmentation import (  # noqa: F401
    _clean,
    build_median_background,
    masks_to_skeletons_2d,
    segment_backlight,
    segment_bg_subtract,
    segment_color,
    segment_views,
    segment_white_on_blue,
)

__all__ = [
    "_clean",
    "build_median_background",
    "masks_to_skeletons_2d",
    "segment_backlight",
    "segment_bg_subtract",
    "segment_color",
    "segment_views",
    "segment_white_on_blue",
]
```

- [ ] **Step 5: 补依赖**

在 `real_validation/requirements.txt` 末尾追加:
```
opencv-python>=4.5
scipy>=1.7
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_parity -v`
Expected: 全部 PASS

- [ ] **Step 7: 确认既有调用点未被破坏**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
grep -rn "data\.real\.segmentation\|data\.real import segmentation" --include=*.py scripts/ src/ sam2/ | sort
python -c "from src.data.real.segmentation import build_median_background, segment_white_on_blue, segment_views, masks_to_skeletons_2d; print('shim ok')"
python scripts/real/segment_batch.py --help > /dev/null && echo "segment_batch --help ok"
```
Expected: grep 命中的名字都在 `__all__` 内;两条命令均成功。

- [ ] **Step 8: 运行全部测试**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core tests.test_perception_parity -v`
Expected: 全部 OK

- [ ] **Step 9: 询问用户后提交**

```bash
git add real_validation/perception/segmentation.py src/data/real/segmentation.py \
        real_validation/requirements.txt tests/test_perception_parity.py
git commit -m "refactor(perception): 分割迁为唯一实现 + src 薄壳 + parity 测试"
```

---

### Task 4: import 卫生锁(T9)—— 用机制保护偶然的纯净性

反向 import 之所以可行,全靠 `real_validation/__init__.py` 的传递闭包恰好是 stdlib-only,**这份纯净性是偶然的、无任何机制保护**:任何人往 `__init__.py` 加一行 `from .model_runtime import ModelRuntime`,离线数据准备脚本就会开始依赖 torch;任何人往 `perception/__init__.py` 加 `from .segmentation import ...`,只要骨架的脚本就会被迫装 cv2+scipy(反面教材:`src/data/real/__init__.py:10` 就是 eager 绝对 import)。

**Files:**
- Create: `tests/test_import_hygiene.py`
- Modify: `real_validation/__init__.py`(docstring 写死约束)
- Modify: `real_validation/requirements.txt` + Create: `real_validation/requirements-perception.txt`
- Modify: `real_validation/README.md:20-28, 43-47`
- Modify: `requirements.txt`(根,补 scipy)

**Interfaces:**
- Consumes: Task 1/3 的 perception 模块
- Produces: 无新 API;产出一条**永久架构约束的可执行断言**

- [ ] **Step 1: 先实测当前依赖可用性(inconclusive 项,必须实跑)**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -c "import cv2; print(cv2.__version__, hasattr(cv2,'ORB_create'), hasattr(cv2,'BFMatcher'), hasattr(cv2,'findHomography'), hasattr(cv2,'estimateAffinePartial2D'))"
python -c "import scipy; print('scipy', scipy.__version__)"
grep -n "opencv\|scipy" requirements.txt
```
Expected: cv2 各 `hasattr` 全 True(**若 `ORB_create` 为 False 说明装的是 headless 精简包,需先 `pip install opencv-python`**);记录 scipy 版本;`requirements.txt` 有 opencv-python 无 scipy。

- [ ] **Step 2: 写失败测试**

创建 `tests/test_import_hygiene.py`:

```python
"""import 卫生:锁死 real_validation 反向被 src/ 依赖时的最小依赖面。

必须用子进程 —— 同进程里其它测试早已 import torch，会把本测试变成假阴性。
"""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FORBIDDEN = ("torch", "PyQt5", "pyqtgraph", "cv2", "scipy", "matplotlib")
# numpy 不在禁列 —— perception.skeleton 本来就需要 numpy。


def _leaked(statement: str) -> list[str]:
    script = textwrap.dedent(f"""
        import sys
        {statement}
        forbidden = {FORBIDDEN!r}
        leaked = sorted({{name.split('.')[0] for name in sys.modules
                         if name.split('.')[0] in forbidden}})
        print(",".join(leaked))
    """)
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                               capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise AssertionError(f"子进程失败:\n{completed.stderr}")
    payload = completed.stdout.strip()
    return payload.split(",") if payload else []


class ImportHygieneTest(unittest.TestCase):
    def test_package_root_is_stdlib_only(self):
        self.assertEqual(_leaked("import real_validation"), [])

    def test_perception_package_has_no_side_effects(self):
        self.assertEqual(_leaked("import real_validation.perception"), [])

    def test_skeleton_module_needs_no_cv2_or_scipy(self):
        self.assertEqual(
            _leaked("from real_validation.perception.skeleton import "
                    "extract_skeleton_2d, batch_extract_skeleton_2d"), [])

    def test_segmentation_shim_import_never_raises(self):
        # src/data/real/__init__.py:10 是 eager 绝对 import，capture_to_npz.py:31 与
        # inspect_capture.py:30 都会走到；薄壳 import-time 绝不能 raise。
        completed = subprocess.run(
            [sys.executable, "-c", "import src.data.real; import src.utils.skeleton_2d; print('ok')"],
            cwd=REPO, capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_import_hygiene -v`
Expected: 4 tests PASS。**若任何一条 leak 出 `cv2`/`scipy`,说明 `perception/__init__.py` 或 `real_validation/__init__.py` 被加了 import,必须回退。**

- [ ] **Step 4: 在 `real_validation/__init__.py` docstring 写死约束**

把 docstring 换成:

```python
"""实机模型验证工作台。

核心模块不依赖 Qt；GUI、CLI 和测试共用同一套 session、preflight 与 executor。

⚠️ **本文件的 import 闭包必须保持 stdlib-only**。
   src/ 与离线数据准备脚本反向依赖本包（src/utils/skeleton_2d.py 与
   src/data/real/segmentation.py 是 real_validation.perception 的薄壳），
   一旦这里 import 了 torch / PyQt5 / cv2 / scipy，仿真训练与 npz 准备就会被迫
   拉入部署侧依赖。新增重导出前先跑 tests/test_import_hygiene.py。
"""
```

- [ ] **Step 5: 拆分依赖声明**

创建 `real_validation/requirements-perception.txt`:
```
# 在线感知（分割 / 骨架 / 配准 / 质量门控）需要的额外依赖。
# 只跑离线 Mock 工作台不需要装本文件。
opencv-python>=4.5
scipy>=1.7
```

把 Task 3 Step 5 加进 `real_validation/requirements.txt` 的两行**移除**,改在文件末尾加一行注释:
```
# 在线感知另见 requirements-perception.txt（opencv-python / scipy）
```

根 `requirements.txt`:在 `opencv-python==4.11.0.86` 附近补一行(scipy 现在是隐式依赖 —— `perception/segmentation.py` 的 `binary_fill_holes`、`scripts/real/repair_masks.py:27` 都在用):
```
scipy>=1.7
```

- [ ] **Step 6: 修 `real_validation/README.md` 的两处**

(a) 第 20-28 行的依赖段,在 `pip install -r requirements.txt` 之后补一段:

```markdown
在线感知（分割 / 骨架 / 配准）另需：

```bash
python -m pip install -r requirements-perception.txt
```
```

(b) 第 43-47 行**现在是坏指令**(它让操作员在 PC 上跑 `python -m unittest tests.test_real_validation_core`,而 PC 上只拷 `real_validation/`,没有 `tests/`)。替换为:

```markdown
## 自检

本目录的单元测试住在仓库的 `tests/`（**不随本目录拷贝到 PC**）。在 PC 上只能做
运行时自检：

```bash
python -c "import real_validation; print('contracts ok')"
python perception_probe.py --source dir --frames 3 --out /tmp/probe   # 需要一段本地帧
```

完整测试（20 个契约测试 + 感知 parity + import 卫生）在仓库根运行：

```bash
python -m unittest tests.test_real_validation_core tests.test_perception_parity tests.test_import_hygiene -v
```
```

- [ ] **Step 7: 运行全部测试**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core tests.test_perception_parity tests.test_import_hygiene -v`
Expected: 20 + parity + 4 全 OK

- [ ] **Step 8: 询问用户后提交**

```bash
git add tests/test_import_hygiene.py real_validation/__init__.py \
        real_validation/requirements.txt real_validation/requirements-perception.txt \
        real_validation/README.md requirements.txt
git commit -m "test(real_validation): import 卫生锁 + 拆分感知依赖 + 修 README 的坏自检指令"
```

---

### Task 5: 背景与位姿注册(`background.py` + `registration.py`)

**注册只做检测,不做 warp** —— 因为重采后采集位姿 == 部署位姿,`camera_pixel → model` 是**恒等映射 + 一个残差门**。`registration.py` 的职责是**证明这个恒等映射仍成立**。

关键设计:输出**两个**数字,门控用后者:
- `fit_residual_px` = 内点重投影误差中位数(**拟合质量**)
- `displacement_px` = 把 H 作用到图像四角后的**最大位移**(**位姿到底移了多远**)

**Files:**
- Create: `real_validation/perception/background.py`
- Create: `real_validation/perception/registration.py`
- Create: `tests/test_perception_registration.py`

**Interfaces:**
- Consumes: 无(独立)
- Produces:
  - `background.load_median_background(path)` → `np.ndarray (H,W) uint8`
  - `background.build_median_background_from_frames(frames, n_bg=500)` → `np.ndarray (H,W) uint8`(与 `segmentation.build_median_background` 同算法,但吃内存中的帧序列而非目录)
  - `background.background_drift(reference_gray, live_gray)` → `float`(逐像素绝对差的中位数)
  - `registration.RegistrationResult`(frozen dataclass):`homography: tuple[tuple[float,...],...]`、`fit_residual_px: float`、`displacement_px: float`、`n_inliers: int`、`n_matches: int`、`reference_sha256: str`、`ok: bool`、`reason: str`
  - `registration.estimate_registration(reference_gray, live_gray, *, reference_sha256="", max_displacement_px=2.0, min_inliers=12)` → `RegistrationResult`
  - `registration.save_registration(result, path)` / `registration.load_registration(path)`
  - 常量 `REG_OK="ok"`、`REG_TOO_FEW_FEATURES="too_few_features"`、`REG_TOO_FEW_MATCHES="too_few_matches"`、`REG_HOMOGRAPHY_FAILED="homography_failed"`、`REG_DISPLACED="displaced"`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_perception_registration.py`:

```python
"""位姿注册的离线验收。

三段测法缺一不可：
  (a) 同一帧 → 位移 ≈ 0            —— 证明不误报
  (b) 人工平移 3 px → 位移 ≈ 3      —— **唯一能证明数值有意义而不是恒返回 0 的测试**
  (c) 纯色图特征不足 → 必须显式失败 —— 否则"配准通过"变成默认值
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def textured_frame(seed: int = 11) -> np.ndarray:
    """带足量角点的合成灰度图（ORB 需要真实纹理，随机噪声不够稳）。"""
    import cv2
    rng = np.random.default_rng(seed)
    image = np.full((240, 320), 40, np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(10, 300)), int(rng.integers(10, 220))
        size = int(rng.integers(4, 14))
        shade = int(rng.integers(120, 250))
        cv2.rectangle(image, (x, y), (x + size, y + size), shade, -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


class RegistrationTest(unittest.TestCase):
    def test_identical_frame_reports_zero_displacement(self):
        from real_validation.perception.registration import estimate_registration
        frame = textured_frame()
        result = estimate_registration(frame, frame.copy())
        self.assertTrue(result.ok, result.reason)
        self.assertLess(result.displacement_px, 0.5)
        self.assertLess(result.fit_residual_px, 0.5)

    def test_known_translation_is_recovered(self):
        import cv2
        from real_validation.perception.registration import estimate_registration
        frame = textured_frame()
        shift = np.float32([[1, 0, 3], [0, 1, 0]])
        moved = cv2.warpAffine(frame, shift, (frame.shape[1], frame.shape[0]))
        result = estimate_registration(frame, moved, max_displacement_px=2.0)
        self.assertLess(abs(result.displacement_px - 3.0), 0.3,
                        f"位移={result.displacement_px}")
        self.assertFalse(result.ok)          # 3 px > 阈值 2 px → 必须阻断
        self.assertEqual(result.reason, "displaced")

    def test_featureless_frame_fails_loudly(self):
        from real_validation.perception.registration import estimate_registration
        blank = np.zeros((240, 320), np.uint8)
        result = estimate_registration(blank, blank.copy())
        self.assertFalse(result.ok)
        self.assertIn(result.reason, {"too_few_features", "too_few_matches"})
        # 关键：失败时绝不能报 0 位移，否则"配准通过"成为默认
        self.assertTrue(np.isnan(result.displacement_px))

    def test_round_trip_json(self):
        from real_validation.perception.registration import (
            estimate_registration, load_registration, save_registration)
        frame = textured_frame()
        result = estimate_registration(frame, frame.copy(), reference_sha256="deadbeef")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            save_registration(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["reference_sha256"], "deadbeef")
            restored = load_registration(path)
            self.assertEqual(restored.reference_sha256, "deadbeef")
            self.assertAlmostEqual(restored.displacement_px, result.displacement_px, places=9)


class BackgroundTest(unittest.TestCase):
    def test_median_background_ignores_moving_object(self):
        from real_validation.perception.background import build_median_background_from_frames
        base = textured_frame(seed=3)
        frames = []
        for index in range(9):
            frame = base.copy()
            frame[:, index * 30:index * 30 + 20] = 255      # 移动的亮块
            frames.append(frame)
        median = build_median_background_from_frames(np.stack(frames))
        # 每列被遮挡的时间 < 50% → 中值应回到 base
        self.assertLess(float(np.abs(median.astype(np.int16) -
                                     base.astype(np.int16)).mean()), 3.0)

    def test_drift_detects_shifted_background(self):
        import cv2
        from real_validation.perception.background import background_drift
        base = textured_frame(seed=5)
        moved = cv2.warpAffine(base, np.float32([[1, 0, 8], [0, 1, 0]]),
                               (base.shape[1], base.shape[0]))
        self.assertLess(background_drift(base, base.copy()), 1.0)
        self.assertGreater(background_drift(base, moved), 5.0)


REAL_BG = REPO / "real_capture/data/derived/seq_20260627_163921/bg_median.png"
REAL_CAM0 = REPO / "real_capture/data/raw/seq_20260627_163921/cam0"


@unittest.skipUnless(REAL_BG.is_file() and REAL_CAM0.is_dir(),
                     "真实采集数据不存在（已 gitignore；只能在服务器/本机验收）")
class RegistrationOnRealFramesTest(unittest.TestCase):
    def test_consecutive_real_frames_are_registered(self):
        import cv2
        from real_validation.perception.registration import estimate_registration
        frames = sorted(REAL_CAM0.glob("*.png"))[:2]
        first = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(frames[1]), cv2.IMREAD_GRAYSCALE)
        result = estimate_registration(first, second)
        self.assertTrue(result.ok, f"{result.reason} disp={result.displacement_px}")
        self.assertLess(result.displacement_px, 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_registration -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.perception.registration'`

- [ ] **Step 3: 实现 `real_validation/perception/background.py`**

```python
"""静态背景的加载 / 重建 / 漂移检测。

中值背景是 white_on_blue 分割的第 2 步（背景差）所依赖的量，且它逐像素绑定相机
位姿 —— 相机一动，absdiff 全图激活、分割崩溃。因此这里同时提供漂移检测。
"""

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc


def load_median_background(path):
    """读取 bg_median.png → (H,W) uint8 灰度。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取背景图：{path}")
    return image


def build_median_background_from_frames(frames, n_bg: int = 500):
    """(T,H,W) 灰度或 (T,H,W,3) BGR 帧序列 → per-pixel 中值背景 (H,W) uint8。

    与 segmentation.build_median_background 同算法，但吃内存中的序列而非目录，
    供在线"开机重建背景"使用。机器人移动占每像素 <50% 时间 → 中值 ≈ 静态背景。
    """
    array = np.asarray(frames)
    if array.ndim == 4:
        if cv2 is None:
            raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
        array = np.stack([cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in array])
    if array.ndim != 3 or len(array) == 0:
        raise ValueError("frames 必须是 (T,H,W) 灰度或 (T,H,W,3) BGR，且 T>0")
    index = np.linspace(0, len(array) - 1, min(n_bg, len(array))).astype(int)
    return np.median(array[index], axis=0).astype(np.uint8)


def background_drift(reference_gray, live_gray) -> float:
    """两张背景灰度图的逐像素绝对差中位数（灰阶）。

    用中位数而非均值：对局部遮挡（手、异物）稳健，对全局位移敏感。
    """
    reference = np.asarray(reference_gray, dtype=np.int16)
    live = np.asarray(live_gray, dtype=np.int16)
    if reference.shape != live.shape:
        raise ValueError(f"背景尺寸不同：{reference.shape} != {live.shape}")
    return float(np.median(np.abs(reference - live)))
```

- [ ] **Step 4: 实现 `real_validation/perception/registration.py`**

```python
"""相机位姿注册：证明"live 像素 == 训练期像素"这个恒等映射仍成立。

免标定路线把 state 定义成绝对图像像素，于是 pc_center/pc_scale、背景图、关节锚点、
NDI 仿射全部绑死在采集时那个相机位姿上。相机一动，失效方式是**静默的**：分割照样
出 mask、骨架照样出 15 点，数值全错。

本模块只做**检测**，不做 warp：重采后采集位姿 == 部署位姿，camera_pixel → model 是
恒等映射 + 一个残差门。输出两个数字，门控用 displacement_px：
  fit_residual_px  内点重投影误差中位数 —— 拟合质量
  displacement_px  H 作用到图像四角的最大位移 —— 位姿到底移了多远

失败时 displacement_px 是 NaN，绝不是 0 —— 否则"配准通过"会成为默认值。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc

REG_OK = "ok"
REG_TOO_FEW_FEATURES = "too_few_features"
REG_TOO_FEW_MATCHES = "too_few_matches"
REG_HOMOGRAPHY_FAILED = "homography_failed"
REG_DISPLACED = "displaced"

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class RegistrationResult:
    homography: tuple[tuple[float, ...], ...]
    fit_residual_px: float
    displacement_px: float
    n_inliers: int
    n_matches: int
    reference_sha256: str
    ok: bool
    reason: str

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> "RegistrationResult":
        data = dict(value)
        data.pop("schema_version", None)
        data["homography"] = tuple(tuple(float(v) for v in row)
                                   for row in data["homography"])
        return cls(**data)


def _failure(reason: str, n_matches: int = 0, n_inliers: int = 0,
             reference_sha256: str = "") -> RegistrationResult:
    return RegistrationResult(
        homography=_IDENTITY, fit_residual_px=float("nan"),
        displacement_px=float("nan"), n_inliers=n_inliers, n_matches=n_matches,
        reference_sha256=reference_sha256, ok=False, reason=reason)


def _corner_displacement(homography, width: int, height: int) -> float:
    corners = np.float32([[0, 0], [width - 1, 0],
                          [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return float(np.linalg.norm(mapped - corners.reshape(-1, 2), axis=1).max())


def estimate_registration(reference_gray, live_gray, *, reference_sha256: str = "",
                          max_displacement_px: float = 2.0,
                          min_inliers: int = 12) -> RegistrationResult:
    """ORB + BFMatcher(Hamming, crossCheck) + RANSAC homography。

    只用 opencv-python 主包（ORB 不在 contrib）。
    """
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    reference = np.asarray(reference_gray)
    live = np.asarray(live_gray)
    if reference.shape != live.shape:
        raise ValueError(f"参考帧与 live 帧尺寸不同：{reference.shape} != {live.shape}")
    height, width = reference.shape[:2]

    orb = cv2.ORB_create(nfeatures=2000)
    key_ref, desc_ref = orb.detectAndCompute(reference, None)
    key_live, desc_live = orb.detectAndCompute(live, None)
    if desc_ref is None or desc_live is None or len(key_ref) < min_inliers \
            or len(key_live) < min_inliers:
        return _failure(REG_TOO_FEW_FEATURES, reference_sha256=reference_sha256)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_ref, desc_live)
    if len(matches) < min_inliers:
        return _failure(REG_TOO_FEW_MATCHES, n_matches=len(matches),
                        reference_sha256=reference_sha256)

    source = np.float32([key_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([key_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        return _failure(REG_HOMOGRAPHY_FAILED, n_matches=len(matches),
                        reference_sha256=reference_sha256)
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return _failure(REG_TOO_FEW_MATCHES, n_matches=len(matches),
                        n_inliers=inliers, reference_sha256=reference_sha256)

    projected = cv2.perspectiveTransform(source, homography)
    errors = np.linalg.norm(projected.reshape(-1, 2) - target.reshape(-1, 2), axis=1)
    fit_residual = float(np.median(errors[mask.ravel().astype(bool)]))
    displacement = _corner_displacement(homography, width, height)
    ok = math.isfinite(displacement) and displacement <= max_displacement_px
    return RegistrationResult(
        homography=tuple(tuple(float(v) for v in row) for row in homography),
        fit_residual_px=fit_residual, displacement_px=displacement,
        n_inliers=inliers, n_matches=len(matches),
        reference_sha256=reference_sha256,
        ok=bool(ok), reason=REG_OK if ok else REG_DISPLACED)


def save_registration(result: RegistrationResult, path) -> None:
    """写 registration.json（NaN 会被 json 拒绝，故显式转 None）。"""
    payload = result.to_dict()
    for key in ("fit_residual_px", "displacement_px"):
        if not math.isfinite(payload[key]):
            payload[key] = None
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_registration(path) -> RegistrationResult:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    for key in ("fit_residual_px", "displacement_px"):
        if payload.get(key) is None:
            payload[key] = float("nan")
    return RegistrationResult.from_dict(payload)
```

> ⚠️ `save_registration` **必须**把 NaN 转 None:`real_validation/io.py:24-27` 与 `:45-46` 都是 `allow_nan=False`,而 registration 结果会经 `session.save_snapshot()` 落进 `experiment.json` —— 直接塞 NaN 会让 `test_session_arm_and_plan_invalidation` 等 3 个测试在写快照时抛 ValueError。

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_registration -v`
Expected: 6 tests PASS(本机有真实数据时 7 个)

- [ ] **Step 6: 重跑 import 卫生(registration 引入了 cv2,必须确认没污染包根)**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_import_hygiene -v`
Expected: 4 tests PASS

- [ ] **Step 7: 询问用户后提交**

```bash
git add real_validation/perception/background.py real_validation/perception/registration.py \
        tests/test_perception_registration.py
git commit -m "feat(perception): 背景漂移检测 + ORB/RANSAC 相机位姿注册(只检测不 warp)"
```

---

### Task 6: 在线质量门控(`quality.py`)

离线那套"坏帧时间插值修复"在线**全部不可用**(需要未来帧)。在线只能**拒帧**。

**关键设计决定:数据相关阈值不给默认值。** 仓库里"mask 面积中位数"有 **4 个互相矛盾的值**(`segment_meta.json` 的 8562 / `outliers.txt` 的 6718 / `qc/` 的 6323–7099 / 运行时重算),而部署 checkpoint 用的 **SAM2 mask 一个统计都没有**;阈值倍数也有 3 套(0.3×/3.0× 的 `segment_batch.py:137-144`、1.6× 的 `compare_skeleton_methods.py:401`、`min_area_frac=0.003·H·W`)。P1a 随手挑一个会变成**第 5 套约定**。因此:`area_median_px` **无默认值,必须由调用方显式提供**;策略常量(比例、行阈值)才有默认。P2 按 §8 协议重采后再供给真实数字。

**Files:**
- Create: `real_validation/perception/quality.py`
- Create: `tests/test_perception_quality.py`

**Interfaces:**
- Consumes: Task 2 的 `extract_skeleton_2d(..., return_info=True)` 的 `info` dict
- Produces:
  - 判决常量 `Q_OK="ok"`、`Q_DEGRADED="degraded"`、`Q_REJECT="reject"`
  - 原因常量 `R_EMPTY_MASK="empty_mask"`、`R_AREA_LOW="area_ratio_low"`、`R_AREA_HIGH="area_ratio_high"`、`R_HEIGHT_LOW="height_frac_low"`、`R_TOP_ROW_HIGH="top_row_high"`、`R_SECOND_BLOB="second_blob_present"`、`R_TIP_FIX_SKIPPED="tip_fix_skipped"`、`R_NODE_STEP_HIGH="node_step_high"`、`R_FRAME_STALE="frame_stale"`、`R_REGISTRATION_DISPLACED="registration_displaced"`
  - `QualityThresholds(area_median_px, *, area_ratio_min=0.7, area_ratio_max=1.3, min_height_frac=0.15, max_top_row=20, max_second_blob_ratio=0.15, max_node_step_px=4.0, max_frame_age_s=0.5, max_registration_displacement_px=2.0)` —— **`area_median_px` 是唯一位置参数且无默认**
  - `FrameQuality`(frozen):`verdict: str`、`reasons: tuple[str,...]`、`flags: dict[str, Any]`
  - `assess_frame(mask, skeleton, skeleton_info, thresholds, *, prev_skeleton=None, frame_age_s=None, registration_displacement_px=None)` → `FrameQuality`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_perception_quality.py`:

```python
"""在线质量门控:每条判据一个测试。

在线没有未来帧 → 坏帧只能拒,不能像离线那样时间插值修复。
"""

import unittest

import numpy as np

from tests.test_perception_parity import synthetic_masks


def _skeleton(mask, n_points=15):
    from real_validation.perception.skeleton import extract_skeleton_2d
    return extract_skeleton_2d(mask, n_points, tip_fix=True, return_info=True)


def _thresholds(area_median_px=680.0, **overrides):
    from real_validation.perception.quality import QualityThresholds
    return QualityThresholds(area_median_px, **overrides)


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.bent = dict(synthetic_masks())["bent_tube"]
        self.area = float(self.bent.sum())

    def _assess(self, mask, **kwargs):
        from real_validation.perception.quality import assess_frame
        skeleton, info = _skeleton(mask)
        thresholds = kwargs.pop("thresholds", _thresholds(self.area))
        return assess_frame(mask, skeleton, info, thresholds, **kwargs)

    def test_healthy_frame_is_ok(self):
        quality = self._assess(self.bent)
        self.assertEqual(quality.verdict, "ok", quality.reasons)
        self.assertEqual(quality.reasons, ())
        self.assertAlmostEqual(quality.flags["mask_area_ratio"], 1.0, places=6)

    def test_empty_mask_is_rejected(self):
        quality = self._assess(dict(synthetic_masks())["empty"])
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("empty_mask", quality.reasons)

    def test_area_too_small_is_rejected(self):
        quality = self._assess(self.bent, thresholds=_thresholds(self.area * 2.0))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("area_ratio_low", quality.reasons)

    def test_area_too_large_is_rejected(self):
        quality = self._assess(self.bent, thresholds=_thresholds(self.area * 0.5))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("area_ratio_high", quality.reasons)

    def test_arm_not_reaching_base_is_rejected(self):
        truncated = self.bent.copy()
        truncated[:40, :] = 0                       # 顶部 40 行清空 → top_row 变大
        quality = self._assess(truncated, thresholds=_thresholds(float(truncated.sum())))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("top_row_high", quality.reasons)

    def test_second_blob_is_degraded_not_rejected(self):
        with_blob = self.bent.copy()
        blob_area = int(0.25 * self.bent.sum())
        side = max(2, int(np.sqrt(blob_area)))
        with_blob[5:5 + side, 65:65 + side] = 1     # 手/异物
        quality = self._assess(with_blob, thresholds=_thresholds(float(with_blob.sum())))
        self.assertEqual(quality.verdict, "degraded")
        self.assertIn("second_blob_present", quality.reasons)

    def test_silent_tip_fix_skip_is_degraded(self):
        thin = np.zeros((40, 20), np.uint8)
        thin[10:13, 8:11] = 1                       # 前景 < 10px → tip_fix 静默跳过
        quality = self._assess(thin, thresholds=_thresholds(float(thin.sum()),
                                                           min_height_frac=0.0,
                                                           max_top_row=40))
        self.assertIn("tip_fix_skipped", quality.reasons)
        self.assertEqual(quality.flags["tip_fix_reason"], "foreground_lt_10")

    def test_node_jump_is_rejected(self):
        skeleton, _ = _skeleton(self.bent)
        moved = skeleton + 50.0
        quality = self._assess(self.bent, prev_skeleton=moved)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("node_step_high", quality.reasons)
        self.assertGreater(quality.flags["max_node_step_px"], 4.0)

    def test_stale_frame_is_rejected(self):
        quality = self._assess(self.bent, frame_age_s=1.2)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("frame_stale", quality.reasons)

    def test_registration_displacement_is_rejected(self):
        quality = self._assess(self.bent, registration_displacement_px=7.5)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("registration_displaced", quality.reasons)

    def test_flags_are_json_safe(self):
        import json
        quality = self._assess(self.bent, frame_age_s=0.03,
                               registration_displacement_px=0.4)
        # io.py 的 atomic_write_json 是 allow_nan=False，且 json 不认 numpy 标量
        payload = json.dumps(quality.flags, allow_nan=False)
        self.assertIn("mask_area_ratio", payload)
        for value in quality.flags.values():
            self.assertIsInstance(value, (bool, int, float, str, type(None)))
            self.assertNotIsInstance(value, np.generic)

    def test_area_median_has_no_default(self):
        from real_validation.perception.quality import QualityThresholds
        with self.assertRaises(TypeError):
            QualityThresholds()          # 数据相关阈值必须显式提供


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_quality -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.perception.quality'`

- [ ] **Step 3: 实现 `real_validation/perception/quality.py`**

```python
"""在线单帧质量门控。

离线管线对"坏帧"的处置一律是**时间插值修复**(clean_outlier_skeletons /
clean_transition_npz / repair_masks),那需要未来帧,在线不可复现。
在线只能**拒帧**:verdict=reject 的帧不进模型、不更新 anchor，但仍写入隐藏评价流。

阈值分两类:
  - **数据相关**(area_median_px):无默认值，必须由调用方从 manifest 提供。
    理由:仓库里"mask 面积中位数"有 4 个互相矛盾的值(8562 white_on_blue /
    6718 outliers / 6323-7099 qc / 运行时重算)，而部署 checkpoint 用的 SAM2 mask
    一个统计都没有。在这里写死任何一个都会变成第 5 套约定。
  - **策略常量**(比例、行阈值、速度上界):有默认值，与离线判据对齐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc

Q_OK = "ok"
Q_DEGRADED = "degraded"
Q_REJECT = "reject"

R_EMPTY_MASK = "empty_mask"
R_AREA_LOW = "area_ratio_low"
R_AREA_HIGH = "area_ratio_high"
R_HEIGHT_LOW = "height_frac_low"
R_TOP_ROW_HIGH = "top_row_high"
R_SECOND_BLOB = "second_blob_present"
R_TIP_FIX_SKIPPED = "tip_fix_skipped"
R_NODE_STEP_HIGH = "node_step_high"
R_FRAME_STALE = "frame_stale"
R_REGISTRATION_DISPLACED = "registration_displaced"

_REJECT_REASONS = frozenset({
    R_EMPTY_MASK, R_AREA_LOW, R_AREA_HIGH, R_HEIGHT_LOW, R_TOP_ROW_HIGH,
    R_NODE_STEP_HIGH, R_FRAME_STALE, R_REGISTRATION_DISPLACED,
})


@dataclass(frozen=True)
class QualityThresholds:
    """area_median_px 无默认值 —— 它是数据相关量，必须显式提供。"""
    area_median_px: float
    area_ratio_min: float = 0.7
    area_ratio_max: float = 1.3
    min_height_frac: float = 0.15
    max_top_row: int = 20
    max_second_blob_ratio: float = 0.15
    max_node_step_px: float = 4.0
    max_frame_age_s: float = 0.5
    max_registration_displacement_px: float = 2.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.area_median_px) or self.area_median_px <= 0:
            raise ValueError("area_median_px 必须是正的有限值")
        if self.area_ratio_min > self.area_ratio_max:
            raise ValueError("area_ratio_min 不能大于 area_ratio_max")


@dataclass(frozen=True)
class FrameQuality:
    verdict: str
    reasons: tuple[str, ...] = ()
    flags: dict[str, Any] = field(default_factory=dict)


def _blob_stats(mask):
    """返回 (area, height, top_row, second_ratio)。无前景时 (0, 0, H, 0.0)。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height_total = binary.shape[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return 0, 0, height_total, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    largest = 1 + int(order[0])
    area = int(stats[largest, cv2.CC_STAT_AREA])
    box_height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    top_row = int(stats[largest, cv2.CC_STAT_TOP])
    second = float(areas[order[1]]) / float(area) if len(order) > 1 and area else 0.0
    return area, box_height, top_row, second


def assess_frame(mask, skeleton, skeleton_info: dict, thresholds: QualityThresholds,
                 *, prev_skeleton=None, frame_age_s: float | None = None,
                 registration_displacement_px: float | None = None) -> FrameQuality:
    """对单帧给出 ok / degraded / reject 判决与全部标志。

    flags 里的值全部是 Python 标量(不是 numpy 标量、不含 NaN)，因为它们会经
    io.atomic_write_json(allow_nan=False) 落进 run 目录。
    """
    reasons: list[str] = []
    area, box_height, top_row, second_ratio = _blob_stats(mask)
    frame_height = int(np.asarray(mask).shape[0])
    area_ratio = float(area) / float(thresholds.area_median_px)
    height_frac = float(box_height) / float(frame_height) if frame_height else 0.0

    if area == 0:
        reasons.append(R_EMPTY_MASK)
    else:
        if area_ratio < thresholds.area_ratio_min:
            reasons.append(R_AREA_LOW)
        if area_ratio > thresholds.area_ratio_max:
            reasons.append(R_AREA_HIGH)
        if height_frac < thresholds.min_height_frac:
            reasons.append(R_HEIGHT_LOW)
        if top_row > thresholds.max_top_row:
            reasons.append(R_TOP_ROW_HIGH)
        if second_ratio > thresholds.max_second_blob_ratio:
            reasons.append(R_SECOND_BLOB)

    if skeleton_info.get("tip_fix_requested") and not skeleton_info.get("tip_fix_applied"):
        reasons.append(R_TIP_FIX_SKIPPED)

    max_step = 0.0
    if prev_skeleton is not None:
        current = np.asarray(skeleton, dtype=np.float64)[:, :2]
        previous = np.asarray(prev_skeleton, dtype=np.float64)[:, :2]
        if current.shape != previous.shape:
            raise ValueError(f"骨架形状不同：{current.shape} != {previous.shape}")
        max_step = float(np.linalg.norm(current - previous, axis=1).max())
        if max_step > thresholds.max_node_step_px:
            reasons.append(R_NODE_STEP_HIGH)

    if frame_age_s is not None and float(frame_age_s) > thresholds.max_frame_age_s:
        reasons.append(R_FRAME_STALE)

    displacement = registration_displacement_px
    if displacement is not None:
        value = float(displacement)
        if not math.isfinite(value) or value > thresholds.max_registration_displacement_px:
            reasons.append(R_REGISTRATION_DISPLACED)

    if any(reason in _REJECT_REASONS for reason in reasons):
        verdict = Q_REJECT
    elif reasons:
        verdict = Q_DEGRADED
    else:
        verdict = Q_OK

    flags: dict[str, Any] = {
        "mask_area_px": int(area),
        "mask_area_ratio": float(area_ratio),
        "blob_height_frac": float(height_frac),
        "top_row": int(top_row),
        "second_blob_ratio": float(second_ratio),
        "tip_fix_applied": bool(skeleton_info.get("tip_fix_applied", False)),
        "tip_fix_reason": str(skeleton_info.get("tip_fix_reason", "")),
        "n_valid_rows": int(skeleton_info.get("n_valid_rows", 0)),
        "max_node_step_px": float(max_step),
        "frame_age_s": None if frame_age_s is None else float(frame_age_s),
        "registration_displacement_px": (
            None if displacement is None or not math.isfinite(float(displacement))
            else float(displacement)),
        "verdict": verdict,
    }
    return FrameQuality(verdict=verdict, reasons=tuple(reasons), flags=flags)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_quality -v`
Expected: 12 tests PASS

- [ ] **Step 5: 重跑 import 卫生 + 全部感知测试**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_import_hygiene tests.test_perception_parity \
                  tests.test_perception_registration tests.test_perception_quality -v
```
Expected: 全部 OK

- [ ] **Step 6: 询问用户后提交**

```bash
git add real_validation/perception/quality.py tests/test_perception_quality.py
git commit -m "feat(perception): 在线单帧质量门控(只拒帧;数据相关阈值不给默认值)"
```

---

### Task 7: 命令行感知探针 + 文档同步 + 终验

探针是 M0 的验收产物:它**不需要 GUI、不需要 checkpoint**,却把在线感知链每个算子都跑通,并给出 P2 采集协议所需的全部参数(实际分割参数、单帧耗时、坏帧率)。

**放在 `real_validation/perception_probe.py`(与 `main_validation.py` 同级)** —— 放 `scripts/real/` 则无法随 `real_validation/` 搬到 PC,"可移植感知探针"的目标直接落空;放 `real_validation/tools/` 则 `__package__` 引导的 `.parent.parent` 要改成 `.parent.parent.parent`,是容易错的地方。

**诚实边界(必须写进计划正文)**:`real_capture/.gitignore:3 = data/` → `git ls-files real_capture/data` 返回 **0** 个文件。**探针的真实数据验收只能在本机/服务器完成,CI 与他人 clone 无法复现**;且只有 `seq_20260627_163921` 有 derived 产物,另两条序列跑探针需先跑 `segment_batch.py`。

**Files:**
- Create: `real_validation/perception_probe.py`
- Create: `tests/test_perception_probe.py`
- Modify: `scripts/real/write_data_readme.py:159`
- Modify: `src/data/real/triangulation.py:5,7,52`
- Modify: `real_capture/data/derived/seq_20260627_163921/README.md:13`(若存在)
- Modify: `CLAUDE.md`(测试框架现状)

**Interfaces:**
- Consumes: Task 1–6 的全部 perception 模块
- Produces:
  - `perception_probe.run_probe(frames, background, segment_params, n_points, thresholds, reference=None)` → `dict`(含 `timing`/`quality`/`registration`/`overlay`)
  - CLI:`python perception_probe.py --source {dir,live} ...`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_perception_probe.py`:

```python
"""感知探针的离线验收。

合成帧写进临时目录 → 探针必须产出 overlay/timing/quality 三件产物。
真实数据的验收另有一个 skipUnless 测试（数据已 gitignore，CI 跑不到）。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "real_validation" / "perception_probe.py"


def _write_synthetic_sequence(directory: Path, count: int = 4):
    """写 count 帧合成 BGR PNG + 一张中值背景，返回 (frames_dir, background_path)。"""
    import cv2
    from tests.test_perception_parity import synthetic_bgr_scene

    frames_dir = directory / "cam0"
    frames_dir.mkdir(parents=True)
    frame, bg_gray = synthetic_bgr_scene()
    for index in range(count):
        shifted = np.roll(frame, index, axis=1)
        cv2.imwrite(str(frames_dir / f"{index:05d}.png"), shifted)
    background = directory / "bg_median.png"
    cv2.imwrite(str(background), bg_gray)
    return frames_dir, background


class ProbeTest(unittest.TestCase):
    def test_probe_produces_overlay_timing_and_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames_dir, background = _write_synthetic_sequence(root)
            out = root / "probe"
            completed = subprocess.run(
                [sys.executable, str(PROBE), "--source", "dir",
                 "--frames-dir", str(frames_dir), "--background", str(background),
                 "--n-points", "15", "--frames", "3", "--out", str(out)],
                cwd=REPO, capture_output=True, text=True, timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertTrue((out / "overlay.png").is_file())
            timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
            for key in ("segment_ms", "skeleton_ms", "quality_ms", "total_ms"):
                self.assertIn(key, timing)
                self.assertIn("mean", timing[key])
                self.assertIn("p90", timing[key])
                self.assertGreater(timing[key]["mean"], 0.0)
            self.assertEqual(timing["n_frames"], 3)

            records = [json.loads(line) for line in
                       (out / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertIn(record["verdict"], {"ok", "degraded", "reject"})
                self.assertIn("mask_area_ratio", record)

    def test_source_is_required(self):
        completed = subprocess.run([sys.executable, str(PROBE)], cwd=REPO,
                                   capture_output=True, text=True, timeout=120)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--source", completed.stderr)

    def test_no_builtin_default_frames_dir(self):
        """探针不得内置仓库路径默认值（否则在 PC 上会指向不存在的目录并静默失败）。"""
        source = PROBE.read_text(encoding="utf-8")
        self.assertNotIn("real_capture/data", source)
        self.assertNotIn("seq_20260627", source)


REAL_CAM0 = REPO / "real_capture/data/raw/seq_20260627_163921/cam0"
REAL_BG = REPO / "real_capture/data/derived/seq_20260627_163921/bg_median.png"
REAL_META = REPO / "real_capture/data/derived/seq_20260627_163921/segment_meta.json"


@unittest.skipUnless(REAL_CAM0.is_dir() and REAL_BG.is_file() and REAL_META.is_file(),
                     "真实采集数据不存在（已 gitignore；只能在服务器/本机验收）")
class ProbeOnRealDataTest(unittest.TestCase):
    def test_probe_runs_on_real_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "probe"
            completed = subprocess.run(
                [sys.executable, str(PROBE), "--source", "dir",
                 "--frames-dir", str(REAL_CAM0), "--background", str(REAL_BG),
                 "--segment-params", str(REAL_META), "--n-points", "15",
                 "--frames", "6", "--out", str(out)],
                cwd=REPO, capture_output=True, text=True, timeout=600)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
            # 采集节拍 ~5 fps（action_interval_s=0.2）→ 单帧总耗时必须远低于 200 ms
            self.assertLess(timing["total_ms"]["p90"], 200.0,
                            f"单帧 p90 {timing['total_ms']['p90']:.1f} ms 超过采集节拍预算")
            records = [json.loads(line) for line in
                       (out / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_probe -v`
Expected: FAIL — 探针文件不存在(`can't open file .../perception_probe.py`)

- [ ] **Step 3: 实现 `real_validation/perception_probe.py`**

```python
"""命令行感知探针：抓帧 → 分割 → 骨架 → 质量门控 → 叠加图 + 逐算子耗时。

不需要 GUI、不需要 checkpoint，却把在线感知链每个算子都跑通，并给出采集协议所需的
参数（实际分割参数、单帧耗时、坏帧率）。

用法（--source 必填；**不内置任何仓库路径默认值**，否则在 PC 上会指向不存在的目录）：

  # 离线：用已采集的一段帧（开发机没有相机时的唯一途径）
  python perception_probe.py --source dir \\
      --frames-dir <seq>/cam0 --background <derived>/bg_median.png \\
      [--segment-params <derived>/segment_meta.json] \\
      [--reference <derived>/bg_median.png] --n-points 15 --frames 12 --out <out>

  # 在线：从 RealSense 实时取流（需要 real_capture/ 并排存在 + pyrealsense2）
  python perception_probe.py --source live --background <bg.png> --frames 12 --out <out>

产物：overlay.png（叠加网格）/ timing.json（逐算子 mean+p90）/ quality.jsonl（逐帧标志）
      / registration.json（若给了 --reference）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # 支持复制目录后直接 ``python perception_probe.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "real_validation"

import numpy as np

from .perception.background import background_drift, load_median_background
from .perception.quality import QualityThresholds, assess_frame
from .perception.registration import estimate_registration, save_registration
from .perception.segmentation import segment_white_on_blue
from .perception.skeleton import extract_skeleton_2d

_SKELETON_COLOR = (255, 255, 0)   # BGR 青
_MASK_COLOR = (0, 0, 255)         # BGR 红


def _percentile(values, ratio: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), ratio)) if values else 0.0


def _stats(values) -> dict:
    return {"mean": float(np.mean(values)) if values else 0.0,
            "p90": _percentile(values, 90.0),
            "max": float(np.max(values)) if values else 0.0}


def list_frames(frames_dir) -> list[str]:
    """masks_repaired/ 之类目录里含子目录 → 必须 glob '*.png'，不能 os.listdir。"""
    files = sorted(glob.glob(os.path.join(str(frames_dir), "*.png")))
    if not files:
        raise FileNotFoundError(f"目录里没有 PNG：{frames_dir}")
    return files


def load_segment_params(path) -> dict:
    """从 derived/<seq>/segment_meta.json 读真实分割参数。

    ⚠️ 必须读这个文件而不是用代码默认值：批产用的是 val=100，而
    segment_white_on_blue 的默认是 val=120。
    """
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{path} 缺少 params 对象")
    return {key: params[key] for key in
            ("sat", "val", "diff", "dil", "open_k", "close_k",
             "min_area_frac", "min_h_frac") if key in params}


def draw_overlay(bgr, mask, skeleton, label: str):
    """mask 半透明红 + 骨架青线 + 末端圈 + 左上角文字（纯 cv2，无 matplotlib）。"""
    import cv2
    canvas = bgr.copy()
    tint = np.zeros_like(canvas)
    tint[mask > 0] = _MASK_COLOR
    canvas = cv2.addWeighted(tint, 0.22, canvas, 0.78, 0.0)
    points = np.asarray(skeleton, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) >= 2 and np.abs(skeleton).max() > 0:
        cv2.polylines(canvas, [points], False, _SKELETON_COLOR, 1, cv2.LINE_AA)
        for point in points.reshape(-1, 2):
            cv2.circle(canvas, tuple(int(v) for v in point), 2, _SKELETON_COLOR, -1)
        cv2.circle(canvas, tuple(int(v) for v in points[0, 0]), 5, (0, 255, 0), 1)
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def grid(tiles, columns: int = 4):
    """把若干同尺寸图拼成网格（不足处填黑）。"""
    if not tiles:
        raise ValueError("没有可拼接的图")
    height, width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    return canvas


def run_probe(frames_bgr, background_gray, *, segment_params: dict, n_points: int,
              thresholds: QualityThresholds, reference_gray=None) -> dict:
    """对一批 BGR 帧跑完整在线链，返回 {timing, quality, overlay, registration}。"""
    timing = {"segment_ms": [], "skeleton_ms": [], "quality_ms": [], "total_ms": []}
    quality_records = []
    tiles = []
    previous_skeleton = None

    for index, bgr in enumerate(frames_bgr):
        start = time.perf_counter()
        mark = time.perf_counter()
        mask = segment_white_on_blue(bgr, background_gray, **segment_params)
        timing["segment_ms"].append((time.perf_counter() - mark) * 1e3)

        mark = time.perf_counter()
        skeleton, info = extract_skeleton_2d(mask, n_points, tip_fix=True, return_info=True)
        timing["skeleton_ms"].append((time.perf_counter() - mark) * 1e3)

        mark = time.perf_counter()
        quality = assess_frame(mask, skeleton, info, thresholds,
                               prev_skeleton=previous_skeleton)
        timing["quality_ms"].append((time.perf_counter() - mark) * 1e3)
        timing["total_ms"].append((time.perf_counter() - start) * 1e3)

        record = {"frame": index, **quality.flags,
                  "reasons": list(quality.reasons)}
        quality_records.append(record)
        tiles.append(draw_overlay(bgr, mask, skeleton,
                                  f"#{index} {quality.verdict}"))
        if quality.verdict != "reject":
            previous_skeleton = skeleton

    registration = None
    if reference_gray is not None and frames_bgr:
        import cv2
        live_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
        registration = estimate_registration(
            reference_gray, live_gray,
            max_displacement_px=thresholds.max_registration_displacement_px)

    verdicts = [record["verdict"] for record in quality_records]
    return {
        "timing": {key: _stats(values) for key, values in timing.items()} |
                  {"n_frames": len(frames_bgr)},
        "quality": quality_records,
        "verdict_counts": {name: verdicts.count(name)
                           for name in ("ok", "degraded", "reject")},
        "overlay": grid(tiles),
        "registration": registration,
    }


def _load_frames_from_dir(frames_dir, count: int):
    import cv2
    files = list_frames(frames_dir)
    picked = [files[i] for i in
              np.linspace(0, len(files) - 1, min(count, len(files))).astype(int)]
    frames = []
    for path in picked:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取帧：{path}")
        frames.append(image)
    return frames


def _load_frames_from_camera(count: int, warmup: int = 5):
    """从 RealSense 取 count 帧。需要 real_capture/ 并排存在 + pyrealsense2。"""
    sys.path.append(str(Path(__file__).resolve().parent.parent / "real_capture"))
    import pyrealsense2 as rs  # noqa: E402  延迟 import：只有 live 模式需要
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline.start(config)
    try:
        frames = []
        for index in range(count + warmup):
            composite = pipeline.wait_for_frames(2000)
            color = composite.get_color_frame()
            if color is None:
                continue
            if index >= warmup:
                frames.append(np.array(color.get_data()))
        if not frames:
            raise RuntimeError("相机没有返回任何 color frame")
        return frames
    finally:
        pipeline.stop()


def main() -> int:
    import cv2
    parser = argparse.ArgumentParser(description="在线感知链探针（无 GUI、无 checkpoint）")
    parser.add_argument("--source", required=True, choices=("dir", "live"),
                        help="dir=用已采集的一段帧；live=从 RealSense 实时取流")
    parser.add_argument("--frames-dir", help="--source dir 时必填：含 *.png 的目录")
    parser.add_argument("--background", required=True, help="中值背景灰度图 PNG")
    parser.add_argument("--segment-params",
                        help="derived/<seq>/segment_meta.json；不给则用代码默认（val=120，"
                             "与批产的 val=100 不同，仅供快速冒烟）")
    parser.add_argument("--reference", help="位姿注册的基准灰度图；不给则跳过注册")
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--area-median-px", type=float, default=None,
                        help="mask 面积中位数；不给则用本批帧自身的中位数（仅冒烟用，"
                             "正式验收必须从 manifest 提供）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.source == "dir":
        if not args.frames_dir:
            parser.error("--source dir 需要 --frames-dir")
        frames = _load_frames_from_dir(args.frames_dir, args.frames)
    else:
        frames = _load_frames_from_camera(args.frames)

    background = load_median_background(args.background)
    segment_params = (load_segment_params(args.segment_params)
                      if args.segment_params else {})
    reference = (load_median_background(args.reference) if args.reference else None)

    area_median = args.area_median_px
    if area_median is None:
        areas = [float(segment_white_on_blue(frame, background, **segment_params).sum())
                 for frame in frames]
        positive = [value for value in areas if value > 0]
        area_median = float(np.median(positive)) if positive else 1.0
        print(f"[probe] --area-median-px 未提供，用本批中位数 {area_median:.0f} px"
              f"（仅冒烟；正式验收须从 deploy_manifest 提供）")

    thresholds = QualityThresholds(area_median)
    result = run_probe(frames, background, segment_params=segment_params,
                       n_points=args.n_points, thresholds=thresholds,
                       reference_gray=reference)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), result["overlay"])
    with open(out / "timing.json", "w", encoding="utf-8") as stream:
        json.dump(result["timing"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    with open(out / "quality.jsonl", "w", encoding="utf-8") as stream:
        for record in result["quality"]:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if result["registration"] is not None:
        save_registration(result["registration"], out / "registration.json")
        print(f"[probe] 配准 displacement={result['registration'].displacement_px:.2f} px "
              f"ok={result['registration'].ok} reason={result['registration'].reason}")
    if reference is not None:
        print(f"[probe] 背景漂移中位数="
              f"{background_drift(reference, background):.2f} 灰阶")

    print(f"[probe] {result['timing']['n_frames']} 帧  "
          f"total p90={result['timing']['total_ms']['p90']:.1f} ms  "
          f"(segment {result['timing']['segment_ms']['mean']:.1f} / "
          f"skeleton {result['timing']['skeleton_ms']['mean']:.1f} / "
          f"quality {result['timing']['quality_ms']['mean']:.1f} ms 均值)")
    print(f"[probe] 判决分布 {result['verdict_counts']}")
    print(f"[probe] 产物写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行探针测试**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_perception_probe -v`
Expected: 3 tests PASS(本机有真实数据时 4 个,且真实数据那条会打印单帧 p90 耗时)

- [ ] **Step 5: 在真实数据上手工跑一次(M0 的验收产物)**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
SEQ=real_capture/data/raw/seq_20260627_163921
DER=real_capture/data/derived/seq_20260627_163921
python real_validation/perception_probe.py --source dir \
    --frames-dir $SEQ/cam0 --background $DER/bg_median.png \
    --segment-params $DER/segment_meta.json --reference $DER/bg_median.png \
    --n-points 15 --frames 12 --out output/perception_probe
```
Expected(记录进验收):叠加图里 12 帧骨架都贴合臂体;`total p90` **远低于 200 ms**(采集节拍 `action_interval_s=0.2`);判决分布以 `ok` 为主;配准 `displacement < 1 px`(同一序列自比)。

- [ ] **Step 6: 同步文档里的旧路径(4 处,漏了会永久漂移)**

(a) `scripts/real/write_data_readme.py:159` —— 它是 `data/real_seq/README.md` 的**生成器**,不改则重跑会写回旧路径:
```python
        "## 骨架化(real_validation/perception/skeleton.py;src/utils/skeleton_2d.py 为薄壳)",
```

(b) `src/data/real/triangulation.py` 第 5、7、52 行的 `src/utils/skeleton_2d` 字样 → 改为 `real_validation/perception/skeleton.py`(薄壳仍在 `src/utils/skeleton_2d.py`)。

(c) 若 `real_capture/data/derived/seq_20260627_163921/README.md` 存在,修第 13 行的坏路径:`scripts/real/segment_rd/segment_batch.py` → `scripts/real/segment_batch.py`。

(d) `CLAUDE.md` 的 "**No formal test framework** — validation uses Jupyter notebooks..." 一句改为:
```markdown
- **Tests**: `unittest`(无 pytest)。4 个测试文件:`tests/test_real_validation_core.py`(20 个契约测试)、`tests/test_perception_parity.py`(感知迁移冻结参考)、`tests/test_import_hygiene.py`(子进程断言 real_validation 包根 stdlib-only)、`tests/test_perception_{registration,quality,probe}.py`。运行:`python -m unittest discover -s tests -v`。此外仍用 `notebooks/` 与 `scripts/evaluation/` 做实验级验证
```

- [ ] **Step 7: 终验 —— 全量测试 + 卫生 + 向后兼容**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_real_validation_core -v 2>&1 | tail -3     # 期望 20 passed
python -m unittest tests.test_perception_parity tests.test_import_hygiene \
                  tests.test_perception_registration tests.test_perception_quality \
                  tests.test_perception_probe -v 2>&1 | tail -3
python -c "import real_validation, sys; print('root deps:', sorted(k for k in sys.modules if k.split('.')[0] in ('torch','cv2','scipy','PyQt5','pyqtgraph')))"
python -c "import src.data.real, src.utils.skeleton_2d; print('shim import ok')"
python scripts/real/segment_batch.py --help > /dev/null && echo "segment_batch ok"
python scripts/real/masks_to_transition_npz.py --help > /dev/null && echo "masks_to_npz ok"
python scripts/real/compare_skeleton_methods.py --help > /dev/null && echo "compare ok"
```
Expected:第 1 条 `OK (20 tests)` 或 `Ran 20 tests ... OK`;第 2 条全 OK;第 3 条打印 `root deps: []`;后 4 条全部成功。

- [ ] **Step 8: 改计划文件名以匹配 P1a 范围,询问用户后提交**

```bash
cd /Data5/ddf/projects/SelfSoftRobot
git mv docs/superpowers/plans/2026-07-28-real-validation-p1-perception-and-contracts.md \
       docs/superpowers/plans/2026-07-28-real-validation-p1a-perception-migration.md
git add real_validation/perception_probe.py tests/test_perception_probe.py \
        scripts/real/write_data_readme.py src/data/real/triangulation.py CLAUDE.md \
        docs/superpowers/plans/
git commit -m "feat(perception): 命令行感知探针(离线可跑)+ 文档路径同步 + P1a 终验"
```

---

## Self-Review(计划自审)

**1. Spec 覆盖(P1a = spec §9 的 M0)**

| spec 要求 | 落在哪 |
|---|---|
| 感知实现迁到 `real_validation/perception/` | Task 1(skeleton)、Task 3(segmentation) |
| `src/` 改薄壳、签名不变 | Task 1 Step 5、Task 3 Step 4;向后兼容由 Task 1 Step 7 / Task 3 Step 7 / Task 7 Step 7 三处验证 |
| 修 B13(tip_fix 静默跳过) | Task 2 |
| `perception/background.py` | Task 5 Step 3 |
| `perception/registration.py`(只检测不 warp) | Task 5 Step 4 |
| `perception/quality.py`(spec §6.1 的 8 条判据) | Task 6;逐条对应 `R_*` 常量 |
| 命令行探针,`--source dir` 可离线跑 | Task 7 |
| T1 parity 测试 | Task 1 Step 2(合成 + 真实 mask 双轨) |
| T9 import 卫生(spec §7.2 新增) | Task 4 |

spec §6.1 的 8 条判据逐一核对:mask 面积比 ✓(`R_AREA_LOW/HIGH`)、height/H ✓(`R_HEIGHT_LOW`)、top_row ✓(`R_TOP_ROW_HIGH`)、次大连通区 ✓(`R_SECOND_BLOB`)、tip_fix 静默跳过 ✓(`R_TIP_FIX_SKIPPED`)、节点位移 ✓(`R_NODE_STEP_HIGH`)、frame_age ✓(`R_FRAME_STALE`)、配准残差 ✓(`R_REGISTRATION_DISPLACED`)。**无遗漏。**

**2. 占位扫描**:全文无 TBD / TODO / "implement later" / "similar to Task N"。每个代码步骤都是可直接粘贴的完整实现;每个测试步骤都有完整测试代码与 Expected。

**3. 类型一致性**:`extract_skeleton_2d(..., return_info=False)` 在 Task 2 定义、Task 6/7 消费 —— `info` 的键名(`tip_fix_requested`/`tip_fix_applied`/`tip_fix_reason`/`n_foreground_px`/`n_valid_rows`)三处一致。`QualityThresholds` 的字段名在 Task 6 定义、Task 7 消费(`max_registration_displacement_px`)一致。`RegistrationResult.displacement_px` 在 Task 5 定义、Task 7 打印一致。`assess_frame` 的签名(4 位置参数 + 3 keyword-only)在 Task 6 定义、Task 7 调用一致。

**4. 歧义扫描**:`area_median_px` 无默认值这一条在 Task 6 的 Interfaces、实现、`test_area_median_has_no_default` 三处显式;探针的 `--area-median-px` 缺省行为(用本批中位数 + 打印警告)明确标为"仅冒烟"。

**5. 范围**:7 个任务、7 次提交,全程不碰 torch、不需要 checkpoint、不需要新数据。可独立验收。

## 已知缺口(P1a 交付后仍未解决,记录以免遗忘)

| 缺口 | 归属 |
|---|---|
| `live_anchor.py`(帧 → Anchor)需要真 `pc_center/pc_scale` | P3(M4);测试可用 `type("Geometry", (), {"pc_center": torch.zeros(3), "pc_scale": torch.ones(3)})()` stub(`tests/test_real_validation_core.py:167-169` 已有先例);**必须显式 `n_points = descriptor.n_nodes`(=15),不能吃 `extract_skeleton_2d` 的默认 31**;归一化必须与 `offline_anchor.py:55` 共享同一函数,否则单位链分叉 |
| 探针真实数据验收不可在 CI 复现(`real_capture/data/` 已 gitignore,`git ls-files` 返回 0 个文件) | 本计划已显式承认;合成路径覆盖全部代码分支 |
| SAM2 mask 的面积统计不存在 → `area_median_px` 对 SAM2 数据无权威值 | P2 按 §8 协议改用 white_on_blue 重采后一并产出 |
| `quality.py` 的真实失效模式样本(静态段截断 / 半 mask / 手污染)未纳入测试 | P2 采集后补;现成可提交的量化依据是 `outliers.txt`(420 B,53 帧)+ `qc/metrics.txt`(8 帧) |
