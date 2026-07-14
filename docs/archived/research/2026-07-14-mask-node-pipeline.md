# 实物数据 mask→node 处理管线(每步算法 + 各 QC 列含义)

> 把"原始照片"一路变成"训练用的骨架 npz"的每一步算法、以及 `full_chain_*.png` 各列到底代表什么。
> 配套工作流见 [`2026-07-10-real-data-2d-workflow.md`](2026-07-10-real-data-2d-workflow.md)。更新 2026-07-14。

## 0. 管线总览

```
real_capture/.../cam0/<f>.png  (原图)
   │ ① 分割(white_on_blue)
   ▼
derived/<seq>/masks/<f>.png            (RAW mask —— 有腐败: 半mask/缺块/手污染)
   │ ② repair_masks.py  (mask 级, 3 步: 手帧→静态共识→动作段时间插值)
   ▼
derived/<seq>/masks_repaired/<f>.png   (REPAIRED mask —— 形态预测的 GT)
   │ ③ 骨架化 skeleton_2d.extract_skeleton_2d (逐行质心+弧长重采样+tip_fix)
   ▼
raw skeleton (node, col,row)           (从 repaired mask 提的骨架, 清洗前)
   │ ④ clean_transition_npz.py  (node 级: 整帧离群插值 + 静态段共识)
   ▼
data/real_seq/<seq>_n15_rep_clean/*.npz  (CLEANED skeleton —— 训练吃这个)
```

两条修复轨道(独立): **mask 轨道**(②,产 masks_repaired,作形态 GT) + **node 轨道**(④,产清洗骨架,作训练 state)。
当前默认 **N=15 节点**(已验证降节点误差不大)。

---

## 1. 每一步算法

### ① 分割 — `white_on_blue`(已固化, 在 segment_rd/segment_batch)
白半透明硅胶臂 / 蓝背景 / 白气管 → 阈值出二值 mask。产物 `masks/`。**这一步会出腐败**(见 §3)。

### ② mask 级修复 — `scripts/real/repair_masks.py`(3 步, 顺序执行)
| 步骤 | 函数 | 修什么 | 怎么修 |
|---|---|---|---|
| ②a 手帧 | `repair_hand_frames` | **整帧**手污染/管茬/臂缺失(area>1.3×中位 或 臂没到顶部行>20) | 找最近 **clean** 邻帧(顶部行≤20 且 0.7~1.3×中位 完整臂), 逐行 [min,max] col 按 α 线性插值**整帧替换**(跟随臂运动, 手被剔除) |
| ②b 静态段 | `repair_static_segment` | **静态段**(关节以上)顶部截断/抖动 | 逐行跨帧 [min,max] col **中位共识**, 每帧静态行替换为共识宽 |
| ②c 动作段 | `repair_actuated` | **动作段**(关节以下)半mask/缺块 | **时间插值**(主): 那块在单帧不可见(半透明), 从邻帧补——边重合配准(不用腐败质心); **宽度补全**(辅, 无健康邻帧时单边扩展) |

关节行由**宽度凸起**定位(管-臂合并处最宽, row~91)。三步都用 `--hand/--no-hand`、`--actuated/--no-actuated` 开关(默认全开)。

> 为什么手帧要时间插值: 手在单帧里和臂合并(或臂被手遮), **无法从该帧分割出臂**; 只能从邻帧(臂可见)重建。partial-hand 帧(area 刚低于阈值)也判 needs_fix(用顶部行判别, 不只看 area), 否则会污染插值。

### ③ 骨架化 — `src/utils/skeleton_2d.py::extract_skeleton_2d`
- **逐行质心**: 每行白像素列均值, 底→顶排列。
- **弧长重采样**到 N 点(默认 15)。
- **tip_fix**(末端 corner 修复, `_perpendicular_tip_fix`): 弯管 cap 倾斜时逐行质心把末端 node0 落到角落 → 改用"垂直于局部轴的尖端切片质心"=cap 中点。

### ④ node 级清洗 — `scripts/real/clean_transition_npz.py`(2 步)
| 步骤 | 函数 | 修什么 | 怎么修 |
|---|---|---|---|
| ④a 整帧离群 | `clean_outlier_skeletons` | 整帧骨架偏离时间中位 >80px(管茬等) | 时间插值整帧骨架 |
| ④b 静态段共识 | `stabilize_static_region` | 静态段节点抖动 | 关节绝对位置锚定(每帧关节=离绝对位最近), 静态段弧长重采样→跨帧中位共识→映射回 |

---

## 3. 三类 mask 腐败 + 修复状态

| 腐败类型 | 典型帧 | 现象 | 谁修 | 状态 |
|---|---|---|---|---|
| 静态段截断 | f4080 | 顶部宽 17(应 31) | ②b `repair_static_segment` | ✅ |
| 动作段半mask | f4902 | 中段 r132-156 只剩右半(宽19, 应31) | ②c `repair_actuated` | ✅ area 6050→8669 |
| 整帧手污染/臂缺失 | f2330, f2316 | 手占据/臂没到顶(area 28000 或 mask 只在底部) | ②a `repair_hand_frames` | ✅(新, 整帧插值) |

---

## 4. ★ `full_chain_*.png` 各列含义(回答"哪列是最终结果")

`viz_qc.py dataset` 产的全链路图, 每帧 4 列(左→右):

| 列 | 标注 | 内容 | 是"最终"吗 |
|---|---|---|---|
| **列1** | `RAW mask` | 原始分割 mask(含腐败) | ❌ 原始, 仅对比 |
| **列2** | `{src} mask (=pipeline input)` | repair_masks 后的 mask(repaired, ②的产物) | ✅ **形态预测的 GT**(最终 mask) |
| **列3** | `skeleton from {src} mask (cyan)` | 从 repaired mask 提的骨架(③+tip_fix, **清洗前**) | 中间(节点轨道原始) |
| **列4** | `CLEANED skeleton from npz (yellow)` | clean 后的骨架(④的产物, npz 里存的就是它) | ✅ **训练用的最终骨架(node)** |

**一句话: 形态(shape)预测的 GT = 列2(REPAIRED mask); 训练 state 用的骨架 = 列4(CLEANED skeleton)。**
列1→列2 = mask 修复效果; 列3→列4 = 骨架清洗效果(静态段共识等)。
列3 vs 列4 通常差别很小(主要是静态段节点被共识稳定); 列2 vs 列1 才是 mask 修复的主战场(看腐败帧是否补全)。

---

## 5. 坐标空间(易混, 再强调)
- 骨架/预测/mask 都在**像素 [col,row,0]**(免标定, z=0 平面假设)。
- **整体形态误差只能 px**(只有图像 GT)。**末端误差 px + mm**(NDI 独立度量 GT, 仿射自标定 px→mm)。
- 不用相机矩阵投影(无度量 3D/内参)。

## 6. 关键参数(默认)
- `--n-points 15`(节点数, 已验证误差不大; 节点索引全按 N 分数自适应)。
- `--tip-fix`(末端 corner 修复, 开)。
- repair_masks: `--hand`/`--actuated`(默认全开)。
- 训练数据 = `data/real_seq/<seq>_n15_rep_clean/`(repaired mask + 15 节点 + clean)。
