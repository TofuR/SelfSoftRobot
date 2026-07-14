# sam2/ — SAM2 视频分割(实物 mask 修复, 持久化)

本目录是 SAM2 的**持久化**工作区(从 /tmp 迁出, 避免 /tmp 被清)。用于用 SAM2 **视频模式**
修复实物 mask 的"半mask/缺块"——那块硅胶在单帧图里看不见(半透明), image 模式补不回, 但视频
模式从邻帧传播 mask 能补回(实测 f4902 r132-156 宽 19→32, area 6050→8860, 比启发式时间插值更干净)。

## 目录结构

```
sam2/
├── segment_video.py     # 视频分割脚本(自包含, 顶部设 SAM2_HOME)  ← 入库
├── README.md            # 本文件                                 ← 入库
├── sam2_src/            # SAM2 包源码(editable install 来源)      ← gitignore(大)
├── checkpoints/         # sam2.1_hiera_tiny.pt (149M)            ← gitignore(*.pt)
├── masks/               # 输出 mask(QC + 各窗口)                 ← gitignore
└── _jpeg_tmp/           # SAM2 要求的连续名 JPEG 临时帧           ← gitignore
```

## 一次性安装(已完成; 重装见下)

```
# 包: editable 装自本目录源码(指向持久路径, 不依赖 /tmp)
pip install -e sam2/sam2_src --no-deps --no-build-isolation
# checkpoint: sam2/checkpoints/sam2.1_hiera_tiny.pt (已从 /tmp 拷入)
```

`import sam2` 需要 `SAM2_HOME=sam2/sam2_src`——**已在 segment_video.py 顶部 setdefault**, 直接跑即可。

## 用法

对一段帧做视频分割(用一个干净锚帧的 mask 作 prompt, 向前传播):

```bash
CUDA_VISIBLE_DEVICES=2 python sam2/segment_video.py \
    --seq seq_20260627_163921 --start 4880 --end 4930 \
    --anchor 4904 --anchor-mask-dir real_capture/data/derived/seq_20260627_163921/masks_repaired \
    --out sam2/masks/seq_20260627_163921_win4880-4930
```
- `--anchor`: 干净锚帧(用其 mask 引导, 比单纯 box 稳)。
- `--anchor-mask-dir`: 锚帧 mask 来源(`masks_repaired` 里的干净帧, 或 `masks`)。
- 输出: `<out>/<NNNNN>.png` (0/255, 与 cam0 同名) + `qc_sam2video.png` + `area_curve.txt`。
- 注意: SAM2 视频默认**向前**传播(锚帧→end)。要覆盖锚帧之前的帧, 把锚帧放窗口起点, 或分段。

## 全序列 10214 帧 — `segment_video_full.py`(分块双向, 多 GPU, 断点续跑)

`segment_video.py` 只做单窗口前向; `segment_video_full.py` 扩展到全序列:

- **分块**(默认每块 200 帧), 块内选一个**干净锚帧**(从 `masks_repaired`: 顶部行≤20 且 area∈[0.7,1.3]×中位, 离块中心最近)。
- **双向传播**: 官方 `propagate_in_video(reverse=False)` 前向 100 + `(reverse=True)` 反向 100, 一个锚帧覆盖整块(无需重叠拼接)。
- **块间隔离**(每块独立 `init_state`): 一块漂移/失败不污染其他块; 失败块记 `failures.txt` 继续。
- **多 GPU 分片**: `--shards N --shard k` 取 `chunk_idx % N == k`; 各分片写同一 out 目录, 帧不重叠。
- **断点续跑**: 块内所有输出帧已存在则跳过。
- **输出单独保存, 不覆盖** `masks_repaired`: `sam2/masks/<seq>_full/<NNNNN>.png` + `area_curve.txt` + `failures.txt`。

```bash
# 单卡
CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921
# 两卡并行(快一倍)
CUDA_VISIBLE_DEVICES=0 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 0 &
CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 1 &
# 小样冒烟(前 300 帧)
CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --end-frame 299
```

实测: 10214 帧 / 2 卡(GPU0+GPU3) 约 15-30 分钟; 锚帧 f4900 类腐败帧 f4902 area 恢复 8659(≈启发式 8669)。

## 对比图 — `compare_masks.py`(之前 mask vs SAM2 mask)

三列并排(每帧): `[RAW | PREV=repaired | SAM2]` 叠原图, 标 area + 顶部行 + IoU(prev∩sam);
另产全序列 area 散点(prev_area vs sam_area + 1:1 线 + 差值直方图)量化 SAM2 在哪改了 mask。

```bash
# 全套(三列对比图 + area 散点)
python sam2/compare_masks.py --seq seq_20260627_163921
# 只看指定帧
python sam2/compare_masks.py --seq seq_20260627_163921 --frames 4080,4902,2330,2316,100
# 只画全序列 area 散点(量化)
python sam2/compare_masks.py --seq seq_20260627_163921 --scatter-only
```

输出: `sam2/masks/<seq>_full/qc/compare_raw_prev_sam.png` + `area_scatter_prev_vs_sam.png`。
也可用通用 `scripts/real/viz_qc.py mask-compare --tag-a prev --tag-b sam2` 画两列版。

## 已知坑(API, 已在脚本里处理)

1. **Hydra 配置**: `build_sam2_video_predictor(config_file=...)` 的 config_file 须相对 configs 根
   (`sam2.1/sam2.1_hiera_t.yaml`), 且调用前 `GlobalHydra.instance().clear()` + `initialize_config_dir`。
2. **只吃 JPEG**: SAM2 `load_video_frames` 硬编码 `.jpg/.jpeg`; 脚本自动把目标帧转连续名 JPEG。
3. **输出是 logits 非 bool**: `propagate_in_video` 返回 `(num_objs,1,H,W)` float logits, 需 `(>0)` 阈值 + `squeeze()` 到 (H,W)。
4. `_C` 未编译 → `fill_holes_in_mask_scores` 被跳过(官方: 可忽略, 不影响结果)。

## 与启发式修复(repair_masks.py)的关系

- `repair_masks.py`(启发式时间插值): 已在全 10214 帧跑过, f4902 类半mask 修好(area 6050→8669),
  在 `masks_repaired/`, **快、已就绪**。
- SAM2 视频(本目录): 更干净(area 8860 vs 8669)、无需手写规则, 但需分段传播脚本跑全量。
- 二者择一或组合均可。当前训练数据 `_rep_clean` 用的是启发式 `masks_repaired`。
