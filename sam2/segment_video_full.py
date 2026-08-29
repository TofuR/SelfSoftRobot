"""sam2/segment_video_full.py — SAM2 视频分割全序列(10214 帧), 分块双向传播。

为什么: 实物 mask 的"半mask/缺块"在单帧图里看不见(半透明硅胶), image 模式补不回; SAM2 视频
从邻帧传播 mask 能补回。segment_video.py 只做单窗口前向; 本脚本扩展到**全序列**:
  - 分块(默认每块 200 帧),优先读取prepare_sam2_anchors生成的动作无关质量清单；
    旧序列没有清单时才回退到顶部/面积启发式。
  - 块内**双向传播**: 官方 propagate_in_video(reverse=False) 前向 max=100 + (reverse=True) 反向
    max=100, 一个锚帧覆盖整块(无需重叠拼接)。
  - 块间隔离(每块独立 init_state): 各块独立传播; 失败块记 failures_shardK.txt 继续。
  - 多 GPU 分片: --shards N --shard k 取 chunk_idx % N == k 的块(各分片写同一 out 目录,
    帧不重叠)。
  - 断点续跑: 块内所有输出帧已存在则跳过。

输出**单独保存, 不覆盖候选mask**: sam2/masks/<seq>_full/<NNNNN>.png + area_curve.txt
+ failures_shardK.txt + run_meta + candidate/SAM2对比QC。

用法(单卡):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921

两卡并行(快一倍):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 0 &
  CUDA_VISIBLE_DEVICES=0 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 1 &

小样冒烟(前 300 帧, 验证双向传播):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --end-frame 299
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import traceback

# ---- SAM2_HOME 必须在 import sam2 前设(指向持久 sam2_src) ----
HERE = os.path.dirname(os.path.abspath(__file__))
SAM2_SRC = os.path.join(HERE, "sam2_src")
os.environ.setdefault("SAM2_HOME", SAM2_SRC)
sys.path.insert(0, SAM2_SRC)

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "checkpoints", "sam2.1_hiera_tiny.pt")
CONFIG_DIR = os.path.join(SAM2_SRC, "sam2", "configs")
CONFIG_FILE = "sam2.1/sam2.1_hiera_t.yaml"


def build_predictor(device):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1")
    return build_sam2_video_predictor(config_file=CONFIG_FILE, ckpt_path=CKPT, device=device)


def mask_stats(m):
    """返回 (area, top_row)。top_row = 最上方白像素行(臂到顶→0; 缺顶→大)。空 mask→(0, H)。"""
    ys, _ = np.where(m > 0)
    if len(ys) == 0:
        return 0, m.shape[0]
    return int(ys.size), int(ys.min())


def global_median_area(anchor_mask_dir, n_total, sample_step=10):
    """跨序列抽样(每 sample_step 帧)算 area 中位, 作干净判据基准。"""
    areas = []
    for f in range(0, n_total, sample_step):
        p = os.path.join(anchor_mask_dir, f"{f:05d}.png")
        if not os.path.isfile(p):
            continue
        m = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        a, _ = mask_stats(m)
        if a > 0:
            areas.append(a)
    return float(np.median(areas)) if areas else 8000.0


def load_anchor_manifest(path):
    """读取 prepare_sam2_anchors.py 的逐帧质量；不存在时返回空字典。"""
    if not path or not os.path.isfile(path):
        return {}
    result = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(row["frame"])
                result[frame] = {
                    "quality": float(row.get("quality", 0.0)),
                    "selected": bool(int(row.get("selected", 0))),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return result


def select_anchor(anchor_mask_dir, chunk_frames, med_area, anchor_manifest=None):
    """块内选锚帧: clean(顶部行≤20 且 0.7~1.3×med) 中离块中心最近; 无 clean→area 最接近 med。
    返回 (anchor_frame, anchor_mask)。无可用→(None, None)。"""
    center = chunk_frames[len(chunk_frames) // 2]
    stats = {}
    for f in chunk_frames:
        p = os.path.join(anchor_mask_dir, f"{f:05d}.png")
        if not os.path.isfile(p):
            continue
        m = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        a, top = mask_stats(m)
        if a > 0:
            stats[f] = (a, top, m)
    if not stats:
        return None, None
    lo, hi = 0.7 * med_area, 1.3 * med_area
    manifest = anchor_manifest or {}
    preselected = {f: v for f, v in stats.items()
                   if manifest.get(f, {}).get("selected", False)}
    if preselected:
        anchor = max(preselected, key=lambda f: manifest[f]["quality"])
        return anchor, preselected[anchor][2]
    scored = {f: v for f, v in stats.items() if manifest.get(f, {}).get("quality", 0) > 0}
    if scored:
        anchor = max(scored, key=lambda f: manifest[f]["quality"])
        return anchor, scored[anchor][2]
    clean = {f: v for f, v in stats.items() if v[1] <= 20 and lo <= v[0] <= hi}
    pool = clean if clean else stats
    anchor = min(pool, key=lambda f: abs(f - center))
    return anchor, pool[anchor][2]


def prepare_jpeg_dir(cam0, chunk_frames, jpeg_dir):
    """SAM2 load_video_frames 只吃 .jpg; 拷块内帧为连续名 JPEG。缺原图→False。"""
    os.makedirs(jpeg_dir, exist_ok=True)
    for i, f in enumerate(chunk_frames):
        img = cv2.imread(os.path.join(cam0, f"{f:05d}.png"))
        if img is None:
            return False
        cv2.imwrite(os.path.join(jpeg_dir, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def write_mask(out_dir, frame, m):
    cv2.imwrite(os.path.join(out_dir, f"{frame:05d}.png"), (m.astype(np.uint8)) * 255)


def chunk_done(out_dir, chunk_frames):
    """块内所有输出帧已存在→True(断点续跑跳过)。"""
    return all(os.path.isfile(os.path.join(out_dir, f"{f:05d}.png")) for f in chunk_frames)


def _overlay_mask(image_bgr, mask, color):
    out = image_bgr.copy()
    tint = out.copy(); tint[mask > 0] = color
    cv2.addWeighted(tint, 0.38, out, 0.62, 0, dst=out)
    return out


def save_sam2_qc(cam0, candidate_dir, sam_dir, shard, n=12):
    """保存本分片 candidate→SAM2 对比与面积曲线，不依赖旧静态段修复。"""
    frames = []
    for path in sorted(glob.glob(os.path.join(sam_dir, "*.png"))):
        frame = int(os.path.splitext(os.path.basename(path))[0])
        if os.path.isfile(os.path.join(candidate_dir, f"{frame:05d}.png")):
            frames.append(frame)
    if not frames:
        return
    qc_dir = os.path.join(sam_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    pick = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    rows = []
    areas = []
    for index in pick:
        frame = frames[index]
        image = cv2.imread(os.path.join(cam0, f"{frame:05d}.png"))
        candidate = (cv2.imread(os.path.join(candidate_dir, f"{frame:05d}.png"),
                                cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        sam = (cv2.imread(os.path.join(sam_dir, f"{frame:05d}.png"),
                          cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if image is None:
            continue
        left = _overlay_mask(image, candidate, (0, 0, 255))
        right = _overlay_mask(image, sam, (0, 255, 0))
        inter = np.logical_and(candidate, sam).sum()
        union = np.logical_or(candidate, sam).sum()
        iou = float(inter / union) if union else 0.0
        cv2.putText(left, f"f{frame} candidate A={int(candidate.sum())}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, f"SAM2 A={int(sam.sum())} IoU={iou:.3f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([left, right]))
    if rows:
        cv2.imwrite(os.path.join(qc_dir, f"compare_candidate_sam2_shard{shard}.png"),
                    np.vstack(rows))
    for frame in frames:
        candidate = cv2.imread(os.path.join(candidate_dir, f"{frame:05d}.png"),
                               cv2.IMREAD_GRAYSCALE)
        sam = cv2.imread(os.path.join(sam_dir, f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if candidate is not None and sam is not None:
            areas.append((frame, int((candidate > 127).sum()), int((sam > 127).sum())))
    if areas:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        values = np.asarray(areas)
        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax[0].plot(values[:, 0], values[:, 1], label="candidate", lw=.7)
        ax[0].plot(values[:, 0], values[:, 2], label="SAM2", lw=.7)
        ax[0].set_ylabel("area [px]"); ax[0].legend(); ax[0].grid(alpha=.25)
        ax[1].plot(values[:, 0], values[:, 2] - values[:, 1], lw=.7)
        ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel("SAM2-candidate")
        ax[1].set_xlabel("frame"); ax[1].grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"area_candidate_sam2_shard{shard}.png"), dpi=120)
        plt.close(fig)


def process_chunk(predictor, cam0, anchor_mask_dir, out_dir, jpeg_root, c_start, c_end,
                  med_area, anchor_manifest=None):
    """处理一块 [c_start, c_end]: 选锚→双向传播→写 mask。返回 [(frame, area)] / None(失败) / [](跳过)。"""
    chunk_frames = list(range(c_start, c_end + 1))
    if chunk_done(out_dir, chunk_frames):
        print(f"  [skip] 块 {c_start}-{c_end} 已完成({len(chunk_frames)} 帧)", flush=True)
        return []
    anchor, amask = select_anchor(
        anchor_mask_dir, chunk_frames, med_area, anchor_manifest=anchor_manifest)
    if anchor is None or amask is None or not amask.any():
        print(f"  [warn] 块 {c_start}-{c_end} 无可用锚帧, 跳过", flush=True)
        return None
    jpeg_dir = os.path.join(jpeg_root, f"chunk_{c_start:05d}")
    if not prepare_jpeg_dir(cam0, chunk_frames, jpeg_dir):
        print(f"  [warn] 块 {c_start}-{c_end} 缺原图, 跳过", flush=True)
        return None
    anchor_local = chunk_frames.index(anchor)
    print(f"  块 {c_start}-{c_end} ({len(chunk_frames)} 帧) 锚 f{anchor}(local {anchor_local}) "
          f"area={int(amask.sum())}", flush=True)

    state = predictor.init_state(video_path=jpeg_dir, offload_video_to_cpu=True, async_loading_frames=True)
    predictor.add_new_mask(state, frame_idx=anchor_local, obj_id=1, mask=amask)

    areas = []
    written = set()
    # 前向 [anchor, c_end]
    for fi, _, mt in predictor.propagate_in_video(
            state, start_frame_idx=anchor_local,
            max_frame_num_to_track=c_end - anchor, reverse=False):
        m = (mt[0].cpu().numpy() > 0).squeeze().astype(np.uint8)
        gf = chunk_frames[fi]
        if gf not in written:
            write_mask(out_dir, gf, m)
            written.add(gf)
            areas.append((gf, int(m.sum())))
    # 反向 [anchor, c_start](anchor_local>0 才有效; SAM2 reverse 在 start_frame_idx=0 时自动跳过)
    if anchor_local > 0:
        for fi, _, mt in predictor.propagate_in_video(
                state, start_frame_idx=anchor_local,
                max_frame_num_to_track=anchor - c_start, reverse=True):
            m = (mt[0].cpu().numpy() > 0).squeeze().astype(np.uint8)
            gf = chunk_frames[fi]
            if gf not in written:
                write_mask(out_dir, gf, m)
                written.add(gf)
                areas.append((gf, int(m.sum())))
    shutil.rmtree(jpeg_dir, ignore_errors=True)
    missing = [f for f in chunk_frames if f not in written]
    if missing:
        print(f"    [warn] 块 {c_start}-{c_end} 缺 {len(missing)} 帧: {missing[:5]}...", flush=True)
    return areas


def main():
    pa = argparse.ArgumentParser(description="SAM2 视频分割全序列(分块双向, 多 GPU 分片, 断点续跑)")
    pa.add_argument("--seq", required=True)
    pa.add_argument("--camera", default="cam0", help="输入视角目录名，默认cam0")
    pa.add_argument("--anchor-mask-dir", default=None,
                    help="候选锚帧mask目录；默认优先masks_candidate，旧序列回退masks_repaired")
    pa.add_argument("--anchor-manifest", default=None,
                    help="prepare_sam2_anchors.py输出；默认derived/<seq>/anchor_manifest.csv")
    pa.add_argument("--out", default=None, help="输出目录(默认 sam2/masks/<seq>_full)")
    pa.add_argument("--chunk-size", type=int, default=200, help="块大小(默认 200; 锚居中, 前/反各 100)")
    pa.add_argument("--shards", type=int, default=1, help="分片总数(多 GPU 并行)")
    pa.add_argument("--shard", type=int, default=0, help="本进程处理第几片(chunk_idx %% shards == shard)")
    pa.add_argument("--start-frame", type=int, default=0)
    pa.add_argument("--end-frame", type=int, default=None, help="默认到最后一帧")
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    seq = args.seq.rstrip("/")
    seq_name = os.path.basename(seq)
    raw_seq_dir = (os.path.abspath(seq) if os.path.isdir(os.path.join(seq, args.camera))
                   else os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq_name))
    cam0 = os.path.join(raw_seq_dir, args.camera)
    derived_dir = os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq_name)
    candidate_dir = os.path.join(derived_dir, "masks_candidate")
    legacy_dir = os.path.join(derived_dir, "masks_repaired")
    anchor_dir = args.anchor_mask_dir or (
        candidate_dir if os.path.isdir(candidate_dir) else legacy_dir)
    manifest_path = args.anchor_manifest or os.path.join(derived_dir, "anchor_manifest.csv")
    anchor_manifest = load_anchor_manifest(manifest_path)
    out_dir = args.out or os.path.join(HERE, "masks", f"{seq_name}_full")
    jpeg_root = os.path.join(HERE, "_jpeg_tmp", f"{seq_name}_shard{args.shard}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(jpeg_root, exist_ok=True)

    all_fs = sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(os.path.join(cam0, "*.png")))
    first_image = (cv2.imread(os.path.join(cam0, f"{all_fs[0]:05d}.png"))
                   if all_fs else None)
    input_size_wh = ([int(first_image.shape[1]), int(first_image.shape[0])]
                     if first_image is not None else [])
    f_lo = max(args.start_frame, all_fs[0]) if all_fs else args.start_frame
    f_hi = min(args.end_frame if args.end_frame is not None else all_fs[-1], all_fs[-1]) if all_fs else (args.end_frame or 0)
    chunks = [(s, min(s + args.chunk_size - 1, f_hi)) for s in range(f_lo, f_hi + 1, args.chunk_size)]
    my_chunks = [(i, c) for i, c in enumerate(chunks) if i % args.shards == args.shard]
    print(f">>> {seq_name}: 帧 [{f_lo}..{f_hi}] 共 {f_hi - f_lo + 1} 帧, {len(chunks)} 块; "
          f"分片 {args.shard}/{args.shards} 处理 {len(my_chunks)} 块", flush=True)
    print(f"    锚 mask: {anchor_dir}\n    锚清单: {manifest_path if anchor_manifest else '无(旧启发式回退)'}"
          f"\n    输出:   {out_dir}\n    device: {args.device}", flush=True)

    med = global_median_area(anchor_dir, f_hi + 1)
    print(f"    全局 area 中位(抽样)={med:.0f} → clean 判据 [0.7,1.3]×med = "
          f"[{0.7*med:.0f},{1.3*med:.0f}]", flush=True)

    import torch  # noqa: F401
    predictor = build_predictor(args.device)

    area_log = os.path.join(out_dir, "area_curve.txt")
    fail_log = os.path.join(out_dir, f"failures_shard{args.shard}.txt")
    # 每次 shard 调用重写自己的失败清单；成功断点续算会清除该 shard 的历史失败状态。
    with open(area_log, "a") as fa, open(fail_log, "w") as ff:
        fa.write(f"# shard {args.shard}/{args.shards} start; med_area={med:.0f}\n")
        for ci, (c_start, c_end) in my_chunks:
            try:
                areas = process_chunk(predictor, cam0, anchor_dir, out_dir, jpeg_root,
                                      c_start, c_end, med, anchor_manifest=anchor_manifest)
                if areas is None:
                    ff.write(f"块 {c_start}-{c_end}: 无锚帧/缺图, 跳过\n")
                elif areas:
                    for f, a in sorted(areas):
                        fa.write(f"{f} {a}\n")
                    fa.flush()
                    avs = [a for _, a in areas]
                    print(f"    ✓ 块 {c_start}-{c_end}: {len(areas)} 帧, area min={min(avs)} "
                          f"mean={np.mean(avs):.0f} max={max(avs)}", flush=True)
            except Exception as e:
                ff.write(f"块 {c_start}-{c_end}: {e}\n{traceback.format_exc()}\n")
                ff.flush()
                print(f"    [ERR] 块 {c_start}-{c_end}: {e}", flush=True)
    shutil.rmtree(jpeg_root, ignore_errors=True)
    with open(os.path.join(out_dir, f"run_meta_shard{args.shard}.json"), "w") as handle:
        json.dump({
            "schema_version": 1,
            "sequence": seq_name,
            "input_sequence_path": os.path.abspath(raw_seq_dir),
            "input_image_size_wh": input_size_wh,
            "input_frame_count": len(all_fs),
            "camera": args.camera,
            "anchor_mask_dir": os.path.abspath(anchor_dir),
            "anchor_manifest": os.path.abspath(manifest_path) if anchor_manifest else None,
            "chunk_size": args.chunk_size,
            "frame_range": [f_lo, f_hi],
            "shard": args.shard,
            "shards": args.shards,
            "device": args.device,
        }, handle, indent=2, ensure_ascii=False)
    save_sam2_qc(cam0, anchor_dir, out_dir, args.shard)
    print(f">>> 分片 {args.shard} 完成 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
