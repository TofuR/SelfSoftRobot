"""sam2/segment_video_full.py — SAM2 视频分割全序列(10214 帧), 分块双向传播。

为什么: 实物 mask 的"半mask/缺块"在单帧图里看不见(半透明硅胶), image 模式补不回; SAM2 视频
从邻帧传播 mask 能补回。segment_video.py 只做单窗口前向; 本脚本扩展到**全序列**:
  - 分块(默认每块 200 帧), 块内选一个**干净锚帧**(从 masks_repaired 选: 顶部行≤20 且 area 在
    [0.7,1.3]×中位 的帧, 离块中心最近; 无干净帧则退到 area 最接近中位的帧)。
  - 块内**双向传播**: 官方 propagate_in_video(reverse=False) 前向 max=100 + (reverse=True) 反向
    max=100, 一个锚帧覆盖整块(无需重叠拼接)。
  - 块间隔离(每块独立 init_state): 一块的漂移/失败不污染其他块; 失败块记 failures.txt 继续。
  - 多 GPU 分片: --shards N --shard k 取 chunk_idx % N == k 的块(各分片写同一 out 目录,
    帧不重叠)。
  - 断点续跑: 块内所有输出帧已存在则跳过。

输出**单独保存, 不覆盖** masks_repaired: sam2/masks/<seq>_full/<NNNNN>.png + area_curve.txt
+ qc_sam2full.png + failures.txt。

用法(单卡):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921

两卡并行(快一倍):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 0 &
  CUDA_VISIBLE_DEVICES=0 python sam2/segment_video_full.py --seq seq_20260627_163921 --shards 2 --shard 1 &

小样冒烟(前 300 帧, 验证双向传播):
  CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921 --end-frame 299
"""
import argparse
import glob
import os
import shutil
import sys
import traceback

# ---- SAM2_HOME 必须在 import sam2 前设(指向持久 sam2_src) ----
HERE = os.path.dirname(os.path.abspath(__file__))
SAM2_SRC = os.path.join(HERE, "sam2_src")
os.environ.setdefault("SAM2_HOME", SAM2_SRC)
sys.path.insert(0, SAM2_SRC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

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


def select_anchor(anchor_mask_dir, chunk_frames, med_area):
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


def process_chunk(predictor, cam0, anchor_mask_dir, out_dir, jpeg_root, c_start, c_end, med_area):
    """处理一块 [c_start, c_end]: 选锚→双向传播→写 mask。返回 [(frame, area)] / None(失败) / [](跳过)。"""
    chunk_frames = list(range(c_start, c_end + 1))
    if chunk_done(out_dir, chunk_frames):
        print(f"  [skip] 块 {c_start}-{c_end} 已完成({len(chunk_frames)} 帧)", flush=True)
        return []
    anchor, amask = select_anchor(anchor_mask_dir, chunk_frames, med_area)
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
    pa.add_argument("--anchor-mask-dir", default=None,
                    help="锚帧 mask 目录(默认 derived/<seq>/masks_repaired)")
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
    cam0 = os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "cam0")
    anchor_dir = args.anchor_mask_dir or os.path.abspath(
        os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq_name, "masks_repaired"))
    out_dir = args.out or os.path.join(HERE, "masks", f"{seq_name}_full")
    jpeg_root = os.path.join(HERE, "_jpeg_tmp", f"{seq_name}_shard{args.shard}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(jpeg_root, exist_ok=True)

    all_fs = sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(os.path.join(cam0, "*.png")))
    f_lo = max(args.start_frame, all_fs[0]) if all_fs else args.start_frame
    f_hi = min(args.end_frame if args.end_frame is not None else all_fs[-1], all_fs[-1]) if all_fs else (args.end_frame or 0)
    chunks = [(s, min(s + args.chunk_size - 1, f_hi)) for s in range(f_lo, f_hi + 1, args.chunk_size)]
    my_chunks = [(i, c) for i, c in enumerate(chunks) if i % args.shards == args.shard]
    print(f">>> {seq_name}: 帧 [{f_lo}..{f_hi}] 共 {f_hi - f_lo + 1} 帧, {len(chunks)} 块; "
          f"分片 {args.shard}/{args.shards} 处理 {len(my_chunks)} 块", flush=True)
    print(f"    锚 mask: {anchor_dir}\n    输出:   {out_dir}\n    device: {args.device}", flush=True)

    med = global_median_area(anchor_dir, f_hi + 1)
    print(f"    全局 area 中位(抽样)={med:.0f} → clean 判据 [0.7,1.3]×med = "
          f"[{0.7*med:.0f},{1.3*med:.0f}]", flush=True)

    import torch  # noqa: F401
    predictor = build_predictor(args.device)

    area_log = os.path.join(out_dir, "area_curve.txt")
    fail_log = os.path.join(out_dir, "failures.txt")
    with open(area_log, "a") as fa, open(fail_log, "a") as ff:
        fa.write(f"# shard {args.shard}/{args.shards} start; med_area={med:.0f}\n")
        for ci, (c_start, c_end) in my_chunks:
            try:
                areas = process_chunk(predictor, cam0, anchor_dir, out_dir, jpeg_root,
                                      c_start, c_end, med)
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
    print(f">>> 分片 {args.shard} 完成 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
