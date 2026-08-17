"""sam2/segment_video.py — SAM2 视频分割(持久化版, 自包含)。

为什么用视频模式: 实物 mask 的"半mask/缺块"在单帧图里看不见(半透明硅胶), image 模式补不回;
视频模式从邻帧传播 mask, 能补回(实测 f4902 r132-156 宽 19→32, 比启发式更干净)。比 /tmp 临时
脚本持久: SAM2_HOME/checkpoint/config 全指向本目录 sam2/。

前置: sam2 包已 editable 装自 sam2/sam2_src(pip install -e sam2/sam2_src)。本脚本顶部设
SAM2_HOME 指向 sam2/sam2_src(否则 import sam2 报 NoneType)。

用法(对一段帧做视频分割, 用一个干净锚帧的 mask 作 prompt 传播):
  CUDA_VISIBLE_DEVICES=2 python sam2/segment_video.py \\
      --seq seq_20260627_163921 --start 4880 --end 4930 \\
      --anchor 4904 --anchor-mask-dir real_capture/data/derived/seq_20260627_163921/masks_repaired \\
      --out sam2/masks/seq_20260627_163921_win4880-4930

输出: <out>/<NNNNN>.png (0/255 mask, 与 cam0 同名对齐) + area 曲线 + QC montage。

扩展到全 10214 帧: 分段(每段给一个干净锚帧的 --anchor + --anchor-mask-dir), 段间重叠几帧。
"""
import argparse
import os
import sys
import shutil

# ---- SAM2_HOME 必须在 import sam2 前设(指向持久 sam2_src) ----
HERE = os.path.dirname(os.path.abspath(__file__))
SAM2_SRC = os.path.join(HERE, "sam2_src")
os.environ.setdefault("SAM2_HOME", SAM2_SRC)
sys.path.insert(0, SAM2_SRC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(HERE)
CKPT = os.path.join(HERE, "checkpoints", "sam2.1_hiera_tiny.pt")
CONFIG_DIR = os.path.join(SAM2_SRC, "sam2", "configs")        # hydra config_dir
CONFIG_FILE = "sam2.1/sam2.1_hiera_t.yaml"                     # 相对 configs 根


def build_predictor(device):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor
    GlobalHydra.instance().clear()                             # 重复 init 需先 clear
    initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1")
    return build_sam2_video_predictor(config_file=CONFIG_FILE, ckpt_path=CKPT, device=device)


def prepare_jpeg_dir(cam0, start, end, jpeg_dir):
    """SAM2 load_video_frames 只吃 .jpg; 拷目标帧为连续名 JPEG。返回帧号列表。"""
    os.makedirs(jpeg_dir, exist_ok=True)
    frames = list(range(start, end + 1))
    for i, f in enumerate(frames):
        img = cv2.imread(os.path.join(cam0, f"{f:05d}.png"))
        if img is None:
            raise FileNotFoundError(f"无原图 {cam0}/{f:05d}.png")
        cv2.imwrite(os.path.join(jpeg_dir, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return frames


def main():
    pa = argparse.ArgumentParser(description="SAM2 视频分割(持久版)")
    pa.add_argument("--seq", required=True)
    pa.add_argument("--start", type=int, required=True)
    pa.add_argument("--end", type=int, required=True)
    pa.add_argument("--anchor", type=int, required=True, help="锚帧(干净, 用其 mask 作 prompt)")
    pa.add_argument("--anchor-mask-dir", required=True, help="锚帧 mask 目录(如 masks_repaired)")
    pa.add_argument("--out", required=True, help="输出 mask 目录")
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    cam0 = os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", args.seq, "cam0")
    tmp_jpeg = os.path.join(HERE, "_jpeg_tmp", f"{args.seq}_{args.start}-{args.end}")
    frames = prepare_jpeg_dir(cam0, args.start, args.end, tmp_jpeg)
    anchor_local = frames.index(args.anchor)
    print(f">>> {len(frames)} 帧 [{args.start}..{args.end}], 锚帧 f{args.anchor}(local idx {anchor_local})")

    import torch  # noqa: F401
    predictor = build_predictor(args.device)
    state = predictor.init_state(video_path=tmp_jpeg, offload_video_to_cpu=True, async_loading_frames=True)

    amask_path = os.path.join(args.anchor_mask_dir, f"{args.anchor:05d}.png")
    amask = (cv2.imread(amask_path, cv2.IMREAD_GRAYSCALE) > 127)
    if not amask.any():
        sys.exit(f"锚帧 mask 空: {amask_path}")
    predictor.add_new_mask(state, frame_idx=anchor_local, obj_id=1, mask=amask)
    print(f"  锚帧 mask area={int(amask.sum())} (from {args.anchor_mask_dir})")

    os.makedirs(args.out, exist_ok=True)
    areas = []
    masks_by_frame = {}
    for fi, obj_ids, mask_tensor in predictor.propagate_in_video(state):
        m = (mask_tensor[0].cpu().numpy() > 0).squeeze().astype(np.uint8)  # logits→bool→(H,W)
        orig_frame = frames[fi]
        cv2.imwrite(os.path.join(args.out, f"{orig_frame:05d}.png"), m * 255)
        areas.append((orig_frame, int(m.sum())))
        masks_by_frame[orig_frame] = m
    avs = [a for _, a in areas]
    print(f"  传播完成: {len(areas)} 帧, area min={min(avs)} mean={np.mean(avs):.0f} "
          f"max={max(avs)} std={np.std(avs):.0f}")

    sample = frames[::max(1, len(frames) // 6)][:6]
    cells = []
    for f in sample:
        img = cv2.imread(os.path.join(cam0, f"{f:05d}.png"))
        m = masks_by_frame.get(f)
        if m is None or img is None:
            continue
        ov = img.copy(); ov[m > 0] = (0, 0, 255)
        cv2.addWeighted(ov, 0.4, img, 0.6, 0, dst=img)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        cv2.putText(img, f"f{f} area={int(m.sum())}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(img)
    if cells:
        cv2.imwrite(os.path.join(args.out, "qc_sam2video.png"), np.hstack(cells))
    with open(os.path.join(args.out, "area_curve.txt"), "w") as fp:
        for f, a in areas:
            fp.write(f"{f} {a}\n")
    shutil.rmtree(tmp_jpeg, ignore_errors=True)
    print(f"→ {args.out}  ({len(areas)} mask + qc_sam2video.png + area_curve.txt)")


if __name__ == "__main__":
    main()
