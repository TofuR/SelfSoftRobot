"""visualize_real_overlay.py — 实物 transition 前向预测叠在真实机器人照片上。

``robot_planar_mm_v1`` 数据先在模型毫米坐标中计算预测误差，再读取 NPZ 保存的
``SkeletonFrameTransform`` 映射到源相机像素。历史 ``camera_pixel_v1`` 数据直接使用
像素坐标。两种状态合同共用同一叠图入口，误差单位由 NPZ 明确给出。

模式(从 checkpoint 自动识别, 可 --mode 覆盖):
  gt         GTObservedTransitionModel: 观测驱动单步 ŝ_t=F(GT s_{t-1}, z, a_t)（部署=每步观测）。
  open_loop  OpenLoopTransitionModel: 窗口开环——每 K 步用 GT 重种子, 窗口内喂自身预测
             （部署=观测一次预测 K 步）; 漂移随 k 累积, 正是 GT vs open_loop 对比的核心。

输出(out/<ckpt_name>/): 每帧 photo+mask+GT(绿)+pred(青) 叠图 + montage + 误差曲线 + summary。

用法:
  # GT 模型(每帧叠 GT vs 预测)
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/visualize_real_overlay.py \
      --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_clean/train

  # open_loop 模型(窗口开环, 看漂移; 可加 --with-onestep 画出单步上界)
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/visualize_real_overlay.py \
      --checkpoint train_log/open_loop_transition/exp_*/phase_open_loop_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_clean/train --mode open_loop

  # val 集(自动算 frame offset) / 指定帧 / 全量
  ... --data_dir .../val  --frames 100,500,900  --all
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.model_loader import load_model  # noqa: E402
from src.data.action_view import project_actions, resolve_action_contract  # noqa: E402
from real_validation.perception.coordinates import SkeletonFrameTransform  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------- 模型 rollout(归一化空间) -----------------------------
def build_action_window(actions, t, window_size):
    """以 t 结尾的动作窗口, 不足前向 zero-pad（与 dataset._get_action_window / eval 一致）。"""
    D = actions.shape[1]
    start = t - window_size + 1
    if start >= 0:
        return actions[start:t + 1].copy()
    pad = np.zeros((-start, D), dtype=actions.dtype)
    return np.concatenate([pad, actions[0:t + 1]], axis=0)


def run_rollout(model, mode, actions, positions, window_size, norm_factor, device,
                K=40, max_steps=None):
    """跑 rollout 得每帧归一化预测 pred_norm (T,N,3)。模式:
      gt        观测驱动(prev=GT 每步)。
      open_loop 窗口开环(每 K 步 GT 重种子, 窗口内喂自身预测)。
      onestep   纯 teacher-forcing 单步(prev=GT), 作 open_loop 的误差上界参考。
    forward/归一化完全照 eval_rollout.py / eval_gt_transition.py(已验证)。
    """
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)
    actions_norm = actions / norm_factor
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()
    N = positions.shape[2]

    def to_norm(pos_3N):
        s = pos_3N.T.astype(np.float32)
        s = (s - pc_center) / pc_scale
        return torch.from_numpy(s).float().unsqueeze(0).to(device)

    def aw_tensor(t):
        return torch.from_numpy(build_action_window(actions_norm, t, window_size)
                                ).float().unsqueeze(0).to(device)

    pred = np.zeros((T, N, 3), np.float32)

    with torch.no_grad():
        if mode in ("gt", "onestep"):
            z_t = model.init_z_from_action(aw_tensor(0))
            for t in range(T):
                prev = to_norm(positions[max(t - 1, 0)])
                prev2 = to_norm(positions[max(t - 2, 0)])
                out = model.forward(aw_tensor(t), prev, prev2, z_t)
                pred[t] = out['skeleton'].squeeze(0).cpu().numpy()
                z_t = out['latent_z']
        else:  # open_loop windowed
            # t=0 是该序列的真实观测锚点，不是模型预测。保存其归一化坐标，供
            # 相机坐标反变换和叠图使用；统计时由 prediction_valid 单独排除。
            pred[0] = to_norm(positions[0]).squeeze(0).cpu().numpy()
            t = 1
            while t < T:
                z_t = model.init_z_from_action(aw_tensor(t))
                s_roll = to_norm(positions[t - 1])
                s_prev = s_roll
                for k in range(K):
                    tt = t + k
                    if tt >= T:
                        break
                    out = model.forward(aw_tensor(tt), s_roll, s_prev, z_t)
                    pred[tt] = out['skeleton'].squeeze(0).cpu().numpy()
                    z_t = out['latent_z']
                    s_prev = s_roll
                    s_roll = out['skeleton']
                t += K
    return pred, pc_center, pc_scale


def denorm(pred_norm, pc_scale, pc_center):
    """归一化 (T,N,3) → 像素 [col,row,z]。"""
    return pred_norm * pc_scale + pc_center


# ----------------------------- 绘图 -----------------------------
def draw_skel(img, xy, color, r=3, lw=2):
    """xy: (N,2) [col,row]=[x,y]。全 0 跳过。"""
    if xy is None or np.abs(xy).max() == 0:
        return
    pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, lw, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)


def overlay(photo, mask, gt_xy, pred_xy, onestep_xy=None, *,
            mask_alpha=0.22, gt_color=(0, 255, 0), pred_color=(255, 255, 0),
            one_color=(0, 165, 255), tip_gt=(0, 220, 0), tip_pred=(0, 200, 255)):
    """合成单帧: 原图 + mask 半透明红 + GT骨架(绿) + 预测骨架(青) [+onestep(橙)]。"""
    img = photo.copy()
    if mask is not None and mask.any():
        ov = img.copy()
        ov[mask > 0] = (0, 0, 255)
        cv2.addWeighted(ov, mask_alpha, img, 1 - mask_alpha, 0, dst=img)
    if onestep_xy is not None:
        draw_skel(img, onestep_xy, one_color, r=2, lw=2)
    draw_skel(img, gt_xy, gt_color, r=3, lw=2)
    draw_skel(img, pred_xy, pred_color, r=3, lw=2)
    if gt_xy is not None and np.abs(gt_xy).max() > 0:
        cv2.circle(img, (int(gt_xy[-1, 0]), int(gt_xy[-1, 1])), 6, tip_gt, 2, cv2.LINE_AA)
    if pred_xy is not None and np.abs(pred_xy).max() > 0:
        cv2.circle(img, (int(pred_xy[-1, 0]), int(pred_xy[-1, 1])), 6, tip_pred, 2, cv2.LINE_AA)
    return img


# ----------------------------- 路径推断 -----------------------------
def sequence_from_npz(npz_path):
    """从通用 transition NPZ 文件名解析原始采集序列名。"""
    name = os.path.basename(npz_path)
    for suffix in ("_train.npz", "_val.npz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def auto_offset(data_dir, selected_npz=None):
    """npz 索引 t → 原始帧号；多 run 目录按同名 train/val 配对。"""
    base = os.path.dirname(os.path.normpath(data_dir))
    split = os.path.basename(os.path.normpath(data_dir))
    if split == "val":
        if selected_npz:
            seq = sequence_from_npz(selected_npz)
            matching_train = os.path.join(base, "train", f"{seq}_train.npz")
            if os.path.isfile(matching_train):
                return int(np.load(matching_train)['positions'].shape[0])
        tr = sorted(glob.glob(os.path.join(base, "train", "*.npz")))
        if tr:
            return int(np.load(tr[0])['positions'].shape[0])
    return 0


def detect_mode(model):
    """从模型类名识别 gt/open_loop。"""
    name = type(model).__name__
    if "OpenLoop" in name:
        return "open_loop"
    return "gt"


# ----------------------------- main -----------------------------
def main(argv=None):
    pa = argparse.ArgumentParser(description="实物 transition 预测叠在真实照片上")
    pa.add_argument("--checkpoint", required=True, help="best_model.pt")
    pa.add_argument("--data_dir", required=True, help="npz 目录(train/val)")
    pa.add_argument("--cam0", default=None, help="原图目录(默认 real_capture/data/raw/<seq>/cam0)")
    pa.add_argument("--masks", default=None, help="mask 目录(默认 derived/<seq>/masks)")
    pa.add_argument("--seq_idx", type=int, default=0, help="第几个 npz(sorted)")
    pa.add_argument("--mode", choices=["auto", "gt", "open_loop", "onestep"], default="auto")
    pa.add_argument("--window-len", type=int, default=40, help="[open_loop] 窗口 K(每 K 步重种子)")
    pa.add_argument("--max-steps", type=int, default=None, help="最多 rollout 步(默认整序列)")
    pa.add_argument("--frames", default=None, help="指定渲染帧(逗号分隔 npz 索引); 默认均匀采样 16 帧")
    pa.add_argument("--all", action="store_true", help="渲染全部帧(慢, 多图)")
    pa.add_argument("--with-onestep", action="store_true",
                    help="[open_loop] 额外画单步上界(橙)对比漂移")
    pa.add_argument("--frame-offset", type=int, default=None,
                    help="npz 索引→cam0 帧号偏移(默认自动: train=0, val=len(train))")
    pa.add_argument("--out", default=None, help="输出目录(默认 output/real_overlay/<ckpt 父目录>)")
    args = pa.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model'].eval()
    window_size = info['window_size']
    norm_factor = info['norm_factor']
    mode = args.mode if args.mode != "auto" else detect_mode(model)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    selected_npz = files[args.seq_idx]
    raw = np.load(selected_npz)
    raw_actions = raw['actions'].astype(np.float32)
    action_contract = resolve_action_contract(args.data_dir, "auto")
    actions = project_actions(
        raw_actions, action_contract.model_action_channels).astype(np.float32)
    positions = raw['positions'].astype(np.float32)    # (T,3,N) 模型状态坐标
    state_frame = (str(raw['state_coordinate_frame'].item())
                   if 'state_coordinate_frame' in raw else 'camera_pixel_v1')
    state_unit = (str(raw['state_length_unit'].item())
                  if 'state_length_unit' in raw else 'px')
    checkpoint_state = (info.get('saved_config') or {}).get('state_view', {})
    checkpoint_frame = checkpoint_state.get(
        'state_coordinate_frame', 'camera_pixel_v1')
    if checkpoint_frame != state_frame:
        raise ValueError(
            f"checkpoint state_coordinate_frame={checkpoint_frame} 与数据 {state_frame} 不一致")
    frame_transform = (SkeletonFrameTransform.from_dict(json.loads(
        str(raw['skeleton_frame_transform'].item())))
        if 'skeleton_frame_transform' in raw else None)
    T = positions.shape[0]
    N = positions.shape[2]
    if actions.shape[1] != info['action_dim']:
        raise ValueError(
            f"叠图动作视图{actions.shape[1]}D与checkpoint {info['action_dim']}D不一致")
    print(f"\n模型: {type(model).__name__} → mode={mode} | T={T} N={N} "
          f"rawD={raw_actions.shape[1]} modelD={actions.shape[1]} "
          f"channels={action_contract.model_action_channels} "
          f"window={window_size} norm_factor={norm_factor:.4g}")

    pred_norm, pc_center, pc_scale = run_rollout(
        model, mode, actions, positions, window_size, norm_factor, device,
        K=args.window_len, max_steps=args.max_steps)
    T = pred_norm.shape[0]
    positions = positions[:T]
    actions = actions[:T]
    pred_world = denorm(pred_norm, pc_scale, pc_center)          # (T,N,3) 模型状态坐标
    gt_world = positions.transpose(0, 2, 1)
    if frame_transform is not None:
        pred_camera = frame_transform.model_to_camera(pred_world[:, :, :2])
        gt_camera = (raw['positions_camera_px'][:T, :2, :].transpose(0, 2, 1)
                     if 'positions_camera_px' in raw else
                     frame_transform.model_to_camera(gt_world[:, :, :2]))
    else:
        pred_camera = pred_world[:, :, :2]
        gt_camera = gt_world[:, :, :2]
    z_mag = float(np.abs(pred_world[:, :, 2]).mean())

    one_world = None
    if mode == "open_loop" and args.with_onestep:
        one_norm, _, _ = run_rollout(model, "onestep", actions, positions, window_size,
                                     norm_factor, device, K=args.window_len, max_steps=args.max_steps)
        one_world = denorm(one_norm, pc_scale, pc_center)
    one_camera = (None if one_world is None else
                  (frame_transform.model_to_camera(one_world[:, :, :2])
                   if frame_transform is not None else one_world[:, :, :2]))

    tip_err = np.hypot(*(pred_world[:, -1, :2] - gt_world[:, -1, :2]).T)   # (T,)
    node_err = np.sqrt(((pred_world[:, :, :2] - gt_world[:, :, :2]) ** 2).sum(-1)).mean(axis=1)
    prediction_valid = (np.arange(T) > 0 if mode == "open_loop" else
                        np.ones(T, dtype=bool))

    seq = sequence_from_npz(selected_npz)
    cam0 = args.cam0 or os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "cam0")
    masks_dir = args.masks or os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq, "masks")
    offset = (args.frame_offset if args.frame_offset is not None else
              auto_offset(args.data_dir, selected_npz))
    print(f"  seq={seq} cam0={'OK' if os.path.isdir(cam0) else 'MISSING'} "
          f"masks={'OK' if os.path.isdir(masks_dir) else 'MISSING'} frame_offset={offset}")
    print(f"  state_frame={state_frame} unit={state_unit}")
    print(f"  预测 z 量级: mean|z|={z_mag:.3f}{state_unit}")
    valid_tip = tip_err[prediction_valid]
    valid_node = node_err[prediction_valid]
    print(f"  末端(tip)误差: mean={valid_tip.mean():.2f}{state_unit} "
          f"median={np.median(valid_tip):.2f}{state_unit} "
          f"max={valid_tip.max():.1f}{state_unit} | "
          f"全节点均误: mean={valid_node.mean():.2f}{state_unit}")

    ckpt_tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint)))) \
        or os.path.basename(args.checkpoint)
    out_dir = args.out or os.path.join(PROJECT_ROOT, "output", "real_overlay", ckpt_tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        sel = list(range(T))
    elif args.frames:
        sel = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    else:
        sel = list(np.linspace(0, T - 1, 16).astype(int))
    sel = sorted(set(sel))

    cells, recs = [], []
    for t in sel:
        f = t + offset
        ip = os.path.join(cam0, f"{f:05d}.png")
        photo = cv2.imread(ip)
        if photo is None:
            print(f"  [跳过 f{f}] 无原图 {ip}")
            continue
        mp = os.path.join(masks_dir, f"{f:05d}.png")
        mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if os.path.isfile(mp) else None
        if mask is not None and mask.shape != photo.shape[:2]:
            crop = (raw["image_crop_xywh"].astype(int).tolist()
                    if "image_crop_xywh" in raw else None)
            if crop and len(crop) == 4:
                x, y, w_crop, h_crop = crop
                if mask.shape != (h_crop, w_crop):
                    mask = cv2.resize(mask, (w_crop, h_crop),
                                      interpolation=cv2.INTER_NEAREST)
                restored = np.zeros(photo.shape[:2], dtype=np.uint8)
                x1, y1 = min(x + w_crop, restored.shape[1]), min(y + h_crop, restored.shape[0])
                restored[y:y1, x:x1] = mask[:y1 - y, :x1 - x]
                mask = restored
            else:
                print(f"  [忽略 f{f} mask] mask={mask.shape} photo={photo.shape[:2]}且无ROI合同")
                mask = None
        is_prediction = bool(prediction_valid[t])
        img = overlay(
            photo, mask, gt_camera[t],
            pred_camera[t] if is_prediction else None,
            onestep_xy=(one_camera[t] if one_camera is not None and is_prediction
                        else None))
        status = (f"tipErr={tip_err[t]:.1f}{state_unit}" if is_prediction
                  else "observed anchor")
        cv2.putText(img, f"f{f} t{t} {mode} {status}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        legend = ("GT green / pred cyan" if is_prediction else
                  "observed anchor green")
        cv2.putText(img, legend + (" / onestep orange" if one_world is not None and is_prediction else ""),
                    (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imwrite(os.path.join(out_dir, f"frame_{f:05d}.png"), img)
        cells.append((f, img))
        if is_prediction:
            recs.append((t, f, tip_err[t], node_err[t]))

    if cells:
        h, w = cells[0][1].shape[:2]
        cols = 4
        rows = int(np.ceil(len(cells) / cols))
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, (f, im) in enumerate(cells):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
        cv2.imwrite(os.path.join(out_dir, "montage.png"), canvas)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3))
        prediction_index = np.flatnonzero(prediction_valid)
        plt.plot(prediction_index, valid_tip,
                 label=f"tip (node{pred_world.shape[1] - 1}) err [{state_unit}]")
        plt.plot(prediction_index, valid_node,
                 label=f"mean node err [{state_unit}]", alpha=0.7)
        plt.xlabel("frame index (npz)"); plt.ylabel(state_unit)
        plt.title(f"{ckpt_tag}  {mode}  z={z_mag:.3f}")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "error_plot.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"  [跳过 error_plot] {e}")

    with open(os.path.join(out_dir, "summary.txt"), "w") as fp:
        fp.write(f"checkpoint: {args.checkpoint}\ndata_dir: {args.data_dir}\nmode: {mode}\n")
        fp.write(f"T={T} N={N} window={window_size} norm_factor={norm_factor:.4g}\n")
        fp.write(f"state_frame={state_frame} state_unit={state_unit}\n")
        fp.write(f"pred |z| mean = {z_mag:.4f}{state_unit}\n")
        fp.write(f"prediction_rows={int(prediction_valid.sum())}/{T}\n")
        fp.write(f"tip err: mean={valid_tip.mean():.3f} median={np.median(valid_tip):.3f} "
                 f"max={valid_tip.max():.3f} {state_unit}\n")
        fp.write(f"node err mean = {valid_node.mean():.3f} {state_unit}\n\n"
                 f"frame,tip_{state_unit},node_{state_unit}\n")
        for t, f, te, ne in recs:
            fp.write(f"{t},{f},{te:.2f},{ne:.2f}\n")

    print(f"\n完成 → {out_dir}")
    print(f"  {len(cells)} 帧叠图 + montage.png + error_plot.png + summary.txt")
    print("  绿=GT骨架 青=预测骨架" + (" 橙=onestep上界" if one_world is not None else "") +
          " | 圆圈=末端tip")


if __name__ == "__main__":
    main()
