"""eval_real_quant.py — 实物 transition 模型的**定量**评估(末端 NDI 度量 + 形态 + 漂移)。

为什么需要单独的实物评估脚本:
  transition_metrics.py 的物理指标(*_mm)假设 pc_scale 是**米**(sim)——实物 state 是免标定
  **像素**骨架, 那些 mm 数无意义(毫-像素)。本脚本补两件 transition_metrics 缺的:
    1. 像素空间部署精度(mean_node/endpoint/chamfer/hausdorff **px**, 替代作废的 mm)。
    2. **NDI 末端度量误差(mm)** —— 实物独家: 用采集的 NDI 末端 3D 坐标(独立度量 GT)
       把模型末端预测从像素搬到毫米, 算真·毫米精度(免相机标定, 见下)。

【NDI mm 误差原理(免相机标定)】
  NDI 末端 (x,y,z mm) 与图像骨架 node0 (col,row px) 是同一物理点、逐帧配对。末端在平面内
  做 ~1-DOF 弯曲(x 扫 ~24mm / y 扫 ~9mm, 2D 铺开)。用全部帧 (GT node0 px ↔ NDI x,y mm)
  最小二乘拟合 2D 仿射 A: (col,row,1)→(x,y)。残差RMS = 标定噪声底(骨架化+NDI+非平面)。
  模型末端像素经同一 A→mm, 与 NDI 比 → 末端毫米误差。同时报 GT末端↔NDI 残差(底)以校准。
  严格评价从独立 calibration 序列加载该仿射；同 split 临时拟合只标记为诊断值，避免评价泄漏。
  (不投相机矩阵: 免标定管线无度量3D/无内参；该变换只覆盖末端运动平面。)

指标分四块:
  ① 末端度量(NDI, mm): tip_err_mm 模型 vs NDI; gt_tip_vs_ndi 残差(底)
  ② 像素部署: tip_px, node_mean_px, chamfer_px, hausdorff_px, procrustes_shape_px
  ③ 分段: tip/mid/base 节点段 px; 按 action 分箱的误差(px+mm)
  ④ 漂移(开环): drift_by_k(归一化, 空间不变→对实物有效) + tip_px 随 k

聚合: 每帧(per_frame.csv) + 整体(mean/median/p90/max) + 分段 + 分箱 + drift-by-k。
输出 out/<ckpt_tag>/: summary.txt, per_frame.csv, error_by_action_channel.csv,
  plots (err_vs_action, drift_by_k, per_node_profile, tip_trajectory_mm)。

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
      --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_clean/train

  # open_loop: 额外算窗口漂移; val 集自动算 frame offset
  ... --data_dir .../val --mode open_loop --window-len 40
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.model_loader import load_model  # noqa: E402
from src.data.action_view import project_actions, resolve_action_contract  # noqa: E402
from src.evaluation.shape_metrics import chamfer_distance, hausdorff_distance  # noqa: E402
from src.evaluation.transition_metrics import build_action_window, rollout_one_window  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------- rollout(归一化→像素) -----------------------------
def per_frame_rollout(model, mode, actions, positions, window_size, norm_factor,
                      device, K=40, max_steps=None):
    """逐帧预测 pred_world (T,N,3) 像素 + (open_loop) k_in_window (T,)。

    gt/onestep: prev 恒取 GT(观测驱动 / teacher-forcing 上界)。
    open_loop : 每 K 步 GT 重种子, 窗口内喂自身预测(部署开环); 记录每帧在窗口内的位置 k。
    forward/归一化照 transition_metrics.rollout_one_window(已验证, DRY)。
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

    def aw(t):
        return torch.from_numpy(build_action_window(actions_norm, t, window_size)
                                ).float().unsqueeze(0).to(device)

    pred = np.zeros((T, N, 3), np.float32)
    kin = np.full(T, -1, np.int32)
    with torch.no_grad():
        if mode in ("gt", "onestep"):
            z_t = model.init_z_from_action(aw(0))
            for t in range(T):
                prev = to_norm(positions[max(t - 1, 0)])
                prev2 = to_norm(positions[max(t - 2, 0)])
                out = model.forward(aw(t), prev, prev2, z_t)
                pred[t] = out['skeleton'].squeeze(0).cpu().numpy()
                z_t = out['latent_z']
        else:  # open_loop windowed
            t = 1
            while t < T:
                z_t = model.init_z_from_action(aw(t))
                s_roll = to_norm(positions[t - 1])
                s_prev = s_roll
                for k in range(K):
                    tt = t + k
                    if tt >= T:
                        break
                    out = model.forward(aw(tt), s_roll, s_prev, z_t)
                    pred[tt] = out['skeleton'].squeeze(0).cpu().numpy()
                    kin[tt] = k
                    z_t = out['latent_z']
                    s_prev = s_roll
                    s_roll = out['skeleton']
                t += K
    pred_world = pred * pc_scale + pc_center
    return pred_world, kin


# ----------------------------- NDI 同步 + 仿射标定 -----------------------------
def load_ndi_tip(ndi_csv, frame_times_txt, ndi_index=0, min_quality=None):
    """ndi.csv + frame_times.txt → (T,3) mm, 按帧时间线性插值对齐。"""
    ndi = np.genfromtxt(ndi_csv, delimiter=",", names=True)
    t_ndi = ndi["t_sec"]
    names = set(ndi.dtype.names or ())
    prefix = f"ndi{int(ndi_index)}_"
    xyz_names = ([prefix + axis for axis in ("x", "y", "z")]
                 if prefix + "x" in names else ["x", "y", "z"])
    if not all(name in names for name in xyz_names):
        raise ValueError(f"ndi.csv 无探头 {ndi_index} 的 xyz 列；现有列={sorted(names)}")
    xyz = np.stack([ndi[name] for name in xyz_names], axis=1).astype(np.float64)
    valid = np.isfinite(t_ndi) & np.isfinite(xyz).all(axis=1)
    quality_name = prefix + "quality"
    if min_quality is not None and quality_name in names:
        valid &= np.asarray(ndi[quality_name], dtype=float) >= float(min_quality)
    t_ndi, xyz = np.asarray(t_ndi)[valid], xyz[valid]
    if len(t_ndi) < 2:
        raise ValueError("NDI有效样本不足2个，无法插值")
    ft = np.loadtxt(frame_times_txt)  # (T,)
    order = np.argsort(t_ndi)                                   # 单调化(防重复时间戳)
    t_s = t_ndi[order]
    return np.stack([np.interp(ft, t_s, xyz[order, i]) for i in range(3)], axis=1)


def fit_affine_px_to_mm(gt_tip_px, ndi_tip_mm):
    """最小二乘 2D 仿射 (col,row,1)→(x,y mm)。返回 A(3,2), 残差RMS(mm)。"""
    T = len(gt_tip_px)
    X = np.hstack([gt_tip_px, np.ones((T, 1))])      # (T,3)
    Y = ndi_tip_mm[:, :2]                              # (T,2) 丢 z(近常量)
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (3,2)
    resid = np.sqrt(((X @ A - Y) ** 2).sum(1))         # (T,) mm
    return A, float(np.sqrt((resid ** 2).mean()))      # A, RMS 残差(标定底)


# ----------------------------- 形态指标 -----------------------------
def procrustes_shape_rms(pred_xy, gt_xy):
    """Procrustes 形态误差(px): 去平移/旋转/缩放后的 RMS, 测纯形状(曲率)。"""
    p = pred_xy - pred_xy.mean(0)
    g = gt_xy - gt_xy.mean(0)
    sp, sg = np.linalg.norm(p), np.linalg.norm(g)
    if sp < 1e-9 or sg < 1e-9:
        return 0.0
    p, g = p / sp, g / sg
    M = g.T @ p
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ np.diag([1, d]) @ Vt
    p_rot = p @ R.T
    return float(np.sqrt(((p_rot - g) ** 2).sum(1).mean()) * 0.5 * (sp + sg))


# ----------------------------- main -----------------------------
def main(argv=None):
    pa = argparse.ArgumentParser(description="实物 transition 定量评估(NDI度量+形态+漂移)")
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--data_dir", required=True, help="npz 目录(train/val)")
    pa.add_argument("--ndi", default=None, help="ndi.csv(默认 real_capture/data/raw/<seq>/ndi.csv)")
    pa.add_argument("--frame-times", default=None, help="frame_times.txt(默认 .../raw/<seq>/frame_times.txt)")
    pa.add_argument("--ndi-index", type=int, default=0, help="使用ndiN_*中的第N个探头")
    pa.add_argument("--ndi-min-quality", type=float, default=None,
                    help="可选质量下限；低质量和NaN样本不参与插值")
    pa.add_argument("--calibration-file", default=None,
                    help="独立序列拟合的px→mm仿射npz；未提供则同split自标定并明确警告")
    pa.add_argument("--save-calibration", default=None,
                    help="把本次GT tip↔NDI拟合结果保存为后续独立测试标定")
    pa.add_argument("--seq_idx", type=int, default=0)
    pa.add_argument("--mode", choices=["auto", "gt", "open_loop", "onestep"], default="auto")
    pa.add_argument("--window-len", type=int, default=40)
    pa.add_argument("--max-steps", type=int, default=None)
    pa.add_argument("--no-ndi", action="store_true", help="跳过 NDI(无 ndi.csv 时)")
    pa.add_argument("--frame-offset", type=int, default=None, help="npz索引→原始帧号偏移(默认自动)")
    pa.add_argument("--action-channel", type=int, default=None,
                    help="按 action 分箱用的通道列(默认自动选第一个非零通道;3 腔道可指定)")
    pa.add_argument("--out", default=None)
    args = pa.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model'].eval()
    window_size = info['window_size']
    norm_factor = info['norm_factor']
    name = type(model).__name__
    mode = args.mode if args.mode != "auto" else ("open_loop" if "OpenLoop" in name else "gt")

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    raw = np.load(files[args.seq_idx])
    raw_actions = raw['actions'].astype(np.float32)
    action_contract = resolve_action_contract(args.data_dir, "auto")
    actions = project_actions(raw_actions, action_contract.model_action_channels).astype(np.float32)
    positions = raw['positions'].astype(np.float32)     # (T,3,N) px
    T = positions.shape[0]
    N = positions.shape[2]
    if actions.shape[1] != info['action_dim']:
        raise ValueError(f"评估动作视图{actions.shape[1]}D与checkpoint {info['action_dim']}D不一致")
    print(f"\n{name} → mode={mode} | T={T} N={N} rawD={raw_actions.shape[1]} "
          f"modelD={actions.shape[1]} channels={action_contract.model_action_channels} "
          f"window={window_size}")

    pred_world, kin = per_frame_rollout(
        model, mode, actions, positions, window_size, norm_factor, device,
        K=args.window_len, max_steps=args.max_steps)            # (T_eff,N,3) px
    T = pred_world.shape[0]                                      # 可能被 max_steps 截断
    positions = positions[:T]
    actions = actions[:T]
    gt_world = positions.transpose(0, 2, 1)                      # (T,N,3) px

    # ---- ② 像素部署指标(每帧) ----
    d2 = np.sqrt(((pred_world[:, :, :2] - gt_world[:, :, :2]) ** 2).sum(-1))   # (T,N) px
    tip_px = d2[:, 0]
    node_mean_px = d2.mean(1)
    chamfer_px = np.array([chamfer_distance(pred_world[t, :, :2], gt_world[t, :, :2]) for t in range(T)])
    hausdorff_px = np.array([hausdorff_distance(pred_world[t, :, :2], gt_world[t, :, :2]) for t in range(T)])
    procrustes_px = np.array([procrustes_shape_rms(pred_world[t, :, :2], gt_world[t, :, :2]) for t in range(T)])
    nb, nm = N // 3, 2 * N // 3
    region = {"tip": d2[:, :nb].mean(1),
              "mid": d2[:, nb:nm].mean(1), "base": d2[:, nm:].mean(1)}
    joint_nodes = (tuple(int(v) for v in np.asarray(raw["joint_node_indices"]).tolist())
                   if "joint_node_indices" in raw else ())
    bounds = (0,) + joint_nodes + (N - 1,)
    segment_px = [d2[:, lo:hi + 1].mean(1) for lo, hi in zip(bounds[:-1], bounds[1:])]
    joint_px = (d2[:, list(joint_nodes)].mean(1) if joint_nodes else
                np.full(T, np.nan, dtype=np.float32))

    # ---- ① NDI 末端度量 ----
    seq_dir = os.path.basename(os.path.dirname(os.path.normpath(args.data_dir)))
    seq = seq_dir[:-len("_clean")] if seq_dir.endswith("_clean") else seq_dir
    split = os.path.basename(os.path.normpath(args.data_dir))
    if args.frame_offset is not None:
        offset = args.frame_offset
    elif split == "val":
        tr = sorted(glob.glob(os.path.join(os.path.dirname(os.path.normpath(args.data_dir)), "train", "*.npz")))
        offset = int(np.load(tr[0])['positions'].shape[0]) if tr else 0
    else:
        offset = 0

    tip_mm, gt_tip_mm_floor, ndi_tip = None, None, None
    ndi_z_abs, calibration_mode = None, None
    ndi_csv = args.ndi or os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "ndi.csv")
    ft_csv = args.frame_times or os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "frame_times.txt")
    if args.no_ndi or not os.path.isfile(ndi_csv) or not os.path.isfile(ft_csv):
        print(f"  [跳过 NDI] ndi={'有' if os.path.isfile(ndi_csv) else '无'} ft={'有' if os.path.isfile(ft_csv) else '无'}")
    else:
        ndi_full = load_ndi_tip(ndi_csv, ft_csv, args.ndi_index, args.ndi_min_quality)
        ndi_split = ndi_full[offset:offset + T]
        ndi_tip = ndi_split[:, :2]
        if args.calibration_file:
            with np.load(args.calibration_file, allow_pickle=False) as calibration:
                A = np.asarray(calibration["affine_px_to_mm"], dtype=np.float64)
            floor = float("nan")
            calibration_mode = "independent_file"
        else:
            A, floor = fit_affine_px_to_mm(gt_world[:, 0, :2], ndi_tip)
            calibration_mode = "same_split_diagnostic"
            print("  [警告] 未提供--calibration-file：tip_mm为同split自标定诊断值，不是严格held-out毫米误差")
        if args.save_calibration:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_calibration)), exist_ok=True)
            np.savez_compressed(args.save_calibration, affine_px_to_mm=A,
                                ndi_index=np.array(args.ndi_index),
                                source_data_dir=np.array(os.path.abspath(args.data_dir)))
        gt_tip_mm_floor = floor
        Xm = np.hstack([pred_world[:, 0, :2], np.ones((T, 1))])
        model_tip_mm = Xm @ A                                  # (T,2) mm
        tip_mm = np.sqrt(((model_tip_mm - ndi_tip) ** 2).sum(1))   # (T,) mm
        ndi_z_abs = np.abs(ndi_split[:, 2] - np.nanmedian(ndi_split[:, 2]))
        print(f"  NDI 仿射标定: GT末端↔NDI 残差RMS = {floor:.3f} mm (标定底)")
        print(f"               模型末端↔NDI: mean={tip_mm.mean():.3f} median={np.median(tip_mm):.3f} "
              f"p90={np.percentile(tip_mm,90):.3f} max={tip_mm.max():.2f} mm")

    # ---- ③ 按 action 分箱(1-DOF 用 ch0;多通道自动选第一个非零通道,或 --action-channel) ----
    act_col = args.action_channel
    if act_col is None:
        nonzero = [i for i in range(actions.shape[1]) if np.abs(actions[:, i]).max() > 0]
        act_col = nonzero[0] if nonzero else 0
    act = actions[:, act_col]                                  # (T,) [0,1]
    bins = np.clip((act * 5).astype(int), 0, 4)
    bin_tip_px = [tip_px[bins == b].mean() if (bins == b).any() else np.nan for b in range(5)]
    bin_tip_mm = ([tip_mm[bins == b].mean() if (bins == b).any() else np.nan for b in range(5)]
                  if tip_mm is not None else [np.nan] * 5)
    bin_chamfer_px = [chamfer_px[bins == b].mean() if (bins == b).any() else np.nan for b in range(5)]

    # ---- ④ 漂移 by-k (open_loop 窗口) ----
    drift = None
    if mode == "open_loop":
        pc_center = model.pc_center.view(3).cpu().numpy()
        pc_scale = model.pc_scale.view(3).cpu().numpy()
        K = args.window_len
        roll_k, one_k, tippx_k = [[] for _ in range(K)], [[] for _ in range(K)], [[] for _ in range(K)]
        actions_norm = actions / norm_factor
        t0, n_win = 1, 0
        with torch.no_grad():
            while t0 + K <= T:
                r = rollout_one_window(model, actions_norm, positions, t0, K, window_size,
                                       pc_center, pc_scale, device)
                roll = r['roll'].cpu().numpy(); one = r['one'].cpu().numpy(); gt = r['gt'].cpu().numpy()
                roll_w = roll * pc_scale + pc_center; gt_w = gt * pc_scale + pc_center
                for k in range(K):
                    roll_k[k].append(((roll[k] - gt[k]) ** 2).mean())
                    one_k[k].append(((one[k] - gt[k]) ** 2).mean())
                    tippx_k[k].append(np.hypot(*(roll_w[k, 0, :2] - gt_w[k, 0, :2])))
                n_win += 1; t0 += K
        if n_win:
            rm = np.array([np.mean(x) for x in roll_k]); om = np.array([np.mean(x) for x in one_k])
            drift = {"drift_by_k": (rm / np.maximum(om, 1e-8)).tolist(),
                     "tip_px_by_k": [np.mean(x) for x in tippx_k],
                     "rollout_mse_by_k": rm.tolist(), "onestep_mse_by_k": om.tolist(),
                     "n_windows": n_win}
            print(f"  漂移(open_loop): n_windows={n_win} mean_drift={np.mean(drift['drift_by_k']):.2f}x "
                  f"final_k={drift['drift_by_k'][-1]:.2f}x")

    # ---- 汇总 ----
    def stats(x):
        x = x[~np.isnan(x)]
        return (f"mean={x.mean():.3f} median={np.median(x):.3f} "
                f"p90={np.percentile(x,90):.3f} max={x.max():.3f}")
    ckpt_tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint)))) \
        or os.path.basename(args.checkpoint)
    out_dir = args.out or os.path.join(PROJECT_ROOT, "output", "real_quant", ckpt_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== 整体(px) ===")
    print(f"  tip(末端)    {stats(tip_px)} px")
    print(f"  node_mean    {stats(node_mean_px)} px")
    print(f"  chamfer      {stats(chamfer_px)} px")
    print(f"  hausdorff    {stats(hausdorff_px)} px")
    print(f"  procrustes   {stats(procrustes_px)} px (纯形状)")
    print(f"  分段 base/mid/tip: {region['base'].mean():.2f}/{region['mid'].mean():.2f}/{region['tip'].mean():.2f} px")
    if segment_px:
        print("  物理段 mean: " + "/".join(
            f"segment{i}={values.mean():.2f}px" for i, values in enumerate(segment_px)))
    if joint_nodes:
        print(f"  关节 nodes={joint_nodes}: {np.nanmean(joint_px):.2f}px")
    if tip_mm is not None:
        print(f"\n=== 末端 NDI 度量(mm) ===  calibration={calibration_mode} "
              f"标定底 {gt_tip_mm_floor:.3f} mm")
        print(f"  tip_mm       {stats(tip_mm)} mm")
        print(f"  NDI z离中位面 p50={np.percentile(ndi_z_abs, 50):.3f} "
              f"p95={np.percentile(ndi_z_abs, 95):.3f} max={ndi_z_abs.max():.3f} mm")

    with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["t", "frame", "action", "tip_px", "node_mean_px", "chamfer_px",
                    "hausdorff_px", "procrustes_px", "region_tip_px", "region_mid_px",
                    "region_base_px", "joint_px", "tip_mm", "ndi_z_abs_mm", "k_in_window"]
        header += [f"segment_{i}_px" for i in range(len(segment_px))]
        w.writerow(header)
        for t in range(T):
            row = [t, t + offset, float(act[t]), f"{tip_px[t]:.3f}", f"{node_mean_px[t]:.3f}",
                   f"{chamfer_px[t]:.3f}", f"{hausdorff_px[t]:.3f}", f"{procrustes_px[t]:.3f}",
                   f"{region['tip'][t]:.3f}", f"{region['mid'][t]:.3f}", f"{region['base'][t]:.3f}",
                   f"{joint_px[t]:.3f}" if joint_nodes else "",
                   f"{tip_mm[t]:.3f}" if tip_mm is not None else "",
                   f"{ndi_z_abs[t]:.3f}" if ndi_z_abs is not None else "", int(kin[t])]
            row += [f"{values[t]:.3f}" for values in segment_px]
            w.writerow(row)

    with open(os.path.join(out_dir, "error_by_action_channel.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_action_index", "raw_channel", "bin", "count",
                    "tip_px_mean", "node_mean_px_mean", "tip_mm_mean"])
        for model_index, raw_channel in enumerate(action_contract.model_action_channels):
            channel_bins = np.clip((actions[:, model_index] * 5).astype(int), 0, 4)
            for bin_index in range(5):
                selected = channel_bins == bin_index
                w.writerow([
                    model_index, raw_channel, bin_index, int(selected.sum()),
                    f"{tip_px[selected].mean():.3f}" if selected.any() else "",
                    f"{node_mean_px[selected].mean():.3f}" if selected.any() else "",
                    (f"{tip_mm[selected].mean():.3f}"
                     if tip_mm is not None and selected.any() else ""),
                ])

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\ndata_dir: {args.data_dir}\nmode: {mode}\n")
        f.write(f"T={T} N={N} window={window_size} norm_factor={norm_factor:.4g}\n\n")
        f.write("[pixel space]\n")
        for nm_, x in [("tip_px", tip_px), ("node_mean_px", node_mean_px),
                       ("chamfer_px", chamfer_px), ("hausdorff_px", hausdorff_px),
                       ("procrustes_px", procrustes_px)]:
            xx = x[~np.isnan(x)]
            f.write(f"  {nm_:16s} mean={xx.mean():.3f} median={np.median(xx):.3f} "
                    f"p90={np.percentile(xx,90):.3f} max={xx.max():.3f}\n")
        f.write(f"  region base/mid/tip: {region['base'].mean():.2f}/{region['mid'].mean():.2f}/{region['tip'].mean():.2f}\n")
        for index, values in enumerate(segment_px):
            f.write(f"  segment_{index}_px: {stats(values)}\n")
        if joint_nodes:
            f.write(f"  joint_px nodes={joint_nodes}: {stats(joint_px)}\n")
        if tip_mm is not None:
            f.write(f"\n[tip NDI metric] calibration={calibration_mode} "
                    f"floor(GT vs NDI)={gt_tip_mm_floor:.3f} mm\n")
            f.write(f"  tip_mm mean={tip_mm.mean():.3f} median={np.median(tip_mm):.3f} "
                    f"p90={np.percentile(tip_mm,90):.3f} max={tip_mm.max():.3f} mm\n")
            f.write(f"  ndi_z_abs_mm p50={np.percentile(ndi_z_abs,50):.3f} "
                    f"p95={np.percentile(ndi_z_abs,95):.3f} max={ndi_z_abs.max():.3f}\n")
        f.write("\n[by action bin 0..4]\n bin tip_px tip_mm chamfer_px\n")
        for b in range(5):
            f.write(f"  {b} {bin_tip_px[b]:.2f} {bin_tip_mm[b]:.3f} {bin_chamfer_px[b]:.2f}\n")
        if drift:
            f.write(f"\n[drift open_loop] n_windows={drift['n_windows']} "
                    f"mean={np.mean(drift['drift_by_k']):.2f}x final_k={drift['drift_by_k'][-1]:.2f}x\n")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot(bin_tip_px, "o-", label="tip px"); ax[0].plot(bin_chamfer_px, "s-", label="chamfer px")
        ax[0].set_xlabel("action bin (0..4)"); ax[0].set_ylabel("px"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[0].set_title("error vs bend magnitude")
        if tip_mm is not None:
            ax[1].plot(bin_tip_mm, "o-", color="tab:red", label="tip mm (NDI)")
            ax[1].set_xlabel("action bin"); ax[1].set_ylabel("mm"); ax[1].legend(); ax[1].grid(alpha=.3)
            ax[1].set_title(f"metric tip err (floor {gt_tip_mm_floor:.2f}mm)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "err_vs_action.png"), dpi=120); plt.close()
        plt.figure(figsize=(8, 3))
        plt.plot(d2.mean(0), "o-"); plt.xlabel(f"node (0=tip .. {N - 1}=base)")
        for node in joint_nodes:
            plt.axvline(node, color="tab:orange", ls="--", alpha=.6, label=f"joint node {node}")
        plt.ylabel("mean err [px]"); plt.title("per-node error profile"); plt.grid(alpha=.3)
        if joint_nodes:
            plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "per_node_profile.png"), dpi=120); plt.close()
        if drift:
            plt.figure(figsize=(8, 3))
            plt.plot(drift["drift_by_k"], "o-", label="drift ratio (rollout/onestep)")
            tp = np.array(drift["tip_px_by_k"], dtype=float)
            plt.plot(tp / max(np.nanmax(tp), 1e-9), "s-", label="tip_px (norm)", alpha=.6)
            plt.xlabel("k (steps since last obs)"); plt.legend(); plt.grid(alpha=.3)
            plt.title("open-loop drift by k"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "drift_by_k.png"), dpi=120); plt.close()
        if tip_mm is not None:
            gtm = np.hstack([gt_world[:, 0, :2], np.ones((T, 1))]) @ A
            mdl = np.hstack([pred_world[:, 0, :2], np.ones((T, 1))]) @ A
            plt.figure(figsize=(8, 4))
            plt.plot(ndi_tip[:, 0], ndi_tip[:, 1], ".-", label="NDI (mm)", color="tab:green", ms=3)
            plt.plot(gtm[:, 0], gtm[:, 1], ".-", label="GT tip→mm", color="tab:blue", ms=3, alpha=.7)
            plt.plot(mdl[:, 0], mdl[:, 1], ".-", label="model tip→mm", color="tab:red", ms=3, alpha=.7)
            plt.xlabel("x [mm]"); plt.ylabel("y [mm]"); plt.axis("equal"); plt.legend(); plt.grid(alpha=.3)
            plt.title("tip trajectory (mm)"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "tip_trajectory_mm.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"  [跳过 plots] {e}")

    print(f"\n完成 → {out_dir}  (summary.txt + per_frame.csv + plots)")


if __name__ == "__main__":
    main()
