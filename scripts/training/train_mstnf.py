"""train_mstnf.py — Multi-Scale Temporal Neural Field 训练脚本。

用法:
  python scripts/training/train_mstnf.py
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model_mstnf import MSTNFModel
from src.utils.camera import get_rays
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.experiment import create_experiment, save_config, save_gif
from src.data.dataset import SoftSequenceDataset
from src.config.params import load_config, get_camera_params

# GPU 配置
CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training MSTNF on device: {device}")


def soft_model_forward(rays_o, rays_d, near, far, model, action_window, n_samples=64,
                       target_img=None, n_rays_sample=1024, fg_ratio=0.5):
    """前向传播：EMA编码 → 射线采样 → 空间查询 → 体渲染。

    Args:
        rays_o, rays_d: 全图射线 (N_rays, 3)。
        near, far: 近远平面。
        model: MSTNFModel。
        action_window: (Batch, Window, Action_Dim)。
        n_samples: 每条射线采样点数。
        target_img: (Batch, N_rays) GT 图像。提供时启用前景过采样。
        n_rays_sample: 训练时每步采样的射线数。
        fg_ratio: 前景射线占比（其余随机采样）。

    Returns:
        rgb_map: (Batch, sampled_rays) 或 (Batch, N_rays)。
        ray_indices: 采样的射线索引（用于对齐 GT）。
    """
    B, K, D = action_window.shape
    physics_state = model.encode_temporal(action_window)  # (B, Hidden)
    current_action = action_window[:, -1, :]  # (B, D) skip connection

    # ── 射线采样策略 ──
    if target_img is not None and n_rays_sample < rays_o.shape[0]:
        N_total = rays_o.shape[0]
        # 前景像素全图共享同一个掩码逻辑，取 batch 中第一个的前景像素
        fg_mask = target_img[0] > 0.1
        fg_idx = torch.where(fg_mask)[0]
        n_fg = int(n_rays_sample * fg_ratio)
        n_bg = n_rays_sample - n_fg

        if len(fg_idx) > 0 and n_fg > 0:
            chosen_fg = fg_idx[torch.randint(len(fg_idx), (n_fg,), device=rays_o.device)]
            chosen_bg = torch.randint(N_total, (n_bg,), device=rays_o.device)
            sel = torch.cat([chosen_fg, chosen_bg])
        else:
            sel = torch.randint(N_total, (n_rays_sample,), device=rays_o.device)
        rays_o_sel = rays_o[sel]
        rays_d_sel = rays_d[sel]
    else:
        sel = None
        rays_o_sel = rays_o
        rays_d_sel = rays_d

    pts, z_vals = sample_stratified(rays_o_sel, rays_d_sel, near, far, n_samples)

    # 扩展 state 和 action 匹配 rays
    N_rays = pts.shape[0]
    state_expanded = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(-1, physics_state.shape[-1])
    action_expanded = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(-1, D)
    pts_expanded = pts.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, n_samples, 3)

    # 分块查询避免显存溢出
    chunk_size = 4096
    raw_parts = []
    for i in range(0, pts_expanded.shape[0], chunk_size):
        p_chunk = pts_expanded[i:i + chunk_size]
        s_chunk = state_expanded[i:i + chunk_size]
        a_chunk = action_expanded[i:i + chunk_size]
        raw_parts.append(model.decode_spatial(p_chunk, s_chunk, a_chunk))

    raw = torch.cat(raw_parts, dim=0)
    raw = raw.reshape(B, N_rays, n_samples, 2)

    # 体渲染
    rgb_map, _ = OM_rendering(raw.reshape(-1, n_samples, 2))
    rgb_map = rgb_map.reshape(B, -1)
    return rgb_map, sel


def train():
    """MSTNF 训练主循环。"""
    # ── 加载配置 ──
    cam_cfg = get_camera_params()
    train_cfg = load_config("training")
    model_cfg = train_cfg["model"]
    temp_cfg = train_cfg["temporal"]
    loss_cfg = train_cfg["loss_weights"]
    opt_cfg = train_cfg["optimization"]
    log_cfg = train_cfg["logging"]

    DATA_DIR = "data/sequence_data"
    BASE_LOG_DIR = os.path.join("train_log", "train_mstnf")

    # ── 数据 ──
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No data in {DATA_DIR}")

    split = max(1, int(0.8 * len(all_files)))
    train_files, val_files = all_files[:split], all_files[split:]

    train_ds = SoftSequenceDataset(
        DATA_DIR, seq_len=temp_cfg["window_size"], file_list=train_files, return_pairs=True,
    )
    val_ds = SoftSequenceDataset(
        DATA_DIR, seq_len=temp_cfg["window_size"], file_list=val_files,
        norm_factor=train_ds.norm_factor,
    )

    train_loader = DataLoader(train_ds, batch_size=opt_cfg["batch_size"], shuffle=True, num_workers=4)

    # ── 相机射线 ──
    H, W = train_ds.H, train_ds.W
    focal = torch.tensor(train_ds.focal).float().to(device)
    action_dim = train_ds.action_dim

    rays_o, rays_d = get_rays(H, W, focal, cam_cfg["eye"], cam_cfg["center"], cam_cfg["up"])
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    # ── 模型 ──
    model = MSTNFModel(
        action_dim=action_dim,
        window_size=temp_cfg["window_size"],
        n_scales=temp_cfg["n_scales"],
        hidden_dim=temp_cfg["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt_cfg["scheduler_patience"])

    near, far = cam_cfg["near"], cam_cfg["far"]
    n_samples = cam_cfg["n_samples"]
    w_recon = loss_cfg["recon_current"]
    w_recon_next = loss_cfg["recon_next"]
    w_smooth = loss_cfg["smoothness"]
    n_iters = opt_cfg["n_iters"]
    save_rate = log_cfg["save_rate"]

    # 将总迭代数转换为 epoch 数
    steps_per_epoch = max(1, len(train_ds) // opt_cfg["batch_size"])
    n_epochs = max(1, n_iters // steps_per_epoch)

    # ── 创建实验目录并保存配置 ──
    config_dict = {
        "model": "MSTNFModel",
        "action_dim": action_dim,
        "window_size": temp_cfg["window_size"],
        "n_scales": temp_cfg["n_scales"],
        "hidden_dim": temp_cfg["hidden_dim"],
        "d_filter": model_cfg["d_filter"],
        "n_freqs": model_cfg["n_freqs"],
        "density_bias": -1.0,
        "training": {
            "lr": opt_cfg["lr"],
            "batch_size": opt_cfg["batch_size"],
            "n_epochs": n_epochs,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_patience": opt_cfg["scheduler_patience"],
        },
        "loss_weights": {
            "recon_current": w_recon,
            "recon_next": w_recon_next,
            "smoothness": w_smooth,
        },
        "camera": {
            "eye": list(cam_cfg["eye"]),
            "center": list(cam_cfg["center"]),
            "up": list(cam_cfg["up"]),
            "near": near,
            "far": far,
            "n_samples": n_samples,
        },
        "data": {
            "norm_factor": train_ds.norm_factor,
            "train_files": len(train_files),
            "val_files": len(val_files),
            "image_size": [H, W],
        },
    }
    LOG_DIR = create_experiment(BASE_LOG_DIR, config_dict)

    print(f">>> MSTNF Training: {n_epochs} epochs ({n_iters} iters), {len(train_ds)} samples")
    print(f"    Window={temp_cfg['window_size']}, Scales={temp_cfg['n_scales']}, Hidden={temp_cfg['hidden_dim']}")
    print(f"    Camera: Eye={cam_cfg['eye']}, Image={H}x{W}, Focal={train_ds.focal:.1f}")

    # ── 获取验证集原始动作曲线（用于 GIF） ──
    val_actions = val_ds.get_raw_actions(seq_id=0)

    # ── 验证：渲染多个时间步并保存 GIF ──
    def evaluate_and_save_gif(epoch_idx):
        model.eval()
        pred_frames = []
        gt_frames = []
        val_loss_total = 0
        sample_step = max(1, len(val_ds) // 20)  # 取约 20 帧

        with torch.no_grad():
            for vi in range(0, len(val_ds), sample_step):
                val_seq, val_img = val_ds[vi]
                val_seq = val_seq.unsqueeze(0).to(device)
                val_img_flat = val_img.to(device)

                val_pred, _ = soft_model_forward(
                    rays_o, rays_d, near, far, model, val_seq, n_samples,
                )
                val_loss_total += torch.nn.functional.mse_loss(val_pred[0], val_img_flat).item()

                pred_frames.append(val_pred[0].reshape(H, W).cpu().numpy())
                gt_frames.append(val_img.reshape(H, W).numpy())

        avg_val_loss = val_loss_total / max(len(pred_frames), 1)
        save_gif(LOG_DIR, f"epoch_{epoch_idx:02d}.gif",
                 pred_frames, gt_frames, epoch_idx,
                 action_curves=val_actions, skip=1, fps=10)
        return avg_val_loss

    # ── 训练循环 ──
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")

        for seq_t, seq_t1, img_t, img_t1 in pbar:
            seq_t = seq_t.to(device)
            seq_t1 = seq_t1.to(device)
            img_t = img_t.to(device)
            img_t1 = img_t1.to(device)

            # Loss 1: 重建 (当前帧, 前景过采样)
            pred_t, idx = soft_model_forward(rays_o, rays_d, near, far, model, seq_t, n_samples,
                                              target_img=img_t, n_rays_sample=1024, fg_ratio=0.5)
            gt_sampled = img_t[:, idx] if idx is not None else img_t
            loss_recon = torch.nn.functional.mse_loss(pred_t, gt_sampled)

            # Loss 2: 重建 (下一帧)
            pred_t1, _ = soft_model_forward(rays_o, rays_d, near, far, model, seq_t1, n_samples,
                                             target_img=img_t1, n_rays_sample=1024, fg_ratio=0.5)
            gt_sampled1 = img_t1[:, idx] if idx is not None else img_t1
            loss_recon_next = torch.nn.functional.mse_loss(pred_t1, gt_sampled1)

            # Loss 3: 时序平滑
            loss_smooth = model.compute_smoothness(seq_t, seq_t1)

            loss = w_recon * loss_recon + w_recon_next * loss_recon_next + w_smooth * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})
            global_step += 1

            if global_step % save_rate == 0:
                torch.save(model.state_dict(),
                           os.path.join(LOG_DIR, "model", f"model_{global_step:05d}.pt"))

        # 每 epoch 结束验证
        val_loss = evaluate_and_save_gif(epoch)
        scheduler.step(val_loss)

        decays = model.get_learned_decays()
        decay_str = ", ".join([f"{d:.3f}" for d in decays])
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {val_loss:.5f} | Decays: [{decay_str}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

    # 保存最终参数
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])
    np.savetxt(os.path.join(LOG_DIR, "learned_decays.txt"), model.get_learned_decays())

    print("Training Finished.")
    print(f"Learned decay rates: {model.get_learned_decays()}")


if __name__ == "__main__":
    train()
