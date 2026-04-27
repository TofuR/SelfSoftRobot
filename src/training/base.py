"""BaseTrainer — 共享基础设施（不管理 GPU，device 由外部传入）。"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.camera import get_rays
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.experiment import create_experiment, save_gif
from src.config.params import load_config, get_camera_params


class BaseTrainer:
    """训练基类：只提供工具方法，不做 GPU 管理。"""

    def __init__(self, device):
        self.device = device
        self.cam_cfg = get_camera_params()
        self.train_cfg = load_config("training")
        self.model_cfg = self.train_cfg["model"]
        self.opt_cfg = self.train_cfg["optimization"]

        self.near = self.cam_cfg["near"]
        self.far = self.cam_cfg["far"]
        self.n_samples = self.cam_cfg["n_samples"]

        self.rays_o = None
        self.rays_d = None
        self.H = None
        self.W = None

    def setup_camera(self, H, W, focal, camera_pose=None):
        self.H, self.W = H, W
        focal_t = torch.tensor(focal).float().to(self.device)
        eye = camera_pose['eye'] if camera_pose else self.cam_cfg["eye"]
        center = camera_pose['center'] if camera_pose else self.cam_cfg["center"]
        up = camera_pose['up'] if camera_pose else self.cam_cfg["up"]
        self.rays_o, self.rays_d = get_rays(H, W, focal_t, eye, center, up)
        self.rays_o = self.rays_o.to(self.device)
        self.rays_d = self.rays_d.to(self.device)

    def sample_fg_rays(self, target_img, n_rays=1024, fg_ratio=0.5):
        N_total = self.rays_o.shape[0]
        fg_mask = target_img[0] > 0.1
        fg_idx = torch.where(fg_mask)[0]
        n_fg = int(n_rays * fg_ratio)
        n_bg = n_rays - n_fg

        if len(fg_idx) > 0 and n_fg > 0:
            chosen_fg = fg_idx[torch.randint(len(fg_idx), (n_fg,), device=self.device)]
            chosen_bg = torch.randint(N_total, (n_bg,), device=self.device)
            sel = torch.cat([chosen_fg, chosen_bg])
        else:
            sel = torch.randint(N_total, (n_rays,), device=self.device)

        return sel, self.rays_o[sel], self.rays_d[sel]

    def render_points(self, forward_fn, pts, chunk_size=4096):
        parts = []
        for i in range(0, pts.shape[0], chunk_size):
            parts.append(forward_fn(pts[i:i + chunk_size]))
        raw = torch.cat(parts, dim=0)
        rgb_map, _ = OM_rendering(raw)
        return rgb_map

    def render_full_image(self, forward_fn, perturb=False):
        with torch.no_grad():
            pts, _ = sample_stratified(
                self.rays_o, self.rays_d, self.near, self.far, self.n_samples, perturb=perturb)
            rgb_map = self.render_points(forward_fn, pts)
            return rgb_map.reshape(self.H, self.W).cpu().numpy()

    def validate_and_gif(self, forward_fn, val_ds, epoch, log_dir,
                         action_curves=None):
        pred_frames = []
        gt_frames = []
        val_loss_total = 0
        skip = max(1, len(val_ds) // 50)

        with torch.no_grad():
            for vi in range(0, len(val_ds), skip):
                val_seq, val_img = val_ds[vi]
                val_seq = val_seq.unsqueeze(0).to(self.device)
                val_img_flat = val_img.to(self.device)

                pred_flat = forward_fn(val_seq)
                val_loss_total += torch.nn.functional.mse_loss(pred_flat, val_img_flat).item()

                pred_frames.append(pred_flat.reshape(self.H, self.W).cpu().numpy())
                gt_frames.append(val_img.reshape(self.H, self.W).numpy())

        avg_val_loss = val_loss_total / max(len(pred_frames), 1)
        sampled_actions = action_curves[::skip] if action_curves is not None else None
        save_gif(log_dir, f"epoch_{epoch:02d}.gif",
                 pred_frames, gt_frames, epoch,
                 action_curves=sampled_actions, skip=1, fps=10)
        return avg_val_loss

    @staticmethod
    def save_canonical_comparison(pred_img, gt_img, save_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(gt_img, cmap='gray')
        ax1.set_title("GT")
        ax1.axis('off')
        ax2.imshow(pred_img, cmap='gray')
        ax2.set_title("Pred")
        ax2.axis('off')
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_experiment(base_dir, config=None):
        return create_experiment(base_dir, config)

    @staticmethod
    def make_phase_dirs(exp_dir, phase_name):
        phase_dir = os.path.join(exp_dir, phase_name)
        os.makedirs(os.path.join(phase_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(phase_dir, "vis"), exist_ok=True)
        return phase_dir
