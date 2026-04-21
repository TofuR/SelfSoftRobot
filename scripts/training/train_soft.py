"""train_soft.py — 软体机器人神经场基线训练脚本。

使用 FBV-SM 模型对软体机械臂进行自仿真学习。

工具函数已提取到:
  - src/utils/camera.py    — get_rays
  - src/utils/rendering.py — sample_stratified, robust_mask_rendering
  - src/data/dataset.py    — load_soft_data
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# GPU 配置
CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

from src.models import FBV_SM, PositionalEncoder
from func import prepare_chunks
from src.utils.camera import get_rays
from src.utils.rendering import sample_stratified, robust_mask_rendering
from src.data.dataset import load_soft_data

from elastica_env import CAMERA_EYE, CAMERA_CENTER, CAMERA_UP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")


def soft_model_forward(rays_o, rays_d, near, far, model, action, chunksize, n_samples=192):
    """软体机器人的前向传播：射线采样 → 模型查询 → 渲染。"""
    query_points, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples)

    action_expanded = action.view(1, 1, -1).expand(query_points.shape[0], query_points.shape[1], -1)
    model_input = torch.cat((query_points, action_expanded), dim=-1)

    batches = prepare_chunks(model_input, chunksize=chunksize)
    predictions = []
    for batch in batches:
        batch = batch.to(device)
        predictions.append(model(batch))

    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    rgb_map = robust_mask_rendering(raw, z_vals)
    return {'rgb_map': rgb_map}


def train():
    """软体机器人神经场训练主循环。"""
    # --- 参数配置 ---
    DATA_DIR = "data/sequence_data"
    LOG_DIR = os.path.join("train_log", "train_log_soft", "experiment_3")
    os.makedirs(os.path.join(LOG_DIR, "image"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)

    N_ITERS = 50000
    LR = 5e-4
    DISPLAY_RATE = 500
    SAVE_RATE = 2000
    NEAR, FAR = 0.5, 2.5

    # --- 加载数据 ---
    images_np, actions_np, focal_val = load_soft_data(DATA_DIR)

    action_max = np.max(np.abs(actions_np))
    if action_max > 0:
        actions_np = actions_np / action_max
        print(f"Actions normalized by max value: {action_max}")
    else:
        action_max = 1.0

    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [action_max])

    images = torch.from_numpy(images_np).float().to(device)
    actions = torch.from_numpy(actions_np).float().to(device)
    focal = torch.tensor(focal_val).float().to(device)

    num_samples = len(images)
    H, W = images.shape[1], images.shape[2]
    DOF = actions.shape[1]

    print(f"Data Loaded: {num_samples} frames, {H}x{W}, DOF={DOF}")

    idx = np.arange(num_samples)
    split = int(0.9 * num_samples)
    train_idx = idx[:split]
    test_idx = idx[split:]

    # --- 初始化模型 ---
    d_input = 3 + DOF

    encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)
    model = FBV_SM(encoder=encoder, d_input=d_input, d_filter=128, output_size=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)

    # 预计算射线
    print(f"Generating rays for Camera: Eye={CAMERA_EYE}, Center={CAMERA_CENTER}")
    rays_o, rays_d = get_rays(H, W, focal, CAMERA_EYE, CAMERA_CENTER, CAMERA_UP)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    print(">>> Start Training...")

    for i in trange(N_ITERS):
        model.train()

        target_idx = np.random.choice(train_idx)
        target_img = images[target_idx].reshape(-1)
        target_action = actions[target_idx]

        outputs = soft_model_forward(
            rays_o.reshape(-1, 3), rays_d.reshape(-1, 3),
            NEAR, FAR, model, target_action, chunksize=4096 * 8,
        )

        pred_img = outputs['rgb_map']
        loss = nn.functional.mse_loss(pred_img, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % DISPLAY_RATE == 0:
            model.eval()
            with torch.no_grad():
                val_idx = np.random.choice(test_idx)
                val_action = actions[val_idx]
                val_gt = images[val_idx].cpu().numpy()

                val_out = soft_model_forward(
                    rays_o.reshape(-1, 3), rays_d.reshape(-1, 3),
                    NEAR, FAR, model, val_action, chunksize=4096 * 8,
                )

                val_pred = val_out['rgb_map'].reshape(H, W).cpu().numpy()
                val_loss = np.mean((val_pred - val_gt) ** 2)
                scheduler.step(val_loss)

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title(f"GT (Iter {i})")
                plt.imshow(val_gt, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title(f"Pred (Loss: {val_loss:.5f})")
                plt.imshow(val_pred, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(LOG_DIR, "image", f"step_{i:05d}.png"))
                plt.close()

        if i % SAVE_RATE == 0:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", f"model_{i:05d}.pt"))
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

    print("Training Finished.")


if __name__ == "__main__":
    train()
