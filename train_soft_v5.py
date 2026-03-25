import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入
from src.utils.rendering import OM_rendering
from src.models.model_v5_deformable import DeformableSoftRobotModel
from src.utils.camera import get_rays
from src.data.dataset import SoftSequenceDataset
from src.utils.visualization import generate_validation_gif
from src.training.rendering import run_full_rendering_nerf

# --- 全局设置 ---
CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training Deformable NeRF (V5) on device: {device}")

# ==========================================
# 1. 工具与数据
# ==========================================

# ==========================================
# 2. 渲染核心函数 (新增)
# ==========================================
def run_full_rendering(model, input_seq, rays_o_full, rays_d_full, near, far, n_samples):
    """用于验证/可视化的整帧渲染。

    Args:
        model: 可变形神经场模型。
        input_seq: 动作历史，形状 (1, seq_len, action_dim)。
        rays_o_full: 全图射线起点 (H*W, 3)。
        rays_d_full: 全图射线方向 (H*W, 3)。
        near: 近裁剪面。
        far: 远裁剪面。
        n_samples: 每条射线采样数。

    Returns:
        展平后的预测图像，形状 (H*W,)。
    """
    with torch.no_grad():
        # 1. 获取物理状态 (1, Seq, Hidden)
        latent_seq = model.get_physics_state(input_seq)
        current_state = latent_seq[:, -1, :] # (1, Hidden)
        
        # 2. 分块渲染射线 (防止显存溢出)
        chunk_size = 2048
        n_rays = rays_o_full.shape[0]
        all_rgb = []
        
        t_vals = torch.linspace(0., 1., n_samples, device=device)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        
        for i in range(0, n_rays, chunk_size):
            rays_o = rays_o_full[i : i+chunk_size]
            rays_d = rays_d_full[i : i+chunk_size]
            n_chunk = rays_o.shape[0]
            
            # 采样点 (Chunk, Samp, 3)
            # z_vals 需要匹配当前 chunk 大小
            z_vals_chunk = z_vals.expand(n_chunk, n_samples)
            pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_chunk.unsqueeze(2)
            
            # 扩展物理状态 (Chunk, Hidden)
            state_rep = current_state.repeat_interleave(n_chunk, dim=0)
            
            # Query V5 Model
            # pts: (Chunk, Samp, 3) -> 需要 reshape 成 (-1, 3) 吗？
            # query_field 内部处理:
            #   state_exp: (Chunk, 1, Hidden) -> (Chunk, Samp, Hidden)
            #   input shape ok.
            density, _ = model.query_field(pts, state_rep)
            
            # 渲染
            rgb_fake = torch.ones_like(density).repeat(1, 1, 3)
            raw = torch.cat([rgb_fake, density], dim=-1)
            rgb, _ = OM_rendering(raw)
            
            # 只要单通道灰度
            rgb = rgb.mean(dim=-1, keepdim=True)
            all_rgb.append(rgb)
            
        return torch.cat(all_rgb, dim=0).squeeze() # (H*W)

# ==========================================
# 3. 训练主流程
# ==========================================
def train_v5_deformable():
    """训练 V5 可变形神经场模型。

    Returns:
        无返回值；会在 `train_log_v5/*` 保存模型与可视化结果。
    """
    # --- Config ---
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 30 
    BATCH_SIZE = 4
    LR = 5e-4
    N_EPOCHS = 50
    VIS_INTERVAL = 5 # 每5个epoch保存一次可视化结果
    
    # 物理损失权重
    LAMBDA_TIME_SMOOTH = 1.0
    LAMBDA_ELASTIC = 0.01
    
    LOG_DIR = os.path.join("train_log", "train_log_v5", "experiment_1")
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    # 1. Data
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    train_files, val_files = all_files[:-1], [all_files[-1]]
    
    train_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=train_files)
    val_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=val_files, norm_factor=train_ds.norm_factor)
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # [新增] 验证集加载器
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # 2. Model
    model = DeformableSoftRobotModel(
        action_dim=train_ds.action_dim,
        seq_len=SEQ_LEN,
        hidden_dim=128
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # 3. Rays
    CAM_EYE = (1.5, 0.0, 0.5); CAM_CENTER = (0.0, 0.0, 0.25); CAM_UP = (0.0, 0.0, 1.0)
    rays_o_full, rays_d_full = get_rays(
        train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), CAM_EYE, CAM_CENTER, CAM_UP, device=device
    )
    rays_o_full = rays_o_full.reshape(-1, 3); rays_d_full = rays_d_full.reshape(-1, 3)
    NEAR, FAR, N_SAMPLES = 0.5, 2.5, 64

    # --- Training Loop ---
    print(f">>> Start Training V5 (Deformable)...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_recon = 0; total_phy = 0
        
        for input_seq, target_img_flat in tqdm(train_loader, desc=f"Ep {epoch}"):
            input_seq = input_seq.to(device); target_img_flat = target_img_flat.to(device)
            optimizer.zero_grad()
            curr_bs = input_seq.size(0)
            
            # A. 获取时序物理状态
            latent_seq = model.get_physics_state(input_seq)
            current_state = latent_seq[:, -1, :]
            
            # B. 射线采样 (Training: Random Sampling)
            ray_indices = torch.randint(0, rays_o_full.shape[0], (1024,), device=device)
            rays_o = rays_o_full[ray_indices]
            rays_d = rays_d_full[ray_indices]
            
            t_vals = torch.linspace(0., 1., N_SAMPLES, device=device)
            z_vals = NEAR * (1. - t_vals) + FAR * (t_vals)
            z_vals = z_vals.expand(rays_o.shape[0], N_SAMPLES)
            pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) 
            
            pts_in = pts.unsqueeze(0).expand(curr_bs, -1, -1, -1)
            
            # C. 前向计算
            # 扩展 State 用于 Batch 渲染
            density, offset = model.query_field(
                pts_in.reshape(-1, N_SAMPLES, 3), 
                current_state.repeat_interleave(1024, dim=0) 
            )
            
            # 渲染
            rgb_fake = torch.ones_like(density).repeat(1, 1, 3)
            raw = torch.cat([rgb_fake, density], dim=-1)
            
            pred_pixels, _ = OM_rendering(raw) 
            pred_pixels = pred_pixels.view(curr_bs, 1024)
            
            # D. Loss
            target_pixels = target_img_flat[:, ray_indices]
            loss_recon = criterion(pred_pixels, target_pixels)
            
            vel = latent_seq[:, 1:] - latent_seq[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            loss_smooth = torch.mean(acc ** 2)
            loss_elastic = torch.mean(offset ** 2)
            
            loss = loss_recon + LAMBDA_TIME_SMOOTH * loss_smooth + LAMBDA_ELASTIC * loss_elastic
            
            loss.backward()
            optimizer.step()
            
            total_recon += loss_recon.item()
            total_phy += (loss_smooth + loss_elastic).item()

        print(f"Ep {epoch} | Recon: {total_recon/len(train_loader):.4f} | Phy: {total_phy/len(train_loader):.4f}")
        
        # --- [新增] 可视化与保存 ---
        if epoch % VIS_INTERVAL == 0:
            # 保存模型
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))
            
            # 生成 GIF
            print(f"Generating Validation GIF for Epoch {epoch}...")
            model.eval()
            gt_frames = []
            pred_frames = []
            
            # 为了速度，只取验证集的前 50 帧 (或降采样)
            frame_count = 0
            skip = 5 # 降采样
            
            for v_seq, v_img in tqdm(val_loader, desc="Viz"):
                if frame_count % skip == 0:
                    v_seq = v_seq.to(device)
                    # 全图渲染
                    pred_flat = run_full_rendering(model, v_seq, rays_o_full, rays_d_full, NEAR, FAR, N_SAMPLES)
                    
                    pred_img = pred_flat.reshape(train_ds.H, train_ds.W).cpu().numpy()
                    gt_img = v_img.reshape(train_ds.H, train_ds.W).numpy()
                    
                    pred_frames.append(pred_img)
                    gt_frames.append(gt_img)
                
                frame_count += 1
                if len(pred_frames) >= 50: break # 只画前50帧演示
            
            # 绘图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.set_title("Ground Truth")
            ax2.set_title(f"Pred (Ep {epoch})")
            ax1.axis('off'); ax2.axis('off')
            
            im1 = ax1.imshow(gt_frames[0], cmap='gray', vmin=0, vmax=1)
            im2 = ax2.imshow(pred_frames[0], cmap='gray', vmin=0, vmax=1)
            
            def update(frame):
                im1.set_data(gt_frames[frame])
                im2.set_data(pred_frames[frame])
                return im1, im2
            
            ani = animation.FuncAnimation(fig, update, frames=len(pred_frames), blit=True)
            ani.save(os.path.join(LOG_DIR, "vis", f"epoch_{epoch}.gif"), writer='pillow', fps=10)
            plt.close()

if __name__ == "__main__":
    train_v5_deformable()