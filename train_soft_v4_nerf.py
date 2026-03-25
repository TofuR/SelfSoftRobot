import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入
from func import OM_rendering
from src.models import model_v4_nerf_pinn as model_module
from src.utils.camera import get_rays
from src.data.dataset import SoftSequenceDataset
from src.utils.visualization import generate_validation_gif
from src.training.rendering import run_batch_rendering_nerf

# --- 全局设置 ---
CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training NeRF-PINN (V4) on device: {device}")

# ==========================================
# 1. 工具与数据 (保持原逻辑，确保3D一致性)
# ==========================================

# ==========================================
# 3. 训练主程序
# ==========================================
def train_v4_nerf():
    """训练 V4 NeRF-PINN 模型。"""
    # --- Config ---
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 30 # 增加序列长度，让 PINN 更好地计算平滑性
    BATCH_SIZE = 4
    LR = 5e-4
    N_EPOCHS = 50
    VIS_INTERVAL = 5
    LAMBDA_PHY = 0.5 # 物理平滑的权重 (关键参数)
    LOG_DIR = os.path.join("train_log", "train_log_v4_nerf1", "experiment_1")
    
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    # 1. Data
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    train_files, val_files = all_files[:-1], [all_files[-1]]
    
    train_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=train_files)
    val_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=val_files, norm_factor=train_ds.norm_factor)
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model (V4 NeRF PINN)
    model = model_module.NeRF_PINN(
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
    print(f">>> Start Training NeRF-PINN...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_recon = 0; total_phy = 0
        
        for input_seq, target_img_flat in tqdm(train_loader, desc=f"Ep {epoch}"):
            input_seq = input_seq.to(device); target_img_flat = target_img_flat.to(device)
            
            optimizer.zero_grad()
            
            # 1. 渲染 (Batch, N_sampled_rays)
            pred_pixels, loss_phy, ray_idx = run_batch_rendering_nerf(
                model, input_seq, rays_o_full, rays_d_full, NEAR, FAR, N_SAMPLES, is_train=True, device=device
            )
            
            # 2. 获取对应的 GT 像素
            # target_img_flat: (Batch, H*W)
            target_pixels = target_img_flat[:, ray_idx]
            
            # 3. Loss
            loss_recon = criterion(pred_pixels, target_pixels)
            loss = loss_recon + LAMBDA_PHY * loss_phy
            
            loss.backward()
            optimizer.step()
            
            total_recon += loss_recon.item(); total_phy += loss_phy.item()
            
        # --- Validation ---
        if epoch % VIS_INTERVAL == 0:
            print(f"Ep {epoch} | Recon: {total_recon/len(train_loader):.4f} | Phy: {total_phy/len(train_loader):.4f}")
            evaluate_and_save(model, val_loader, epoch, LOG_DIR, train_ds.H, train_ds.W, rays_o_full, rays_d_full, NEAR, FAR, N_SAMPLES, device)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

def evaluate_and_save(model, loader, epoch, log_dir, H, W, rays_o_full, rays_d_full, NEAR, FAR, N_SAMPLES, device):
    """渲染验证序列并保存 GT/预测对比 GIF。"""
    val_data = loader.dataset.data_cache[0]
    save_path = os.path.join(log_dir, "vis", f"ep_{epoch}.gif")
    generate_validation_gif(model, val_data, loader.dataset.seq_len, loader.dataset.action_dim, H, W, 
                            lambda seq, is_train: run_batch_rendering_nerf(model, seq, rays_o_full, rays_d_full, NEAR, FAR, N_SAMPLES, is_train, device), 
                            save_path, device)
    print(f"    Saved validation GIF to {save_path}")

if __name__ == "__main__":
    train_v4_nerf()
    
    