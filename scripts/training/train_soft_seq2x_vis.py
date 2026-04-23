import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入
from src.utils.rendering import OM_rendering
from src.utils.experiment import create_experiment
from src.models import model_v2 # <--- 导入新模型
from src.utils.camera import get_rays
from src.data.dataset import SoftSequenceDataset

# --- 全局设置 ---
CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# ==========================================
# 1. 基础工具
# ==========================================
# use centralized get_rays from src.utils.camera

# ==========================================
# 2. 训练主程序
# ==========================================
def train_seq_vis():
    """训练带可视化输出的序列模型并定期导出 GIF。"""
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 40            
    BATCH_SIZE = 4
    LR = 5e-4  # 稍微调高一点学习率
    N_EPOCHS = 50
    VIS_INTERVAL = 1
    BASE_LOG_DIR = os.path.join("train_log", "train_log_seq_vis")

    # ── 数据 ──

    # 1. 划分数据
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    train_files = all_files[:-1]
    val_files = [all_files[-1]]
    # 2. 初始化数据集 (自动计算 Norm)
    train_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=train_files)
    # 验证集使用训练集的 Norm 参数，保证一致性
    val_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=val_files, norm_factor=train_ds.norm_factor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    
    # 3. 初始化模型 (使用 Skip Connection 版本)
    model = model_v2(
        action_dim=train_ds.action_dim,
        seq_len=SEQ_LEN,
        hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    
    # 相机
    CAM_EYE = (1.5, 0.0, 0.5)
    CAM_CENTER = (0.0, 0.0, 0.25)
    CAM_UP = (0.0, 0.0, 1.0)
    rays_o, rays_d = get_rays(
        train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), 
        CAM_EYE, CAM_CENTER, CAM_UP, device=device
    )
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    NEAR, FAR = 0.5, 2.5
    N_SAMPLES = 64

    # ── 创建实验目录并保存配置 ──
    config_dict = {
        "model": "model_v2 (LSTM + Skip + NeRF)",
        "action_dim": train_ds.action_dim,
        "seq_len": SEQ_LEN,
        "hidden_dim": 256,
        "training": {
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "optimizer": "Adam",
        },
        "camera": {
            "eye": list(CAM_EYE),
            "center": list(CAM_CENTER),
            "up": list(CAM_UP),
            "near": NEAR,
            "far": FAR,
            "n_samples": N_SAMPLES,
        },
        "data": {
            "norm_factor": train_ds.norm_factor,
            "train_files": len(train_files),
            "val_files": len(val_files),
            "image_size": [train_ds.H, train_ds.W],
        },
    }
    LOG_DIR = create_experiment(BASE_LOG_DIR, config_dict)

    # 保存 Norm 参数供推理使用
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])

    # --- 4. 核心渲染 (适配新模型接口) ---
    def run_batch(batch_actions):
        """对一个 batch 执行分块渲染前向。

        Args:
            batch_actions: (B, T, D) 动作序列。

        Returns:
            预测图像展平张量 (B, H*W)。
        """
        curr_bs = batch_actions.shape[0]
        
        # A. 提取“当前动作” (B, D) - 用于直连
        current_action = batch_actions[:, -1, :] 
        
        # B. 提取“历史状态” (B, Hidden)
        batch_states = model.encode_temporal(batch_actions)
        
        # C. 空间采样
        t_vals = torch.linspace(0., 1., N_SAMPLES, device=device)
        z_vals = NEAR * (1. - t_vals) + FAR * (t_vals)
        z_vals = z_vals.expand(rays_o.shape[0], N_SAMPLES)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) 
        
        all_preds = []
        total_pixels = pts.shape[0]
        chunk_size = 512
        
        for i in range(0, total_pixels, chunk_size):
            pts_chunk = pts[i : i+chunk_size] 
            curr_chunk_len = pts_chunk.shape[0]
            
            # D. 扩展数据
            # 几何点
            pts_input = pts_chunk.unsqueeze(0).expand(curr_bs, -1, -1, -1).reshape(-1, N_SAMPLES, 3)
            # 状态 (LSTM)
            state_input = batch_states.unsqueeze(1).expand(-1, curr_chunk_len, -1).reshape(-1, batch_states.shape[-1])
            # 动作 (Skip Connection)
            action_input = current_action.unsqueeze(1).expand(-1, curr_chunk_len, -1).reshape(-1, train_ds.action_dim)
            
            # E. 解码 (传入三个参数)
            raw_out = model.decode_spatial(pts_input, state_input, action_input)
            
            rgb_chunk, _ = OM_rendering(raw_out)
            rgb_chunk = rgb_chunk.view(curr_bs, curr_chunk_len)
            all_preds.append(rgb_chunk)
            
        return torch.cat(all_preds, dim=1)

    # --- 5. 验证可视化逻辑 ---
    def evaluate_and_save_gif(epoch_idx):
        """在验证集上评估并保存当前 epoch 的可视化 GIF。"""
        print(f"Generating GIF for Epoch {epoch_idx}...")
        model.eval()
        pred_frames = []
        gt_frames = []
        val_loss_total = 0
        
        with torch.no_grad():
            for v_input, v_target in tqdm(val_loader, desc="Validating"):
                v_input = v_input.to(device); v_target = v_target.to(device)
                v_pred_flat = run_batch(v_input)
                loss = criterion(v_pred_flat, v_target)
                val_loss_total += loss.item()
                
                curr_preds = v_pred_flat.reshape(-1, train_ds.H, train_ds.W).cpu().numpy()
                curr_gts = v_target.reshape(-1, train_ds.H, train_ds.W).cpu().numpy()
                for k in range(curr_preds.shape[0]):
                    pred_frames.append(curr_preds[k])
                    gt_frames.append(curr_gts[k])
        
        avg_val_loss = val_loss_total / len(val_loader)
        
        # 绘图
        raw_actions = val_ds.get_raw_actions(seq_id=0)
        skip = 5 # 降采样
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1]); ax3 = fig.add_subplot(gs[0,2])
        
        im_gt = ax1.imshow(gt_frames[0], cmap='gray', vmin=0, vmax=1); ax1.set_title("GT"); ax1.axis('off')
        im_pred = ax3.imshow(pred_frames[0], cmap='gray', vmin=0, vmax=1); ax3.set_title(f"Pred (Ep {epoch_idx})"); ax3.axis('off')
        
        for d in range(raw_actions.shape[1]):
            ax2.plot(raw_actions[::skip, d], alpha=0.5, label=f'Act {d}')
        vline = ax2.axvline(x=0, color='r'); ax2.legend(); ax2.set_title("Action")
        
        def update(frame):
            real_idx = frame * skip
            if real_idx >= len(pred_frames): return im_gt, im_pred, vline
            im_gt.set_data(gt_frames[real_idx])
            im_pred.set_data(pred_frames[real_idx])
            vline.set_xdata([frame, frame])
            return im_gt, im_pred, vline

        ani = animation.FuncAnimation(fig, update, frames=len(pred_frames)//skip, blit=True)
        ani.save(os.path.join(LOG_DIR, "vis", f"epoch_{epoch_idx}.gif"), writer='pillow', fps=15)
        plt.close()
        return avg_val_loss

    # --- 6. 训练循环 ---
    print(f">>> Start Training (SeqLen={SEQ_LEN}, Norm={train_ds.norm_factor:.4f})...")
    
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        
        for input_seq, target_img in pbar:
            input_seq = input_seq.to(device); target_img = target_img.to(device)
            optimizer.zero_grad()
            pred_img = run_batch(input_seq)
            loss = criterion(pred_img, target_img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        if epoch % VIS_INTERVAL == 0:
            val_loss = evaluate_and_save_gif(epoch)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_seq_model.pt"))
        else:
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")

if __name__ == "__main__":
    train_seq_vis()
    
    