import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import glob
from matplotlib import animation
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入
from src.utils.rendering import OM_rendering
from src.models import model_v1
from src.utils.camera import get_rays
from src.data.dataset import SoftSequenceDataset

# --- 全局设置 ---
CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# ==========================================
# 训练逻辑 (显存优化版)
# ==========================================
def train_seq():
    """训练序列模型（2x 版本）并保存验证可视化。"""
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 20
    BATCH_SIZE = 8
    LR = 1e-5
    N_EPOCHS = 50
    LOG_DIR = os.path.join("train_log", "train_log_softseq", "experiment_1")
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    split_idx = int(0.8 * len(all_files))
    train_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=all_files[:split_idx])
    val_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=all_files[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = model_v1(action_dim=train_ds.action_dim, seq_len=SEQ_LEN, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    
    # 相机设置
    CAM_EYE = (1.5, 0.0, 0.5)
    CAM_CENTER = (0.0, 0.0, 0.25)
    CAM_UP = (0.0, 0.0, 1.0)
    rays_o, rays_d = get_rays(train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), CAM_EYE, CAM_CENTER, CAM_UP, device=device)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    NEAR, FAR = 0.5, 2.5
    N_SAMPLES = 64

    # --- 优化后的批量处理函数 ---
    def run_batch(batch_actions):
        """按块渲染一个 batch，降低显存占用。

        Args:
            batch_actions: (B, T, D) 动作序列。

        Returns:
            预测图像展平结果，形状 (B, H*W)。
        """
        curr_bs = batch_actions.shape[0]
        
        # 1. 预计算时序状态 (Batch, Hidden) - 整个Batch只算一次！
        # 以前是每个像素算一次，这是显存爆炸的根源
        batch_states = model.encode_temporal(batch_actions) 
        
        # 2. 空间采样
        t_vals = torch.linspace(0., 1., N_SAMPLES, device=device)
        z_vals = NEAR * (1. - t_vals) + FAR * (t_vals)
        z_vals = z_vals.expand(rays_o.shape[0], N_SAMPLES)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (Pixels, Samples, 3)
        
        all_preds = []
        total_pixels = pts.shape[0]
        
        # [修改] 减小 Chunk Size (2048 -> 512)
        chunk_size = 512 
        
        for i in range(0, total_pixels, chunk_size):
            # (Chunk, Samp, 3)
            pts_chunk = pts[i : i+chunk_size] 
            curr_chunk_len = pts_chunk.shape[0]
            
            # --- 扩展数据以适配 Batch ---
            # 目标: (B * Chunk, Samp, 3)
            # 复制点坐标: 每个Batch的机器人在这一块像素的“几何视线”是一样的
            pts_input = pts_chunk.unsqueeze(0).expand(curr_bs, -1, -1, -1).reshape(-1, N_SAMPLES, 3)
            
            # 目标: (B * Chunk, Hidden)
            # 复制状态: 同一个Batch内的所有像素共享同一个物理状态
            state_input = batch_states.unsqueeze(1).expand(-1, curr_chunk_len, -1).reshape(-1, batch_states.shape[-1])
            
            # 空间解码
            raw_out = model.decode_spatial(pts_input, state_input)
            
            # 渲染
            rgb_chunk, _ = OM_rendering(raw_out)
            rgb_chunk = rgb_chunk.view(curr_bs, curr_chunk_len)
            all_preds.append(rgb_chunk)
            
        return torch.cat(all_preds, dim=1)

    print(f">>> Start Training Sequence Model (Batch={BATCH_SIZE}, SeqLen={SEQ_LEN})...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for input_seq, target_img in pbar:
            input_seq, target_img = input_seq.to(device), target_img.to(device)
            
            optimizer.zero_grad()
            pred_img = run_batch(input_seq)
            loss = criterion(pred_img, target_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_input, v_target in val_loader:
                v_input, v_target = v_input.to(device), v_target.to(device)
                v_pred = run_batch(v_input)
                loss = criterion(v_pred, v_target)
                val_loss += loss.item()
                break  # 只计算一个 batch 的损失
            
            if epoch % 5 == 0:
                # 生成验证 GIF
                val_data = val_ds.data_cache[0]  # 取第一个验证序列
                seq_length = val_data['length']
                
                gt_imgs = []
                pred_imgs = []
                
                for t in range(seq_length):
                    # 构建输入序列
                    seq_len = val_ds.seq_len
                    start = max(0, t - seq_len + 1)
                    actions_seq = val_data['actions'][start:t+1]
                    if len(actions_seq) < seq_len:
                        pad = np.zeros((seq_len - len(actions_seq), val_ds.action_dim))
                        actions_seq = np.concatenate([pad, actions_seq], axis=0)
                    
                    input_seq = torch.from_numpy(actions_seq).float().unsqueeze(0).to(device)
                    target_img = torch.from_numpy(val_data['images'][t]).float().unsqueeze(0).to(device)
                    
                    # 预测
                    pred_img_flat = run_batch(input_seq)
                    pred_img = pred_img_flat[0].reshape(val_ds.H, val_ds.W).cpu().numpy()
                    gt_img = target_img[0].reshape(val_ds.H, val_ds.W).cpu().numpy()
                    
                    gt_imgs.append(gt_img)
                    pred_imgs.append(pred_img)
                
                # 生成 GIF
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.set_title("Ground Truth")
                ax2.set_title("Prediction")
                ax1.axis('off'); ax2.axis('off')
                
                im1 = ax1.imshow(gt_imgs[0], cmap='gray', vmin=0, vmax=1)
                im2 = ax2.imshow(pred_imgs[0], cmap='gray', vmin=0, vmax=1)
                
                def update(frame):
                    im1.set_data(gt_imgs[frame])
                    im2.set_data(pred_imgs[frame])
                    ax1.set_title(f"GT (Frame {frame})")
                    ax2.set_title(f"Pred (Frame {frame})")
                    return im1, im2
                
                ani = animation.FuncAnimation(fig, update, frames=len(gt_imgs), blit=True)
                save_path = os.path.join(LOG_DIR, "vis", f"epoch_{epoch}_val.gif")
                ani.save(save_path, writer='pillow', fps=10)
                plt.close()
                print(f"    Saved validation GIF to {save_path}")
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_seq_model.pt"))

if __name__ == "__main__":
    train_seq()
    