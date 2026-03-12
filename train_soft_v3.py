import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

# 导入模型 (请确保文件名正确)
from model_seq_skip_pinn import model_v3

# --- 全局设置 ---
CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training V3 on device: {device}")

# ==========================================
# 1. 图像序列数据集 (自动 Resize 64x64)
# ==========================================
class ImageSequenceDataset(Dataset):
    """图像序列数据集，负责动作归一化与图像缩放。"""
    def __init__(self, data_dir, seq_len=20, target_size=64, file_list=None, norm_factor=None):
        self.seq_len = seq_len
        self.target_size = target_size
        self.samples = []
        
        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # 预扫描计算归一化系数
        if norm_factor is None:
            all_acts = []
            for f in file_list:
                d = np.load(f)
                all_acts.append(d['actions'])
            all_acts = np.concatenate(all_acts, axis=0)
            self.norm_factor = np.max(np.abs(all_acts)) if len(all_acts) > 0 else 1.0
            if self.norm_factor == 0: self.norm_factor = 1.0
            print(f"Auto-calculated Normalization Factor: {self.norm_factor}")
        else:
            self.norm_factor = norm_factor

        print(f"Loading {len(file_list)} sequence files...")
        self.data_cache = []
        
        for f_path in file_list:
            raw = np.load(f_path)
            # 动作归一化
            actions = raw['actions'] / self.norm_factor
            
            # 图像预处理: Resize -> 归一化 -> 增加通道维
            imgs_raw = raw['images'] # [T, 100, 100]
            imgs_processed = []
            for img in imgs_raw:
                # Resize 到 64x64
                img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                # 确保在 [0, 1]
                if img_resized.max() > 1.0: img_resized /= 255.0
                imgs_processed.append(img_resized)
            
            # [T, 1, 64, 64]
            images = np.stack(imgs_processed, axis=0)[:, np.newaxis, :, :]
            
            self.data_cache.append({
                'images': images, 
                'actions': actions, 
                'length': len(images)
            })
            
        # 构建滑动窗口索引
        for seq_id, item in enumerate(self.data_cache):
            # 从 seq_len 开始，保证有足够的历史
            # 为了简化，我们只取能够构成完整 seq_len 的窗口
            # 这里的逻辑是：一个样本包含 [t, t+1, ..., t+seq_len-1] 的序列
            T = item['length']
            if T > seq_len:
                for t in range(T - seq_len + 1):
                    self.samples.append((seq_id, t))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        """返回一个完整序列窗口样本。

        Args:
            idx: 样本索引。

        Returns:
            image_seq: (S, 1, 64, 64)
            action_seq: (S, D)
        """
        seq_id, start_t = self.samples[idx]
        data = self.data_cache[seq_id]
        
        end_t = start_t + self.seq_len
        
        # 获取切片
        image_seq = data['images'][start_t:end_t]   # [S, 1, 64, 64]
        action_seq = data['actions'][start_t:end_t] # [S, D]
        
        return torch.from_numpy(image_seq).float(), torch.from_numpy(action_seq).float()

# ==========================================
# 2. 训练与验证逻辑
# ==========================================
def train_v3():
    """训练 V3（PINN）图像序列模型。"""
    # --- 超参数 ---
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 20
    BATCH_SIZE = 16 
    LR = 1e-4
    N_EPOCHS = 50
    VIS_INTERVAL = 5
    LAMBDA_PHY = 0.1 # 物理损失权重
    LOG_DIR = "train_log_v3"
    
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    # 数据集准备
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not all_files: raise ValueError("No data found!")
    
    # 留最后一段做验证
    train_files = all_files[:-1]
    val_files = [all_files[-1]]
    
    train_ds = ImageSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=train_files)
    val_ds = ImageSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=val_files, norm_factor=train_ds.norm_factor)
    
    # 保存归一化参数
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    # 模型初始化
    # 输入通道1 (灰度), 动作维度自动获取
    act_dim = train_ds.data_cache[0]['actions'].shape[1]
    model = model_v3(
        n_inputs=1, 
        n_outputs=act_dim, 
        input_channel=1, 
        hidden_size=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f">>> Start Training V3 (PINN Mode)...")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        
        for imgs, acts in pbar:
            imgs, acts = imgs.to(device), acts.to(device)
            # imgs: [B, S, 1, 64, 64], acts: [B, S, D]
            
            optimizer.zero_grad()
            
            # 前向传播
            y_pred, latent_seq = model(imgs, acts)
            
            # 计算损失
            recon_loss = criterion(y_pred, imgs)
            phy_loss = model.get_physics_loss(latent_seq)
            
            loss = recon_loss + LAMBDA_PHY * phy_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'recon': recon_loss.item(), 'phy': phy_loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

        # --- 验证与可视化 ---
        if epoch % VIS_INTERVAL == 0:
            visualize_and_save(model, val_loader, epoch, LOG_DIR)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", f"model_{epoch}.pt"))
            # 总是保存最新
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

def visualize_and_save(model, loader, epoch, log_dir):
    """在验证集上可视化并保存当前 epoch 的预测 GIF。"""
    model.eval()
    # 取一个 batch 做可视化
    with torch.no_grad():
        imgs, acts = next(iter(loader))
        imgs, acts = imgs.to(device), acts.to(device)
        
        y_pred, _ = model(imgs, acts)
        
        # 取第一个样本的序列
        # imgs: [B, S, 1, 64, 64]
        gt_seq = imgs[0].cpu().numpy().squeeze()   # [S, 64, 64]
        pred_seq = y_pred[0].cpu().numpy().squeeze() # [S, 64, 64]
        
        # 生成 GIF
        save_path = os.path.join(log_dir, "vis", f"epoch_{epoch}.gif")
        create_comparison_gif(gt_seq, pred_seq, save_path)

def create_comparison_gif(gt_seq, pred_seq, filename):
    """生成 GT 与预测结果的对比 GIF。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction (V3)")
    ax1.axis('off'); ax2.axis('off')
    
    im1 = ax1.imshow(gt_seq[0], cmap='gray', vmin=0, vmax=1)
    im2 = ax2.imshow(pred_seq[0], cmap='gray', vmin=0, vmax=1)
    
    def update(frame):
        im1.set_data(gt_seq[frame])
        im2.set_data(pred_seq[frame])
        return im1, im2
    
    ani = animation.FuncAnimation(fig, update, frames=len(gt_seq), blit=True)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    train_v3()