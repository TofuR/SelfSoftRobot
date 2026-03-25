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

# 导入 V4 模型
from src.models.model_seq_open_loop import RecurrentPhysicsModel

# --- 全局设置 ---
CUDA_DEVICE = 0  # 请根据实际情况修改
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training V4 on device: {device}")

# ==========================================
# 1. 改进的数据集 (适配 CNN 输入 64x64)
# ==========================================
class ImageSequenceDataset(Dataset):
    """V4 训练数据集：构建固定长度窗口并执行图像预处理。"""
    def __init__(self, data_dir, seq_len=20, target_size=64, file_list=None, norm_factor=None):
        self.seq_len = seq_len
        self.target_size = target_size
        self.samples = []
        
        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # 1. 自动计算归一化系数 (参考 train_soft_seq2x_vis.py)
        if norm_factor is None:
            all_acts = []
            print("Scanning data for normalization...")
            for f in file_list:
                try:
                    d = np.load(f)
                    all_acts.append(d['actions'])
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            
            if len(all_acts) > 0:
                all_acts = np.concatenate(all_acts, axis=0)
                self.norm_factor = np.max(np.abs(all_acts))
            else:
                self.norm_factor = 1.0
                
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
            
            # --- 图像处理关键步骤 ---
            # 原始图像是 (T, 100, 100)
            imgs_raw = raw['images'] 
            
            # 必须 Resize 到 64x64 以适配 V4 模型的 Decoder (4次上采样: 4->8->16->32->64)
            imgs_processed = []
            for img in imgs_raw:
                # 确保是 float32 且在 0-1 之间，避免 cv2 转换出错
                if img.max() > 1.0: 
                    img = img.astype(np.float32) / 255.0
                
                # Resize
                img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                imgs_processed.append(img_resized)
            
            # 堆叠并增加通道维: (T, 64, 64) -> (T, 1, 64, 64)
            images = np.stack(imgs_processed, axis=0)[:, np.newaxis, :, :]
            
            self.data_cache.append({
                'images': images, 
                'actions': actions, 
                'length': len(images)
            })
            
        # 构建滑动窗口索引 (仅保留完整窗口)
        # 逻辑：每个样本包含从 t 到 t+seq_len 的完整序列
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            if T > seq_len:
                for t in range(T - seq_len + 1):
                    self.samples.append((seq_id, t))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        """读取单条序列样本。

        Args:
            idx: 样本索引。

        Returns:
            image_seq: (Seq, 1, 64, 64)
            action_seq: (Seq, D)
        """
        seq_id, start_t = self.samples[idx]
        data = self.data_cache[seq_id]
        
        end_t = start_t + self.seq_len
        
        # 获取完整的序列窗口
        image_seq = data['images'][start_t:end_t]   # [Seq, 1, 64, 64]
        action_seq = data['actions'][start_t:end_t] # [Seq, D]
        
        return torch.from_numpy(image_seq).float(), torch.from_numpy(action_seq).float()

    def get_raw_actions(self, seq_id=0):
        # 用于可视化
        return self.data_cache[seq_id]['actions'] * self.norm_factor

# ==========================================
# 2. 训练主程序
# ==========================================
def train_v4():
    """训练 V4 开环序列模型。"""
    # --- 超参数 ---
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 30    # V4 是开环模型，序列长一点更能测试稳定性
    BATCH_SIZE = 16 
    LR = 2e-4
    N_EPOCHS = 50
    VIS_INTERVAL = 5
    LAMBDA_PHY = 0.1 # 物理平滑损失权重
    LOG_DIR = os.path.join("train_log", "train_log_v4")
    
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    # 1. 数据准备
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if len(all_files) == 0:
        raise ValueError(f"No .npz files found in {DATA_DIR}")
        
    # 划分训练集和验证集 (留最后一段做验证)
    if len(all_files) > 1:
        train_files = all_files[:-1]
        val_files = [all_files[-1]]
    else:
        train_files = all_files
        val_files = all_files
    
    # 初始化数据集
    train_ds = ImageSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, target_size=64, file_list=train_files)
    # 验证集使用相同的归一化参数
    val_ds = ImageSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, target_size=64, file_list=val_files, norm_factor=train_ds.norm_factor)
    
    # 保存参数
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    # 2. 模型初始化
    act_dim = train_ds.data_cache[0]['actions'].shape[1]
    
    model = RecurrentPhysicsModel(
        input_channel=1, 
        act_dim=act_dim, 
        hidden_size=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f">>> Start Training V4 (Open-Loop Mode)...")
    print(f"    Seq Len: {SEQ_LEN}, Norm Factor: {train_ds.norm_factor:.5f}")

    # 3. 训练循环
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_recon_loss = 0
        total_phy_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        
        for imgs, acts in pbar:
            imgs, acts = imgs.to(device), acts.to(device)
            # imgs: [B, S, 1, 64, 64]
            # acts: [B, S, D]
            
            optimizer.zero_grad()
            
            # Forward (V4 内部逻辑：只看 imgs[:, 0] 作为初始状态，后续全靠 acts 推演)
            y_pred, latent_seq = model(imgs, acts)
            
            # Loss 1: 重建误差 (对比整个序列的预测值和真实值)
            recon_loss = criterion(y_pred, imgs)
            
            # Loss 2: 物理平滑 (二阶导数约束)
            phy_loss = model.get_physics_loss(latent_seq)
            
            loss = recon_loss + LAMBDA_PHY * phy_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_phy_loss += phy_loss.item()
            
            pbar.set_postfix({'Recon': recon_loss.item(), 'Phy': phy_loss.item()})
            
        avg_recon = total_recon_loss / len(train_loader)
        avg_phy = total_phy_loss / len(train_loader)
        
        # --- 验证与可视化 ---
        if epoch % VIS_INTERVAL == 0:
            print(f"Epoch {epoch} | Recon Loss: {avg_recon:.6f} | Phy Loss: {avg_phy:.6f}")
            visualize_and_save(model, val_loader, epoch, LOG_DIR)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))
        else:
            print(f"Epoch {epoch} | Avg Loss: {avg_recon + LAMBDA_PHY*avg_phy:.6f}")

def visualize_and_save(model, loader, epoch, log_dir):
    """对验证样本进行可视化并保存 GIF。"""
    model.eval()
    with torch.no_grad():
        # 取一个 batch
        imgs, acts = next(iter(loader))
        imgs, acts = imgs.to(device), acts.to(device)
        
        # 推理 (Open Loop)
        y_pred, _ = model(imgs, acts)
        
        # 取第一个样本可视化
        # [S, 1, 64, 64] -> [S, 64, 64]
        gt_seq = imgs[0].cpu().numpy().squeeze()
        pred_seq = y_pred[0].cpu().numpy().squeeze()
        
        # 生成对比 GIF
        save_path = os.path.join(log_dir, "vis", f"epoch_{epoch}.gif")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.set_title("Ground Truth (64x64)")
        ax2.set_title("Pred (Open Loop)")
        ax1.axis('off'); ax2.axis('off')
        
        im1 = ax1.imshow(gt_seq[0], cmap='gray', vmin=0, vmax=1)
        im2 = ax2.imshow(pred_seq[0], cmap='gray', vmin=0, vmax=1)
        
        def update(frame):
            if frame < len(gt_seq):
                im1.set_data(gt_seq[frame])
                im2.set_data(pred_seq[frame])
            return im1, im2
        
        ani = animation.FuncAnimation(fig, update, frames=len(gt_seq), blit=True)
        ani.save(save_path, writer='pillow', fps=10)
        plt.close()
        print(f"    Saved visualization to {save_path}")

if __name__ == "__main__":
    train_v4()