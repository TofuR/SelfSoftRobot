from sympy import im
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import os
import glob
import cv2
import time

# ==========================================
# 1. 模型架构 (Model V3)
# ==========================================
from src.models import model_v3

# ==========================================
# 2. 数据加载类 (Dataset)
# ==========================================
class NPZSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = glob.glob(os.path.join(data_dir, "*.npz"))
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        print(f"Found {len(self.file_list)} sequences.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        
        # 图像处理: [Seq, H, W] -> [Seq, 1, 64, 64], 归一化到 [0, 1]
        imgs = data['images'].astype(np.float32) / 255.0
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, axis=1) # 增加 Channel 维度
        
        # 驱动参数: [Seq, Actuator_Dim]
        acts = data['actuators'].astype(np.float32)
        
        return torch.from_numpy(imgs), torch.from_numpy(acts)

# ==========================================
# 3. 训练主程序
# ==========================================
def main():
    # --- 配置参数 ---
    DATA_DIR = "./data/sequence_data"
    SAVE_DIR = "./results_v3"
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 200
    LAMBDA_PHYS = 0.5 # 物理约束权重
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据准备
    dataset = NPZSequenceDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_v3(n_actuators=4, input_channel=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_recon = nn.MSELoss()

    print(f"Starting training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_recon_loss = 0
        epoch_phys_loss = 0
        
        start_time = time.time()
        for i, (imgs, acts) in enumerate(dataloader):
            imgs, acts = imgs.to(device), acts.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            preds, latent_seq = model(imgs, acts)
            
            # 计算损失
            recon_loss = criterion_recon(preds, imgs)
            phys_loss = model.get_physics_loss(latent_seq)
            
            total_loss = recon_loss + LAMBDA_PHYS * phys_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_recon_loss += recon_loss.item()
            epoch_phys_loss += phys_loss.item()

        # --- 可视化与保存 ---
        if epoch % 10 == 0:
            save_vis_comparison(epoch, imgs[0], preds[0], SAVE_DIR)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_v3_epoch_{epoch}.pth"))
            
        duration = time.time() - start_time
        print(f"Epoch [{epoch}/{EPOCHS}] | Recon: {epoch_recon_loss/len(dataloader):.6f} | "
              f"Phys: {epoch_phys_loss/len(dataloader):.6f} | Time: {duration:.2f}s")

def save_vis_comparison(epoch, gt_seq, pred_seq, save_dir):
    """
    保存 GT 和 Prediction 的序列对比长图
    """
    gt_np = (gt_seq.detach().cpu().numpy() * 255).astype(np.uint8)
    pred_np = (pred_seq.detach().cpu().numpy() * 255).astype(np.uint8)
    
    seq_len = gt_np.shape[0]
    frames = []
    for s in range(seq_len):
        # 横向拼接: 左边是真值, 右边是模型自建模结果
        combined = np.hstack([gt_np[s, 0], pred_np[s, 0]])
        # 放大一点方便观察
        combined = cv2.resize(combined, (256, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(combined, f"S:{s}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
        frames.append(combined)
    
    # 纵向堆叠所有时间步
    final_img = np.vstack(frames)
    cv2.imwrite(os.path.join(save_dir, f"vis_epoch_{epoch}.png"), final_img)

if __name__ == "__main__":
    main()