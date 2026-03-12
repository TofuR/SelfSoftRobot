import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入
from func import OM_rendering
from src.models import model_v4_nerf_pinn as model_module # 请确保文件名匹配

# --- 全局设置 ---
CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training NeRF-PINN (V4) on device: {device}")

# ==========================================
# 1. 工具与数据 (保持原逻辑，确保3D一致性)
# ==========================================
def get_rays_from_camera_params(H, W, focal, eye, center, up):
    """根据针孔相机参数生成每个像素对应射线。

    Args:
        H: 图像高度。
        W: 图像宽度。
        focal: 相机焦距。
        eye: 相机位置。
        center: 注视点。
        up: 上方向向量。

    Returns:
        (rays_o, rays_d)，展平后均为 (H*W, 3)。
    """
    # ... (保持原有的正确实现) ...
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    view_dir = center - eye; view_dir = view_dir / torch.norm(view_dir)
    right = torch.linalg.cross(view_dir, up); right = right / torch.norm(right)
    true_up = torch.linalg.cross(right, view_dir); true_up = true_up / torch.norm(true_up)
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='ij')
    dir_x = (i - W * 0.5) / focal; dir_y = -(j - H * 0.5) / focal; dir_z = torch.ones_like(dir_x)
    rays_d = (dir_x[..., None] * right + dir_y[..., None] * true_up + dir_z[..., None] * view_dir)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye.expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

class SoftSequenceDataset(Dataset):
    """Sequence dataset that pairs action history with target image.

    __getitem__ returns (seq, img_flat):
        seq: (seq_len, action_dim) normalized actions.
        img_flat: flattened grayscale image at the same timestep.
    """
    def __init__(self, data_dir, seq_len=10, file_list=None, norm_factor=None):
        self.seq_len = seq_len; self.samples = []
        if file_list is None: file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # 归一化计算
        if norm_factor is None:
            all_acts = []
            for f in file_list: all_acts.append(np.load(f)['actions'])
            if all_acts: self.norm_factor = np.max(np.abs(np.concatenate(all_acts)))
            else: self.norm_factor = 1.0
            if self.norm_factor == 0: self.norm_factor = 1.0
            print(f"Norm Factor: {self.norm_factor}")
        else: self.norm_factor = norm_factor

        self.data_cache = []
        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor
            # 注意：这里不需要 Resize 图片，因为我们是用射线采样的，需要原始分辨率的 focal
            self.data_cache.append({'images': raw['images'], 'actions': actions, 'length': len(raw['images'])})
            
        for seq_id, item in enumerate(self.data_cache):
            for t in range(item['length']):
                self.samples.append((seq_id, t))
        self.H, self.W = self.data_cache[0]['images'].shape[1:]
        self.action_dim = self.data_cache[0]['actions'].shape[1]
        self.focal = float(raw.get('focal', 130.0))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        """读取单条样本并返回补齐后的动作窗口。

        Args:
            idx: 样本索引。

        Returns:
            seq: (seq_len, action_dim)
            img_flat: (H*W,)
        """
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]
        start, end = t - self.seq_len + 1, t + 1
        if start >= 0: seq = data['actions'][start:end]
        else: 
            pad = np.zeros((self.seq_len - (end), self.action_dim))
            seq = np.concatenate([pad, data['actions'][0:end]], axis=0)
        return torch.from_numpy(seq).float(), torch.from_numpy(data['images'][t]).float().reshape(-1)
    
    def get_raw_actions(self, seq_id=0): return self.data_cache[seq_id]['actions'] * self.norm_factor

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
    LOG_DIR = "train_log_v4_nerf"
    
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
    rays_o_full, rays_d_full = get_rays_from_camera_params(
        train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), CAM_EYE, CAM_CENTER, CAM_UP
    )
    rays_o_full = rays_o_full.reshape(-1, 3); rays_d_full = rays_d_full.reshape(-1, 3)
    
    NEAR, FAR, N_SAMPLES = 0.5, 2.5, 64

    # --- 渲染核心 ---
    def run_batch_rendering(batch_actions, is_train=True):
        """对一个 batch 执行渲染前向。

        Args:
            batch_actions: (B, Seq, action_dim) 动作序列。
            is_train: 训练模式下是否随机采样射线并计算物理平滑损失。

        Returns:
            pred_img: (B, R) 预测像素。
            loss_phy: 物理平滑损失。
            ray_indices: 训练模式下的采样射线索引；验证模式为 None。
        """
        curr_bs = batch_actions.shape[0]
        
        # 1. 获取整个序列的物理状态 (B, Seq, Hidden)
        #    我们需要整个序列来计算 Physics Loss
        latent_seq = model.get_physics_state(batch_actions)
        
        # 2. 提取当前时刻的状态 (最后一步) 用于渲染
        #    (B, Hidden)
        current_state = latent_seq[:, -1, :]
        current_action = batch_actions[:, -1, :]

        # 3. 计算 Physics Loss (针对 latent_seq 的平滑性)
        if is_train:
            loss_phy = model.compute_smoothness_loss(latent_seq)
        else:
            loss_phy = 0.0

        # 4. 射线采样与渲染
        #    为了节省显存，训练时我们随机采样一部分射线，验证时分块渲染全图
        if is_train:
            # 随机采样 1024 条射线
            ray_indices = torch.randint(0, rays_o_full.shape[0], (1024,), device=device)
            rays_o = rays_o_full[ray_indices]
            rays_d = rays_d_full[ray_indices]
        else:
            # 验证时使用全图 (分块处理在下面做)
            rays_o = rays_o_full
            rays_d = rays_d_full

        # 采样 Z 值
        t_vals = torch.linspace(0., 1., N_SAMPLES, device=device)
        z_vals = NEAR * (1. - t_vals) + FAR * (t_vals)
        z_vals = z_vals.expand(rays_o.shape[0], N_SAMPLES)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

        # --- 显存优化：分块送入 Decoder ---
        all_rgb = []
        chunk_size = 2048
        
        # 如果是训练，rays_o 只有 1024，这里只会循环一次
        # 如果是验证，rays_o 是全图，这里会多次循环
        for i in range(0, rays_o.shape[0], chunk_size):
            pts_chunk = pts[i : i+chunk_size]
            n_rays_chunk = pts_chunk.shape[0]
            
            # 扩展 Batch 信息到 Ray 级别
            # 注意：这里的逻辑稍微有点绕。
            # 如果是训练(Batch>1)，我们通常只算 MSE Loss 平均值，
            # 这里的简单做法是：让 Batch 里的每个样本都渲染这些射线，然后取平均？
            # 不，NeRF 训练的标准做法是：Batch 里的每一条数据对应一张图。
            # 但我们这里 rays_o 是固定的。
            # 简化方案：我们一次只训练 Batch 中的一个样本，或者把 Batch 维度和 Ray 维度混合。
            
            # 为了让逻辑通顺：我们把 batch_actions 里的每个样本视为独立的 "World State"
            # 我们需要输出 (Batch, N_rays) 的图像
            
            # 扩展 State: (Batch, Hidden) -> (Batch, N_rays_chunk, Hidden)
            state_in = current_state.unsqueeze(1).expand(-1, n_rays_chunk, -1)
            # 扩展 Action: (Batch, Dim) -> (Batch, N_rays_chunk, Dim)
            act_in = current_action.unsqueeze(1).expand(-1, n_rays_chunk, -1)
            # 扩展 Points: (N_rays_chunk, Samp, 3) -> (Batch, N_rays_chunk, Samp, 3)
            pts_in = pts_chunk.unsqueeze(0).expand(curr_bs, -1, -1, -1)
            
            # 压扁 Batch 和 Rays 维度喂给网络
            # Input: (Batch * N_rays_chunk, Samp, ...)
            raw_out = model.forward_rendering(
                pts_in.reshape(-1, N_SAMPLES, 3), 
                state_in.reshape(-1, 128),
                act_in.reshape(-1, train_ds.action_dim)
            )
            
            # 渲染
            rgb, _ = OM_rendering(raw_out)
            # 恢复维度 (Batch, N_rays_chunk)
            rgb = rgb.view(curr_bs, n_rays_chunk)
            all_rgb.append(rgb)
            
        pred_img = torch.cat(all_rgb, dim=1) # (Batch, N_rays)
        
        return pred_img, loss_phy, ray_indices if is_train else None

    # --- Training Loop ---
    print(f">>> Start Training NeRF-PINN...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_recon = 0; total_phy = 0
        
        for input_seq, target_img_flat in tqdm(train_loader, desc=f"Ep {epoch}"):
            input_seq = input_seq.to(device); target_img_flat = target_img_flat.to(device)
            
            optimizer.zero_grad()
            
            # 1. 渲染 (Batch, N_sampled_rays)
            pred_pixels, loss_phy, ray_idx = run_batch_rendering(input_seq, is_train=True)
            
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
            evaluate_and_save(model, val_loader, epoch, LOG_DIR, train_ds.H, train_ds.W, run_batch_rendering)
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

def evaluate_and_save(model, loader, epoch, log_dir, H, W, render_fn):
    """渲染一个验证 batch 并保存 GT/预测对比图。"""
    model.eval()
    with torch.no_grad():
        seqs, imgs = next(iter(loader))
        seqs = seqs.to(device)
        
        # 渲染全图 (Batch, H*W)
        pred_flat, _, _ = render_fn(seqs, is_train=False)
        
        # Reshape
        pred_img = pred_flat[0].reshape(H, W).cpu().numpy()
        gt_img = imgs[0].reshape(H, W).numpy()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(gt_img, cmap='gray'); ax1.set_title("GT")
        ax2.imshow(pred_img, cmap='gray'); ax2.set_title("Pred")
        plt.savefig(os.path.join(log_dir, "vis", f"ep_{epoch}.png"))
        plt.close()

if __name__ == "__main__":
    train_v4_nerf()
    
    