import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入
from func import OM_rendering
from src.models import model_v1

# --- 全局设置 ---
CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# --- 重新定义 get_rays 以避免 import 问题 ---
def get_rays(height, width, focal_length):
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='ij')
    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)], dim=-1)
    rays_d = directions
    rays_o = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand(directions.shape)
    
    rays_d_clone = rays_d.clone()
    rays_d[..., 0], rays_d[..., 2] = rays_d_clone[..., 2].clone(), rays_d_clone[..., 0].clone()
    
    # 使用 linalg.cross 避免警告，或者直接矩阵乘法
    rotation_matrix = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 
                                   device=device, dtype=torch.float32)[None, None]
    rays_d = torch.matmul(rays_d, rotation_matrix)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def get_rays_from_camera_params(H, W, focal, eye, center, up):
    # 简单的 LookAt 实现
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    
    view_dir = center - eye
    view_dir = view_dir / torch.norm(view_dir)
    right = torch.linalg.cross(view_dir, up)
    right = right / torch.norm(right)
    true_up = torch.linalg.cross(right, view_dir)
    true_up = true_up / torch.norm(true_up)
    
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    dir_x = (i - W * 0.5) / focal
    dir_y = -(j - H * 0.5) / focal
    dir_z = torch.ones_like(dir_x)
    
    rays_d = (dir_x[..., None] * right + dir_y[..., None] * true_up + dir_z[..., None] * view_dir)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye.expand_as(rays_d)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

# ==========================================
# 1. Dataset (保持不变)
# ==========================================
class SoftSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=10, file_list=None):
        self.seq_len = seq_len
        self.samples = []
        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not file_list: raise ValueError(f"No .npz files in {data_dir}")
        print(f"Loading {len(file_list)} sequence files...")
        self.data_cache = []
        for f_path in file_list:
            raw = np.load(f_path)
            # 简单归一化
            actions = raw['actions'] / 5.0
            self.data_cache.append({'images': raw['images'], 'actions': actions, 'length': len(raw['images'])})
        for seq_id, item in enumerate(self.data_cache):
            for t in range(item['length']):
                self.samples.append((seq_id, t))
        self.H, self.W = self.data_cache[0]['images'].shape[1:]
        self.action_dim = self.data_cache[0]['actions'].shape[1]
        self.focal = float(raw.get('focal', 130.0))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]
        actions_full, target_image = data['actions'], data['images'][t]
        
        start_idx, end_idx = t - self.seq_len + 1, t + 1
        if start_idx >= 0:
            input_seq = actions_full[start_idx:end_idx]
        else:
            valid_part = actions_full[0:end_idx]
            padding = np.zeros((self.seq_len - len(valid_part), self.action_dim), dtype=valid_part.dtype)
            input_seq = np.concatenate([padding, valid_part], axis=0)
            
        return torch.from_numpy(input_seq).float(), torch.from_numpy(target_image).float().reshape(-1)

# ==========================================
# 2. 训练逻辑 (显存优化版)
# ==========================================
def train_seq():
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 20
    BATCH_SIZE = 8
    LR = 1e-5
    N_EPOCHS = 50
    LOG_DIR = "train_log_softseq/experiment_1"
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
    rays_o, rays_d = get_rays_from_camera_params(train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), CAM_EYE, CAM_CENTER, CAM_UP)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    NEAR, FAR = 0.5, 2.5
    N_SAMPLES = 64

    # --- 优化后的批量处理函数 ---
    def run_batch(batch_actions):
        """
        Memory Efficient Batch Rendering
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
                
                if epoch % 5 == 0:
                    pred_vis = v_pred[0].reshape(train_ds.H, train_ds.W).cpu().numpy()
                    gt_vis = v_target[0].reshape(train_ds.H, train_ds.W).cpu().numpy()
                    plt.figure(figsize=(6, 3))
                    plt.subplot(1, 2, 1); plt.imshow(gt_vis, cmap='gray'); plt.title("GT")
                    plt.subplot(1, 2, 2); plt.imshow(pred_vis, cmap='gray'); plt.title("Pred")
                    plt.savefig(os.path.join(LOG_DIR, "vis", f"epoch_{epoch}_val.png"))
                    plt.close()
                break # 验证只跑一个 batch
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_seq_model.pt"))

if __name__ == "__main__":
    train_seq()
    