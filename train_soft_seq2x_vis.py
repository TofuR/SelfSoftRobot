import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入
from func import OM_rendering
from model_seq_skip import RecurrentFBV_SM_Skip # <--- 导入新模型

# --- 全局设置 ---
CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# ==========================================
# 1. 基础工具
# ==========================================
def get_rays_from_camera_params(H, W, focal, eye, center, up):
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
        indexing='ij')
    dir_x = (i - W * 0.5) / focal
    dir_y = -(j - H * 0.5) / focal
    dir_z = torch.ones_like(dir_x)
    rays_d = (dir_x[..., None] * right + dir_y[..., None] * true_up + dir_z[..., None] * view_dir)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye.expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

# ==========================================
# 2. 改进的数据集 (自动归一化)
# ==========================================
class SoftSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=10, file_list=None, norm_factor=None):
        self.seq_len = seq_len
        self.samples = []
        
        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # [修改] 预扫描计算归一化系数
        if norm_factor is None:
            all_acts = []
            print("Scanning data for normalization...")
            for f in file_list:
                d = np.load(f)
                all_acts.append(d['actions'])
            all_acts = np.concatenate(all_acts, axis=0)
            self.norm_factor = np.max(np.abs(all_acts))
            if self.norm_factor == 0: self.norm_factor = 1.0
            print(f"Auto-calculated Normalization Factor: {self.norm_factor}")
        else:
            self.norm_factor = norm_factor

        print(f"Loading {len(file_list)} sequence files...")
        self.data_cache = []
        for f_path in file_list:
            raw = np.load(f_path)
            # 使用正确的系数归一化
            actions = raw['actions'] / self.norm_factor
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
        actions_full = data['actions']
        target_image = data['images'][t]
        
        start_idx, end_idx = t - self.seq_len + 1, t + 1
        if start_idx >= 0:
            input_seq = actions_full[start_idx:end_idx]
        else:
            valid_part = actions_full[0:end_idx]
            padding = np.zeros((self.seq_len - len(valid_part), self.action_dim), dtype=valid_part.dtype)
            input_seq = np.concatenate([padding, valid_part], axis=0)
            
        return torch.from_numpy(input_seq).float(), torch.from_numpy(target_image).float().reshape(-1)

    def get_raw_actions(self, seq_id=0):
        # 返回未归一化的原始动作，用于绘图
        return self.data_cache[seq_id]['actions'] * self.norm_factor

# ==========================================
# 3. 训练主程序
# ==========================================
def train_seq_vis():
    DATA_DIR = "data/sequence_data"
    SEQ_LEN = 40            
    BATCH_SIZE = 4
    LR = 5e-4  # 稍微调高一点学习率
    N_EPOCHS = 50
    VIS_INTERVAL = 1        
    LOG_DIR = "train_log_seq_vis/experiment_2"
    
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

    # 1. 划分数据
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    train_files = all_files[:-1]
    val_files = [all_files[-1]]
    
    # 2. 初始化数据集 (自动计算 Norm)
    train_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=train_files)
    # 验证集使用训练集的 Norm 参数，保证一致性
    val_ds = SoftSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, file_list=val_files, norm_factor=train_ds.norm_factor)
    
    # 保存 Norm 参数供推理使用
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    
    # 3. 初始化模型 (使用 Skip Connection 版本)
    model = RecurrentFBV_SM_Skip(
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
    rays_o, rays_d = get_rays_from_camera_params(
        train_ds.H, train_ds.W, torch.tensor(train_ds.focal).to(device), 
        CAM_EYE, CAM_CENTER, CAM_UP
    )
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    NEAR, FAR = 0.5, 2.5
    N_SAMPLES = 64

    # --- 4. 核心渲染 (适配新模型接口) ---
    def run_batch(batch_actions):
        """
        batch_actions: (B, T, D)
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
    
    