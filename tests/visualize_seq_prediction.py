import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import glob

# --- 1. 路径配置 (修复版) ---
# 获取当前脚本所在目录 (.../SelfSoftRobot/tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录作为项目根目录 (.../SelfSoftRobot/)
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以正常导入根目录下的模块了
try:
    from model_seq_skip import RecurrentFBV_SM_Skip
    from func import OM_rendering
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前 sys.path: {sys.path}")
    print("请确保 model_seq_skip.py 和 func.py 位于项目根目录下。")
    sys.exit(1)

# --- 全局设置 ---
CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running visualization on: {DEVICE}")

# ==========================================
#  核心辅助函数
# ==========================================

def get_rays_from_camera_params(H, W, focal, eye, center, up):
    """生成相机射线 (与训练时保持严格一致)"""
    eye = torch.tensor(eye, dtype=torch.float32, device=DEVICE)
    center = torch.tensor(center, dtype=torch.float32, device=DEVICE)
    up = torch.tensor(up, dtype=torch.float32, device=DEVICE)
    
    view_dir = center - eye
    view_dir = view_dir / torch.norm(view_dir)
    # 使用 linalg.cross 避免警告
    right = torch.linalg.cross(view_dir, up)
    right = right / torch.norm(right)
    true_up = torch.linalg.cross(right, view_dir)
    true_up = true_up / torch.norm(true_up)
    
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=DEVICE),
        torch.arange(H, dtype=torch.float32, device=DEVICE),
        indexing='ij'
    )
    
    dir_x = (i - W * 0.5) / focal
    dir_y = -(j - H * 0.5) / focal
    dir_z = torch.ones_like(dir_x)
    
    rays_d = (dir_x[..., None] * right + dir_y[..., None] * true_up + dir_z[..., None] * view_dir)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye.expand_as(rays_d)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def run_inference_frame(model, action_window, current_act, rays_o, rays_d, n_samples=64, near=0.5, far=2.5):
    """
    单帧推理函数
    """
    # 1. 采样空间点
    t_vals = torch.linspace(0., 1., n_samples, device=DEVICE)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand(rays_o.shape[0], n_samples) 
    
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
    
    # 2. 准备网络输入
    n_rays = rays_o.shape[0]
    
    # A. 编码时序状态 (1, Seq, Dim) -> (1, Hidden)
    state = model.encode_temporal(action_window)
    # 扩展: (N_rays, Hidden)
    state_input = state.expand(n_rays, -1)
    
    # B. 准备当前动作 (1, Dim) -> (N_rays, Dim)
    curr_act_input = current_act.expand(n_rays, -1)
    
    # C. 准备几何点
    pts_input = pts 
    
    # 3. 空间解码 (分块处理)
    chunk_size = 4096 * 4
    all_rgb = []
    
    for i in range(0, n_rays, chunk_size):
        pts_chunk = pts_input[i : i+chunk_size]
        state_chunk = state_input[i : i+chunk_size]
        act_chunk = curr_act_input[i : i+chunk_size]
        
        # [注意] model_seq_skip 的 decode_spatial 会自动处理 samples 维度的广播
        # 输入: (Batch, N_samples, Dim)
        raw_out = model.decode_spatial(pts_chunk, state_chunk, act_chunk)
        
        rgb_chunk, _ = OM_rendering(raw_out)
        all_rgb.append(rgb_chunk)
        
    return torch.cat(all_rgb, dim=0)

# ==========================================
#  主可视化逻辑
# ==========================================

def visualize(seq_file_path, model_dir, output_gif="seq_vis_result.gif"):
    # 1. 加载配置与归一化系数
    norm_path = os.path.join(model_dir, "action_norm_factor.txt")
    if os.path.exists(norm_path):
        norm_factor = float(np.loadtxt(norm_path))
        print(f"Loaded Normalization Factor: {norm_factor}")
    else:
        norm_factor = 1.0
        print("Warning: No norm factor found, using 1.0")

    # 2. 加载数据
    print(f"Loading sequence data: {seq_file_path}")
    data = np.load(seq_file_path)
    images_gt = data['images'] # (T, H, W)
    actions_raw = data['actions'] # (T, D)
    
    H, W = images_gt.shape[1], images_gt.shape[2]
    T, action_dim = actions_raw.shape
    focal = float(data.get('focal', 130.0))
    
    # 3. 加载模型
    SEQ_LEN = 20 # 必须与训练一致
    model = RecurrentFBV_SM_Skip(
        action_dim=action_dim, 
        seq_len=SEQ_LEN, 
        hidden_dim=256
    ).to(DEVICE)
    
    # 支持加载 best_seq_model.pt 或 best_model.pt
    weight_path = os.path.join(model_dir, "model/best_seq_model.pt")
    if not os.path.exists(weight_path):
        # 尝试备用名称
        weight_path = os.path.join(model_dir, "model/best_model.pt")
        
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found in {model_dir}/model/")
    
    print(f"Loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 4. 预计算射线
    CAM_EYE = (1.5, 0.0, 0.5)
    CAM_CENTER = (0.0, 0.0, 0.25)
    CAM_UP = (0.0, 0.0, 1.0)
    
    rays_o, rays_d = get_rays_from_camera_params(
        H, W, torch.tensor(focal).to(DEVICE), 
        CAM_EYE, CAM_CENTER, CAM_UP
    )
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    # 5. 准备绘图
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 初始化内容
    im_gt = ax1.imshow(images_gt[0], cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Ground Truth")
    ax1.axis('off')
    
    im_pred = ax3.imshow(np.zeros_like(images_gt[0]), cmap='gray', vmin=0, vmax=1)
    ax3.set_title("FFKSM Prediction (Seq)")
    ax3.axis('off')
    
    # 动作波形图
    lines = []
    colors = ['r', 'b', 'g', 'c']
    for d in range(action_dim):
        line, = ax2.plot(np.arange(T), actions_raw[:, d], color=colors[d%4], alpha=0.6, label=f'Motor {d}')
        lines.append(line)
    
    vline = ax2.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax2.set_xlim(0, T)
    ax2.set_ylim(np.min(actions_raw), np.max(actions_raw))
    ax2.set_title("Driving Signals")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.set_xlabel("Time Step")

    # 6. 动画更新函数
    # 预处理归一化的动作数据
    actions_norm = actions_raw / norm_factor
    actions_tensor = torch.from_numpy(actions_norm).float().to(DEVICE)
    
    # --- [新增] 初始化平滑变量 ---
    prev_pred_img = None
    # 平滑系数 alpha (0 < alpha <= 1)
    # 0.1: 非常平滑但有明显拖影/延迟
    # 0.3: 推荐值，去抖动效果好且延迟低
    # 1.0: 无平滑 (原样)
    SMOOTH_FACTOR = 0.3
    
    
    def update(frame_idx):
        
        nonlocal prev_pred_img # 允许修改外部变量
        
        # A. 准备数据窗口 (Sliding Window)
        start_idx = frame_idx - SEQ_LEN + 1
        end_idx = frame_idx + 1
        
        if start_idx >= 0:
            input_seq = actions_tensor[start_idx:end_idx] # (Seq_Len, D)
        else:
            # Padding
            valid_part = actions_tensor[0:end_idx]
            pad_len = SEQ_LEN - len(valid_part)
            padding = torch.zeros((pad_len, action_dim), device=DEVICE)
            input_seq = torch.cat([padding, valid_part], dim=0)
            
        # 增加 Batch 维度 (1, Seq_Len, D)
        input_seq = input_seq.unsqueeze(0)
        
        # 当前动作 (用于直连) (1, D)
        current_act = actions_tensor[frame_idx].unsqueeze(0)
        
        # B. 运行推理
        with torch.no_grad():
            pred_flat = run_inference_frame(model, input_seq, current_act, rays_o, rays_d)
            pred_img = pred_flat.reshape(H, W).cpu().numpy()
            
        # --- [新增] 结果层面的时序平滑逻辑 (EMA Filter) ---
        if prev_pred_img is not None:
            # 公式: Display = alpha * Current + (1-alpha) * Last
            pred_img = SMOOTH_FACTOR * pred_img + (1 - SMOOTH_FACTOR) * prev_pred_img
            
        # 更新历史帧
        prev_pred_img = pred_img
        # --------------------------------------------------
        
        # C. 更新画面
        im_gt.set_data(images_gt[frame_idx])
        im_pred.set_data(pred_img)
        vline.set_xdata([frame_idx, frame_idx])
        
        ax3.set_title(f"Pred (Frame {frame_idx})")
        return im_gt, im_pred, vline

    # 7. 生成 GIF
    step_size = 5 
    frames = range(0, T, step_size)
    
    print(f"Generating GIF ({len(frames)} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=tqdm(frames), blit=False)
    
    save_path = os.path.join(current_dir, output_gif)
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()
    print(f"✅ Visualization saved to: {save_path}")

if __name__ == "__main__":
    # --- 用户配置区 ---
    
    # 指向训练日志目录 (请根据实际情况修改 experiment_X)
    LOG_DIR = os.path.join(project_root, "train_log_seq_vis/experiment_2") 
    
    # 指向数据目录
    DATA_DIR = os.path.join(project_root, "data/sequence_data")
    
    # 自动寻找最新的数据文件
    all_seqs = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")), key=os.path.getmtime)
    if not all_seqs:
        print(f"Error: No data found in {DATA_DIR}")
    else:
        # 默认使用最后一段数据进行演示
        TEST_SEQ = all_seqs[-1] 
        print(f"Using sequence file: {TEST_SEQ}")
        
        visualize(TEST_SEQ, LOG_DIR, output_gif="vis_seq_final.gif")
        