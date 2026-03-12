import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- 1. 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import FBV_SM, PositionalEncoder

# ==========================================
#  用户配置区域
# ==========================================
MODEL_PATH = os.path.join(parent_dir, "train_log_soft/experiment_3/model/model_34000.pt")
# 自动寻找最新的数据文件，或者您手动指定
DATA_DIR = os.path.join(parent_dir, "data/sequence_data")
files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.npz')], key=os.path.getmtime)
DATA_PATH = files[-1] if files else "data/sequence_data/seq_0_1764565398.npz"

OUTPUT_GIF = os.path.join(current_dir, "prediction_comparison.gif")

# 参数 (必须与训练一致)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 100
NEAR = 1.2
FAR = 1.8
N_SAMPLES = 192
FOV = 30

# ==========================================
#  核心函数 (修复版)
# ==========================================

def get_rays_simple(H, W, focal, c2w):
    """根据相机位姿矩阵生成像素射线。"""
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=DEVICE),
        torch.arange(H, dtype=torch.float32, device=DEVICE),
        indexing='ij'
    )
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def Robust_Mask_Rendering(raw, z_vals):
    """将网络原始输出聚合为二维掩码图。"""
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    last_dist = dists[..., -1:]
    dists = torch.cat([dists, last_dist], -1)
    
    sigma = torch.nn.functional.softplus(raw[..., 1]) 
    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=DEVICE), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    acc_map = torch.sum(weights, -1)
    return acc_map

def sample_stratified(rays_o, rays_d, near, far, n_samples):
    """沿射线进行分层采样。"""
    t_vals = torch.linspace(0., 1., n_samples, device=DEVICE)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def run_inference(model, action, rays_o, rays_d):
    """执行单帧推理（含分块前向与结果重塑）。

    Args:
        model: 已加载权重模型。
        action: 当前时刻动作向量。
        rays_o, rays_d: 射线起点与方向。

    Returns:
        img_pred: 预测图像，形状 (H, W)。
    """
    H, W = rays_o.shape[:2]
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    pts, z_vals = sample_stratified(rays_o_flat, rays_d_flat, NEAR, FAR, N_SAMPLES)
    
    action_expanded = action.view(1, 1, -1).expand(pts.shape[0], pts.shape[1], -1)
    model_input = torch.cat((pts, action_expanded), dim=-1)
    
    # [关键] 展平输入
    input_flat = model_input.reshape(-1, model_input.shape[-1])
    
    chunk_size = 32768
    results = []
    for i in range(0, input_flat.shape[0], chunk_size):
        chunk = input_flat[i:i+chunk_size]
        with torch.no_grad():
            pred = model(chunk)
        results.append(pred)
    
    raw_flat = torch.cat(results, dim=0)
    # [关键] 恢复维度
    raw = raw_flat.reshape(pts.shape[0], pts.shape[1], raw_flat.shape[-1])
    
    img_pred_flat = Robust_Mask_Rendering(raw, z_vals)
    img_pred = img_pred_flat.reshape(H, W)
    
    return img_pred

# ==========================================
#  主程序
# ==========================================

def main():
    """加载数据与模型，生成真实/动作/预测三联动可视化 GIF。"""
    print(f"Loading Model: {MODEL_PATH}")
    print(f"Loading Data: {DATA_PATH}")

    # 1. 数据准备
    data = np.load(DATA_PATH)
    images_gt = data['images']
    actions = data['actions']
    
    # Focal / FOV
    if 'focal' in data and data['focal'].item() > 1:
        focal_val = data['focal'].item()
    else:
        focal_val = 0.5 * IMG_SIZE / np.tan(0.5 * FOV * np.pi / 180)

    # 动作归一化
    norm_file = os.path.join(os.path.dirname(MODEL_PATH), "../action_norm_factor.txt")
    if os.path.exists(norm_file):
        action_norm = np.loadtxt(norm_file).item()
    else:
        action_norm = np.max(np.abs(actions)) if np.max(np.abs(actions)) > 0 else 1.0
    print(f"Action Norm Factor: {action_norm}")
    actions_normed = actions / action_norm

    # 2. 模型准备
    d_input = 3 + actions.shape[1]
    encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)
    model = FBV_SM(encoder=encoder, d_input=d_input, d_filter=128, output_size=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. 射线准备
    def look_at_matrix(eye, center, up):
        z_axis = eye - center; z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
        mat = np.eye(4)
        mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = x_axis, y_axis, z_axis, eye
        return torch.tensor(mat, dtype=torch.float32, device=DEVICE)

    c2w = look_at_matrix(np.array([1.5, 0., 0.5]), np.array([0., 0., 0.25]), np.array([0., 0., 1.]))
    rays_o, rays_d = get_rays_simple(IMG_SIZE, IMG_SIZE, focal_val, c2w)

    # 4. 设置 Matplotlib 画布 (模仿 save_gif.py 风格)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # [左图] 真实
    im_real = ax1.imshow(images_gt[0], cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Real Observation")
    ax1.axis('off')

    # [中图] 动作曲线 (参考 save_gif.py)
    ax2.set_title("Control Input (Time History)")
    # 绘制所有电机的完整轨迹背景
    colors = ['r', 'b', 'g', 'c'] # 不同电机不同颜色
    for i in range(actions.shape[1]):
        ax2.plot(actions[:, i], color=colors[i % len(colors)], alpha=0.5, label=f"Motor {i}")
    
    # 竖线指示器
    vline = ax2.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    
    ax2.set_xlim(0, len(actions))
    # 稍微放宽一点 Y 轴范围以便观察
    y_min, y_max = np.min(actions), np.max(actions)
    margin = (y_max - y_min) * 0.1
    ax2.set_ylim(y_min - margin, y_max + margin)
    ax2.legend(loc='upper right', fontsize='small')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Torque")

    # [右图] 预测
    # 先生成第一帧的空数据
    dummy_pred = np.zeros((IMG_SIZE, IMG_SIZE))
    im_pred = ax3.imshow(dummy_pred, cmap='gray', vmin=0, vmax=1)
    ax3.set_title("Self-Model Prediction")
    ax3.axis('off')

    # 5. 动画更新函数
    def update(frame_idx):
        # 1. 更新真实图
        im_real.set_data(images_gt[frame_idx])
        
        # 2. 更新竖线位置
        # axvline 返回的是 Line2D，设置 xdata 需要两个点 [x, x]
        vline.set_xdata([frame_idx, frame_idx])
        
        # 3. 运行推理更新预测图
        act_tensor = torch.tensor(actions_normed[frame_idx], dtype=torch.float32, device=DEVICE)
        pred_img = run_inference(model, act_tensor, rays_o, rays_d)
        im_pred.set_data(pred_img.cpu().numpy())
        
        # 更新标题显示进度
        fig.suptitle(f"Frame {frame_idx}/{len(images_gt)}", fontsize=12)
        
        return im_real, vline, im_pred

    # 6. 生成 GIF
    # 降采样：每隔 5 帧画一次，防止生成太慢
    step_size = 5 
    frames = range(0, len(images_gt), step_size)
    
    print(f"Starting GIF generation for {len(frames)} frames...")
    
    # blit=False 兼容性更好，虽然稍慢一点
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    
    print(f"Saving to {OUTPUT_GIF} ...")
    ani.save(OUTPUT_GIF, writer='pillow', fps=10)
    print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()