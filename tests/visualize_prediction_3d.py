import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# --- 1. 路径与环境配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入现有模块 (直接复用，不重写)
from src.models import FBV_SM, PositionalEncoder
from elastica_env import ContinuousSoftArmEnv

# ==========================================
#  配置区域
# ==========================================
# 模型路径
MODEL_PATH = os.path.join(parent_dir, "train_log_soft/experiment_3/model/model_34000.pt")

# 数据路径 (自动寻找最新)
DATA_DIR = os.path.join(parent_dir, "data/sequence_data")
files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.npz')], key=os.path.getmtime)
DATA_PATH = files[-1] if files else "data/sequence_data/seq_0_1764565398.npz"

OUTPUT_GIF = os.path.join(current_dir, "prediction_3d_comparison.gif")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3D 采样空间配置 (只在软体臂活动的范围内采样，节省计算)
# 软体臂长 0.5, 半径 0.015
X_RANGE = (-0.1, 0.1)
Y_RANGE = (-0.1, 0.1)
Z_RANGE = (0.0, 0.6)
GRID_RES = 40  # Z轴分辨率 (X/Y轴会根据比例自动调整)

# 物理仿真参数 (必须与 collect_sequence.py 一致)
DT = 1e-4
RECORD_INTERVAL = 50 # 每50个物理步对应1个数据帧

# ==========================================
#  核心逻辑
# ==========================================

def run_physics_reconstruction(actions_seq):
    """重放动作序列并重建真实 3D 杆体轨迹。"""
    print(f">>> 正在重构物理仿真 (共 {len(actions_seq)} 帧)...")
    env = ContinuousSoftArmEnv(dt=DT)
    
    gt_positions = []
    
    # 进度条
    for i in tqdm(range(len(actions_seq))):
        target_action = actions_seq[i]
        
        # 应用动作
        env.set_action(target_action)
        
        # 推进物理时间 (模拟两帧之间的时间间隔)
        env.step(steps=RECORD_INTERVAL)
        
        # 记录真实的 Rod 节点坐标 (3, N_nodes)
        # position_collection 是 PyElastica 的内部数据
        rod = env.simulation[0] # 获取第一个系统(Rod)
        pos = rod.position_collection.copy() 
        gt_positions.append(pos)
        
    return gt_positions

def generate_query_grid():
    """生成模型查询用三维网格点。"""
    # 根据比例计算 X/Y 的分辨率
    z_len = Z_RANGE[1] - Z_RANGE[0]
    x_len = X_RANGE[1] - X_RANGE[0]
    y_len = Y_RANGE[1] - Y_RANGE[0]
    
    z_res = GRID_RES
    x_res = int(z_res * (x_len / z_len)) + 1
    y_res = int(z_res * (y_len / z_len)) + 1
    
    # 稍微增加 X/Y 分辨率以捕捉细细的杆子
    x_res = max(x_res, 15)
    y_res = max(y_res, 15)
    
    xs = torch.linspace(X_RANGE[0], X_RANGE[1], x_res, device=DEVICE)
    ys = torch.linspace(Y_RANGE[0], Y_RANGE[1], y_res, device=DEVICE)
    zs = torch.linspace(Z_RANGE[0], Z_RANGE[1], z_res, device=DEVICE)
    
    # 生成网格 (Meshgrid)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
    
    # 展平为 (N_points, 3)
    flat_pts = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    return flat_pts

def get_model_prediction_cloud(model, action, grid_pts, threshold=2.0):
    """给定动作查询体场密度并返回阈值点云。"""
    n_points = grid_pts.shape[0]
    
    # 扩展 Action: (N_points, Action_Dim)
    action_expanded = action.view(1, -1).expand(n_points, -1)
    
    # 构造输入: (N_points, 3+Action_Dim)
    model_input = torch.cat((grid_pts, action_expanded), dim=-1)
    
    # 推理 (分块处理防止显存溢出)
    chunk_size = 32768
    densities = []
    
    with torch.no_grad():
        for i in range(0, n_points, chunk_size):
            chunk = model_input[i:i+chunk_size]
            pred = model(chunk) # Output: (N, 2) -> [Color, Density_Logits]
            
            # 使用 Softplus 获取密度
            dens = torch.nn.functional.softplus(pred[:, 1])
            densities.append(dens)
            
    densities = torch.cat(densities, dim=0)
    
    # 筛选高密度点 (Threshold)
    # 这里的 threshold 取决于 Softplus 的偏置，通常 > 0.1 即可看到形状
    mask = densities > threshold
    
    valid_points = grid_pts[mask]
    valid_densities = densities[mask]
    
    return valid_points.cpu().numpy(), valid_densities.cpu().numpy()

# ==========================================
#  主程序
# ==========================================

def main():
    """执行 3D 真实/预测对比并导出动画 GIF。"""
    # 1. 加载数据
    print(f"Loading Data: {DATA_PATH}")
    data = np.load(DATA_PATH)
    actions = data['actions']
    
    # 动作归一化 (读取训练时的参数)
    norm_file = os.path.join(os.path.dirname(MODEL_PATH), "../action_norm_factor.txt")
    if os.path.exists(norm_file):
        action_norm = np.loadtxt(norm_file).item()
    else:
        action_norm = np.max(np.abs(actions)) if np.max(np.abs(actions)) > 0 else 1.0
    print(f"Action Norm Factor: {action_norm}")
    actions_normed = actions / action_norm

    # 2. 物理重构 (Ground Truth)
    # 为了演示速度，我们只取前 100 帧或进行降采样
    SKIP_FRAME = 2
    frames_indices = range(0, len(actions), SKIP_FRAME)
    # 注意：物理重构必须连续跑，不能跳帧，所以我们先跑完物理，再采样
    gt_positions_full = run_physics_reconstruction(actions) # 跑全量
    gt_positions_sampled = [gt_positions_full[i] for i in frames_indices]
    actions_sampled = actions[frames_indices]
    actions_normed_sampled = actions_normed[frames_indices]

    # 3. 模型准备
    print(f"Loading Model: {MODEL_PATH}")
    d_input = 3 + actions.shape[1]
    encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)
    model = FBV_SM(encoder=encoder, d_input=d_input, d_filter=128, output_size=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 准备查询网格
    query_grid = generate_query_grid()
    print(f"Sampling Grid Size: {query_grid.shape[0]} points")

    # 4. 设置画布
    fig = plt.figure(figsize=(15, 6))
    
    # Left: Real 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("Real 3D Shape (Physics Engine)")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim(X_RANGE); ax1.set_ylim(Y_RANGE); ax1.set_zlim(Z_RANGE)
    line_gt, = ax1.plot([], [], [], 'b-', linewidth=4, label='Real')
    
    # Middle: Control
    ax2 = fig.add_subplot(132)
    ax2.set_title("Control Input")
    colors = ['r', 'b', 'g', 'c']
    # 绘制全量动作
    x_axis_full = np.arange(len(actions))
    for i in range(actions.shape[1]):
        ax2.plot(x_axis_full, actions[:, i], color=colors[i], alpha=0.5, label=f"Motor {i}")
    
    vline = ax2.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    ax2.set_xlim(0, len(actions))
    ax2.legend()
    
    # Right: Predicted 3D (Point Cloud)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("Predicted 3D Shape (Neural Model)")
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_xlim(X_RANGE); ax3.set_ylim(Y_RANGE); ax3.set_zlim(Z_RANGE)
    # 散点图占位符
    scatter_pred = ax3.scatter([], [], [], c='r', s=10, marker='o', label='Predicted')

    def update(frame_idx):
        # 这里的 frame_idx 是 sampled list 的索引
        real_idx = frames_indices[frame_idx]
        
        # 1. Update Real 3D
        pos = gt_positions_sampled[frame_idx] # (3, N)
        line_gt.set_data(pos[0], pos[1])
        line_gt.set_3d_properties(pos[2])
        
        # 2. Update Control Line
        vline.set_xdata([real_idx, real_idx])
        
        # 3. Update Predicted 3D
        act_tensor = torch.tensor(actions_normed_sampled[frame_idx], dtype=torch.float32, device=DEVICE)
        
        # 获取模型预测的点云
        # 阈值 softplus(0) approx 0.69. 
        # 如果背景是0 (bias -5 -> softplus ~0), 物体是 >0. 
        # 建议尝试 0.1 或 0.5。如果全空，调低；如果全满，调高。
        pts, dens = get_model_prediction_cloud(model, act_tensor, query_grid, threshold=0.1)
        
        ax3.clear()
        ax3.set_title("Predicted 3D Shape (Neural Model)")
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        ax3.set_xlim(X_RANGE); ax3.set_ylim(Y_RANGE); ax3.set_zlim(Z_RANGE)
        
        if len(pts) > 0:
            # 颜色随密度变化，或者纯红
            ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dens, cmap='Reds', s=5, alpha=0.5)
        else:
            ax3.text(0, 0, 0.3, "No Density Detected", ha='center')

        fig.suptitle(f"Frame {real_idx}/{len(actions)}", fontsize=14)
        return line_gt, vline

    print("Generating 3D Animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_indices), interval=100) # interval ms
    
    print(f"Saving GIF to {OUTPUT_GIF} ...")
    ani.save(OUTPUT_GIF, writer='pillow', fps=10)
    print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()
        