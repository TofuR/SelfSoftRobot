import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from tqdm import tqdm

# 导入您的模块
from src.models import model_v2
from elastica_env import ContinuousSoftArmEnv, SimpleDistributedTorque

# --- 全局设置 ---
CUDA_DEVICE = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on device: {device}")

class ResultVisualizer:
    """序列结果可视化器：对比物理仿真真实形态与模型预测点云。"""
    def __init__(self, log_dir, seq_file):
        self.log_dir = log_dir
        
        # 1. 加载归一化系数
        norm_path = os.path.join(log_dir, "action_norm_factor.txt")
        if os.path.exists(norm_path):
            self.norm_factor = float(np.loadtxt(norm_path))
            print(f"Loaded Norm Factor: {self.norm_factor}")
        else:
            self.norm_factor = 1.0
            print("Warning: Using default norm factor 1.0")

        # 2. 加载数据序列 (只取动作)
        data = np.load(seq_file)
        self.actions_raw = data['actions'] # 原始物理参数
        self.seq_len = 20 # 必须与训练时一致
        self.action_dim = self.actions_raw.shape[1]
        
        # 3. 加载模型
        self.model = model_v2(
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            hidden_dim=256
        ).to(device)
        
        weight_path = os.path.join(log_dir, "model/best_seq_model.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model not found: {weight_path}")
            
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()
        
        # 4. 准备 3D 查询网格 (用于生成点云)
        self.grid_res = 30 
        x = np.linspace(-0.3, 0.3, self.grid_res)
        y = np.linspace(-0.3, 0.3, self.grid_res)
        z = np.linspace(0.0, 0.6, self.grid_res)
        
        gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1)
        self.grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=device)
        
    def run_inference_sequence(self):
        """执行整段序列推理并收集可视化所需数据。

        Returns:
            gt_positions: 物理仿真得到的真实杆体坐标序列。
            pred_clouds: 模型预测点云序列。
            actions: 对应动作序列。
        """
        env = ContinuousSoftArmEnv(dt=1e-4)
        
        gt_positions = []   
        pred_clouds = []
        
        # 历史动作 buffer
        history_buffer = torch.zeros((1, self.seq_len, self.action_dim), device=device)
        
        print("Running Simulation & Inference...")
        sim_steps_per_action = 500 
        
        total_actions = len(self.actions_raw)
        demo_length = min(100, total_actions) 
        
        for i in tqdm(range(demo_length)):
            target_action = self.actions_raw[i]
            
            # --- A. 物理仿真 (GT) ---
            env.set_action(target_action)
            for _ in range(sim_steps_per_action):
                env.step()
            
            rod = env.simulation[0]
            pos = rod.position_collection.copy().T
            gt_positions.append(pos)
            
            # --- B. 模型推理 (Pred) ---
            # 1. 更新动作历史
            act_norm = target_action / self.norm_factor
            print(f"Raw action: {target_action}, Norm factor: {self.norm_factor}, Normalized action: {act_norm}")
            act_tensor = torch.tensor(act_norm, dtype=torch.float32, device=device).view(1, 1, -1)
            history_buffer = torch.cat([history_buffer[:, 1:, :], act_tensor], dim=1)
            
            # 2. 预测点云
            with torch.no_grad():
                # 编码时序状态 -> (1, Hidden)
                state = self.model.encode_temporal(history_buffer) 
                
                # 准备输入
                # Grid: (1, N_pts, 3)
                pts_input = self.grid_tensor.unsqueeze(0) 
                
                # [关键修复] 
                # 不要手动扩展 state 和 action 的维度！
                # decode_spatial 内部会自动把 (1, Hidden) 扩展为 (1, N_pts, Hidden)
                
                state_input = state  # 保持 (1, 256)
                curr_act_input = history_buffer[:, -1, :] # 取最后一步动作，保持 (1, 2)
                
                # 推理
                raw_out = self.model.decode_spatial(pts_input, state_input, curr_act_input)
                
                # 密度筛选
                density = 1.0 - torch.exp(-torch.nn.functional.relu(raw_out[0, :, 0])) 
                
                threshold = 0.75
                mask = density > threshold
                
                valid_points = self.grid_points[mask.cpu().numpy()]
                pred_clouds.append(valid_points)

        return gt_positions, pred_clouds, self.actions_raw[:demo_length]

    def save_gif(self, gt_pos, pred_cld, actions, filename="test_3d.gif"):
        """将真实形态、动作曲线和预测点云合成为 GIF。"""
        print("Generating 3D GIF...")
        
        fig = plt.figure(figsize=(15, 5))
        
        # 1. 左图：真实形态
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title("Ground Truth (Simulation)")
        ax1.set_xlim(-0.3, 0.3); ax1.set_ylim(-0.3, 0.3); ax1.set_zlim(0, 0.6)
        line_gt, = ax1.plot([], [], [], 'b-', linewidth=4, label='Soft Arm')
        
        # 2. 中图：驱动参数
        ax2 = fig.add_subplot(132)
        ax2.set_title("Driving Actions")
        lines_act = []
        colors = ['r', 'g']
        for d in range(actions.shape[1]):
            l, = ax2.plot([], [], color=colors[d], label=f'Motor {d}')
            lines_act.append(l)
        ax2.set_xlim(0, len(actions))
        ax2.set_ylim(np.min(actions), np.max(actions))
        vline = ax2.axvline(x=0, color='k', linestyle='--')
        ax2.legend()
        
        # 3. 右图：预测点云
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title("Model Prediction (Point Cloud)")
        ax3.set_xlim(-0.3, 0.3); ax3.set_ylim(-0.3, 0.3); ax3.set_zlim(0, 0.6)
        scatter_pred = ax3.scatter([], [], [], c='r', s=5, alpha=0.6)
        
        def update(frame):
            # 更新 GT
            pos = gt_pos[frame]
            line_gt.set_data(pos[:, 0], pos[:, 1])
            line_gt.set_3d_properties(pos[:, 2])
            
            # 更新 Action
            for d, l in enumerate(lines_act):
                l.set_data(np.arange(len(actions)), actions[:, d]) 
            vline.set_xdata([frame, frame])
            
            # 更新 Pred
            cloud = pred_cld[frame]
            if len(cloud) > 0:
                scatter_pred._offsets3d = (cloud[:, 0], cloud[:, 1], cloud[:, 2])
            else:
                scatter_pred._offsets3d = ([], [], [])
                
            return line_gt, vline, scatter_pred

        ani = animation.FuncAnimation(fig, update, frames=len(gt_pos), interval=100)
        ani.save(filename, writer='pillow', fps=10)
        plt.close()
        print(f"GIF Saved to {filename}")

if __name__ == "__main__":
    LOG_DIR = os.path.join("train_log", "train_log_seq_vis", "experiment_2") 
    data_dir = "data/sequence_data"
    seq_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not seq_files:
        raise ValueError("No sequence data found!")
    test_file = seq_files[-1] 
    
    vis = ResultVisualizer(LOG_DIR, test_file)
    gt, pred, acts = vis.run_inference_sequence()
    vis.save_gif(gt, pred, acts, filename="vis_3d_comparison.gif")
    
    
