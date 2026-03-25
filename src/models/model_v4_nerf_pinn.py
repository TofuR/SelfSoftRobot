import torch
import torch.nn as nn
from .layers import PositionalEncoder, TemporalLSTMEncoder, MLPDecoder

class NeRF_PINN(nn.Module):
    """
    Model V4: Physics-Informed Neural Field (NeRF-based)
    
    核心思想:
    1. Static-Dynamic Decoupling: 
       - 静态流: 学习 (x,y,z) -> Base Feature (机器人的固有形状)
       - 动态流: 学习 Action Sequence -> Physics Latent (物理状态)
    2. Physics Constraints: 
       - 对 Physics Latent 施加二阶导数平滑约束，消除 3D 抖动。
    """
    def __init__(self, action_dim, seq_len, hidden_dim=128):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        # --- A. 静态结构记忆 (Static Structure Memory) ---
        # 这是一个只与坐标有关的 MLP，用于记忆机器人的"本体"
        # 它不接受动作输入，只接受坐标，保证了对物体结构的"记忆"
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=10, log_space=True)
        pos_dim = 3 * (1 + 2 * 10)
        
        self.static_net = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
            # 输出 128 维的静态几何特征
        )
        
        # --- B. 动态物理引擎 (Dynamic Physics Engine) ---
        # LSTM 处理动作序列，输出物理隐状态
        self.temporal = TemporalLSTMEncoder(
            input_dim=action_dim,
            hidden_dim=hidden_dim, # 128
            num_layers=2,
            batch_first=True
        )
        
        # --- C. 融合解码器 (Fusion Decoder) ---
        # 输入: 静态特征(128) + 物理状态(128) + 当前动作(2)
        # 这种设计强迫模型：用静态特征画出形状，用物理状态决定形变/位移
        input_total_dim = 128 + hidden_dim + action_dim
        
        self.decoder = MLPDecoder(
            input_dim=input_total_dim, 
            d_filter=128, 
            output_size=2 # [Visibility, Density]
        )

    def get_physics_state(self, action_seq):
        """
        获取物理隐状态序列 (用于计算 Loss)
        Returns: (Batch, Seq_Len, Hidden)
        """
        return self.temporal(action_seq, return_sequence=True)

    def forward_rendering(self, points, physics_state, current_action):
        """
        渲染查询接口 (用于 Ray Marching)
        points: (N_rays, N_samples, 3)
        physics_state: (N_rays, Hidden) - 当前时刻的物理状态
        current_action: (N_rays, Action_Dim)
        """
        # 1. 计算静态几何特征 (只依赖坐标)
        # (N_rays, N_samples, 3) -> (N_rays, N_samples, Pos_Dim)
        x_embed = self.pos_encoder(points)
        static_feature = self.static_net(x_embed) 
        
        # 2. 扩展动态特征以匹配采样点
        n_samples = points.shape[1]
        # (N_rays, Hidden) -> (N_rays, N_samples, Hidden)
        dynamic_feature = physics_state.unsqueeze(1).expand(-1, n_samples, -1)
        # (N_rays, Action_Dim) -> (N_rays, N_samples, Action_Dim)
        action_feature = current_action.unsqueeze(1).expand(-1, n_samples, -1)
        
        # 3. 融合 (Static + Dynamic + Action)
        latent = torch.cat([static_feature, dynamic_feature, action_feature], dim=-1)
        
        # 4. 解码得到密度
        output = self.decoder(latent)
        return output

    def compute_smoothness_loss(self, latent_seq):
        """
        计算物理平滑 Loss (PINN思想)
        对 LSTM 输出的隐变量求二阶导数 (加速度)，惩罚突变
        latent_seq: (Batch, Seq, Hidden)
        """
        if latent_seq.shape[1] < 3:
            return torch.tensor(0.0, device=latent_seq.device)
        
        # 一阶差分 (速度)
        vel = latent_seq[:, 1:] - latent_seq[:, :-1]
        # 二阶差分 (加速度)
        acc = vel[:, 1:] - vel[:, :-1]
        
        return torch.mean(acc ** 2)
    
    