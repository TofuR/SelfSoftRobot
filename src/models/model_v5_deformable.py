import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PositionalEncoder, TemporalLSTMEncoder

class DeformableSoftRobotModel(nn.Module):
    """
    Model V5: Deformable Physics-Informed Neural Field
    
    核心机制: Canonical Space (本体) + Deformation Field (变形)
    
    1. Canonical Net: F_static(x) -> Density
       - 只记忆机器人静止时的形状 (Rest Pose)。
       - 甚至可以在推理时锁死参数，保证结构不崩坏。
       
    2. Deformation Net: F_deform(x, physics_state) -> dx
       - 学习空间点的"流动"。
       - 真正的物理过程: x_observed = x_canonical + deformation
       - 逆向查询: x_canonical = x_observed + backward_deformation
    """
    def __init__(self, action_dim, seq_len, hidden_dim=128):
        super().__init__()
        
        # --- A. 物理引擎 (Physics Engine) ---
        # LSTM 处理动作序列，输出物理隐状态 (h_t)
        self.temporal = TemporalLSTMEncoder(
            input_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # --- B. 变形网络 (Deformation Network) ---
        # 输入: 观察坐标 x + 物理状态 h_t
        # 输出: 坐标偏移量 (dx, dy, dz)
        # 这里的 PosEnc 频率低一点，因为变形场通常是低频平滑的
        self.deform_pos_enc = PositionalEncoder(d_input=3, n_freqs=6, log_space=True)
        deform_input_dim = 3 * (1 + 2 * 6) + hidden_dim
        
        self.deform_net = nn.Sequential(
            nn.Linear(deform_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 输出 dx, dy, dz
        )
        # 初始化为0，让训练初期模型认为没有变形 (Identity)
        nn.init.constant_(self.deform_net[-1].weight, 0)
        nn.init.constant_(self.deform_net[-1].bias, 0)

        # --- C. 本体网络 (Canonical Network) ---
        # 输入: 正则坐标 x_can
        # 输出: 密度 Density (不输出颜色，因为我们只关心结构)
        # 这里的 PosEnc 频率高，为了捕捉精细结构
        self.canon_pos_enc = PositionalEncoder(d_input=3, n_freqs=10, log_space=True)
        canon_input_dim = 3 * (1 + 2 * 10)
        
        self.canonical_net = nn.Sequential(
            nn.Linear(canon_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # 只输出密度 sigma
            nn.ReLU() # 密度必须 >= 0
        )

    def get_physics_state(self, action_seq):
        """获取整个序列的物理状态，用于计算时序平滑 Loss"""
        output, _ = self.temporal(action_seq, return_all=True)
        return output

    def query_field(self, x_observed, physics_state):
        """
        核心前向过程: 观察点 -> 变形 -> 正则点 -> 密度
        x_observed: (Batch, N, 3)
        physics_state: (Batch, Hidden)
        """
        batch_size, n_points, _ = x_observed.shape
        
        # 1. 扩展物理状态以匹配点数
        # (Batch, 1, Hidden) -> (Batch, N, Hidden)
        state_exp = physics_state.unsqueeze(1).expand(-1, n_points, -1)
        
        # 2. 计算变形量 (Backward Deformation)
        # 我们想知道: "现在看到的这个点 x，在静止状态下原本在哪?"
        x_encoded = self.deform_pos_enc(x_observed)
        deform_input = torch.cat([x_encoded, state_exp], dim=-1)
        
        # offset: (Batch, N, 3)
        offset = self.deform_net(deform_input)
        
        # 3. 映射回正则空间 (Canonical Space)
        x_canonical = x_observed + offset
        
        # 4. 查询本体形状
        x_can_encoded = self.canon_pos_enc(x_canonical)
        density = self.canonical_net(x_can_encoded) # (Batch, N, 1)
        
        # 我们同时返回 offset，用于计算弹性正则化 Loss (惩罚过大变形)
        return density, offset

    def forward_rendering(self, points, physics_state, current_action=None):
        """兼容旧的渲染接口"""
        # 注意：Deformable 模型不需要 current_action 直接进入 Decoder，
        # 因为所有动作影响都通过 physics_state 进入 deformation net 了
        density, _ = self.query_field(points, physics_state)
        
        # 为了兼容 render 函数的输出格式 (raw)，我们拼接一个假的 RGB 通道
        # raw: [..., RGB(3) + Density(1)]
        rgb_fake = torch.sigmoid(density).repeat(1, 1, 3) # 简单的白色显示
        raw = torch.cat([rgb_fake, density], dim=-1)
        
        return raw