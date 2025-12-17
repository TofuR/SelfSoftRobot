import torch
import torch.nn as nn
from model import PositionalEncoder

class RecurrentFBV_SM(nn.Module):
    """
    [优化版] 支持时序输入的自模型
    将 Encoder (LSTM) 和 Decoder (MLP) 分离，极大节省显存和计算量。
    """
    def __init__(self, 
                 action_dim, 
                 seq_len, 
                 hidden_dim=128, 
                 d_filter=128, 
                 output_size=2):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 1. 时序编码器
        self.lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 2. 坐标编码器
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=10, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * 10) 
        
        # 3. 空间解码器
        input_total_dim = pos_enc_dim + hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_total_dim, d_filter * 2),
            nn.ReLU(),
            nn.Linear(d_filter * 2, d_filter * 2),
            nn.ReLU(),
            nn.Linear(d_filter * 2, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter, d_filter // 2),
            nn.ReLU(),
            nn.Linear(d_filter // 2, output_size) 
        )

    def encode_temporal(self, action_seq):
        """
        阶段 1: 处理动作序列，提取物理状态。
        input: (Batch, Seq_Len, Action_Dim)
        output: (Batch, Hidden_Dim)
        """
        # LSTM 输出 output, (h_n, c_n)
        # 我们只取最后一个时间步的隐状态 h_n 的最后一层
        _, (h_n, _) = self.lstm(action_seq)
        current_state = h_n[-1] # (Batch, Hidden_Dim)
        return current_state

    def decode_spatial(self, points, state):
        """
        阶段 2: 根据物理状态和空间坐标，预测占据情况。
        points: (Total_Rays, N_samples, 3)
        state:  (Total_Rays, Hidden_Dim) -> 注意：这里的 state 必须已经扩展并与 points 对齐
        """
        n_samples = points.shape[1]
        
        # (Total_Rays, Hidden_Dim) -> (Total_Rays, N_samples, Hidden_Dim)
        state_expanded = state.unsqueeze(1).expand(-1, n_samples, -1)
        
        # (Total_Rays, N_samples, 3) -> (Total_Rays, N_samples, Pos_Dim)
        x_pos = self.pos_encoder(points)
        
        # 拼接
        latent_input = torch.cat([x_pos, state_expanded], dim=-1)
        
        # 解码
        output = self.net(latent_input)
        return output

    def forward(self, points, action_seq):
        """
        兼容旧接口的完整前向传播 (不推荐在 run_batch 中直接使用，效率低)
        """
        state = self.encode_temporal(action_seq)
        # 需要手动扩展 state 以匹配 points 的 batch 维度
        # 假设 points 是 (Batch, N_rays, Samples, 3)
        # 这里逻辑较复杂，建议直接使用 encode/decode 分步调用
        raise NotImplementedError("请分别调用 encode_temporal 和 decode_spatial 以获得最佳性能")
    
    
    