import torch
import torch.nn as nn
from model import PositionalEncoder

class RecurrentFBV_SM_Skip(nn.Module):
    """
    [改进版] 带直连通道的时序自模型
    结构: 
    1. 历史序列 -> LSTM -> 内部状态 (用于修正迟滞)
    2. 当前动作 -> 直连 -> 解码器 (用于保证基础形状)
    3. 3D坐标 -> PosEnc -> 解码器
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
        
        # 1. 时序编码器 (处理历史记忆)
        self.lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 2. 坐标编码器
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=10, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * 10) 
        
        # 3. 空间解码器 (输入维度增加了 action_dim)
        # 输入: 位置编码(63) + LSTM状态(128) + 当前动作(2)
        input_total_dim = pos_enc_dim + hidden_dim + action_dim
        
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
        阶段 1: 提取历史状态
        input: (Batch, Seq_Len, Action_Dim)
        """
        _, (h_n, _) = self.lstm(action_seq)
        current_state = h_n[-1] # (Batch, Hidden_Dim)
        return current_state

    def decode_spatial(self, points, state, current_action):
        """
        阶段 2: 解码
        points: (Total_Rays, N_samples, 3)
        state:  (Total_Rays, Hidden_Dim) 
        current_action: (Total_Rays, Action_Dim) <--- 新增输入
        """
        n_samples = points.shape[1]
        
        # 扩展状态和动作以匹配采样点数量 (N_samples)
        # (Total_Rays, Hidden) -> (Total_Rays, N_samples, Hidden)
        state_exp = state.unsqueeze(1).expand(-1, n_samples, -1)
        # (Total_Rays, Action) -> (Total_Rays, N_samples, Action)
        action_exp = current_action.unsqueeze(1).expand(-1, n_samples, -1)
        
        # 坐标编码
        x_pos = self.pos_encoder(points)
        
        # 拼接所有信息：位置 + 记忆 + 当前指令
        # 这种 "Skip Connection" 保证了模型即使 LSTM 没训练好，
        # 也能靠 action_exp 像静态模型一样工作。
        latent_input = torch.cat([x_pos, state_exp, action_exp], dim=-1)
        
        output = self.net(latent_input)
        return output
    
    
    