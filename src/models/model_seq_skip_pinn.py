import torch
import torch.nn as nn
import torchvision.models as models
from .layers import ActuatorMLPEncoder, VisualConvDecoder64, TemporalLSTMEncoder

class model_v3(nn.Module):
    """model_v3 (原 Visual_Sensor_Model): 视觉序列 + 驱动的时序感知解码模型。"""

    def __init__(self, n_inputs=20, n_outputs=4, input_channel=1, hidden_size=1024):
        super(model_v3, self).__init__()
        
        # --- 1. 视觉编码器 (保持不变) ---
        # 假设输入是 [Batch, Seq, C, H, W]
        resnet = models.resnet18(pretrained=True)
        # 移除最后的全连接层，保留特征提取部分
        # ResNet18 output before fc is 512
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_feat_dim = 512

        # --- 2. 驱动编码器 (保持不变) ---
        # 使用通用动作编码层（结构等价）
        self.actuator_encoder = ActuatorMLPEncoder(input_dim=n_outputs, feat_dim=32)
        self.actuator_feat_dim = 32

        # --- 3. 动力学核心 (LSTM) ---
        # 输入: 视觉特征 + 驱动特征
        self.lstm_input_size = self.visual_feat_dim + self.actuator_feat_dim
        self.hidden_size = hidden_size
        
        # 使用通用时序编码器，保持与原 LSTM 完全一致的参数和输出
        self.temporal = TemporalLSTMEncoder(
            input_dim=self.lstm_input_size,
            hidden_dim=self.hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )

        # --- 4. 解码器 (融合静态与动态) ---
        # 用通用视觉解码层替换线性+反卷积组合，保持等价维度与流程
        self.decoder_input_dim = self.hidden_size + self.visual_feat_dim
        self.visual_decoder = VisualConvDecoder64(linear_in_dim=self.decoder_input_dim)

    def forward(self, x, q):
        """
        x: [Batch, Sequence, Channels, Height, Width]  (视觉输入)
        q: [Batch, Sequence, Actuator_Dim]            (驱动/压力输入)
        """
        batch_size, seq_len, c, h, w = x.size()

        # 1. 编码视觉特征
        # Reshape 为 [B*S, C, H, W] 进行批量处理
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        vis_feats = self.visual_encoder(x_reshaped) 
        vis_feats = vis_feats.view(batch_size, seq_len, -1) # [B, S, 512]

        # 2. 提取静态外观特征 (Static Appearance)
        # 取每个 Sequence 的第一帧作为"基础形状"
        static_appearance = vis_feats[:, 0, :] # [B, 512]
        # 将静态特征复制扩展到整个时间序列，用于辅助解码
        static_appearance_seq = static_appearance.unsqueeze(1).expand(-1, seq_len, -1) # [B, S, 512]

        # 3. 编码驱动特征
        q_reshaped = q.view(batch_size * seq_len, -1)
        act_feats = self.actuator_encoder(q_reshaped)
        act_feats = act_feats.view(batch_size, seq_len, -1) # [B, S, 32]

        # 4. LSTM 动力学预测
        combined_input = torch.cat([vis_feats, act_feats], dim=2)
        lstm_out, _ = self.temporal(combined_input, return_all=True) # [B, S, 1024], (h,c)
        
        # 5. 解码 (融合静态与动态)
        # 将 LSTM 的动态输出与静态外观特征拼接
        # 这样解码器知道："这是个圆柱体(static)，现在弯曲了30度(dynamic)"
        decode_input_seq = torch.cat([lstm_out, static_appearance_seq], dim=2) # [B, S, 1024+512]
        
        # Flatten for decoder
        decode_input_flat = decode_input_seq.view(batch_size * seq_len, -1)
        
        y_pred = self.visual_decoder(decode_input_flat)
        
        # Reshape back to sequence
        y_pred = y_pred.view(batch_size, seq_len, 3, 64, 64)

        # 返回预测图像 AND 隐变量序列(用于计算物理Loss)
        return y_pred, lstm_out

    def calc_physics_loss(self, latent_seq, dt=0.1):
        """
        计算物理平滑损失 (Physics-Informed Loss)
        解决高频跳变问题
        
        latent_seq: [Batch, Sequence, Hidden_Dim] (即 forward 返回的 lstm_out)
        """
        # 1. 计算一阶导数 (速度)
        # z_{t+1} - z_{t}
        velocity = (latent_seq[:, 1:] - latent_seq[:, :-1]) / dt
        
        # 2. 计算二阶导数 (加速度)
        # v_{t+1} - v_{t}
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        
        # 3. 损失函数：最小化加速度的平方 (最小化 Jerk/突变)
        # 这就是让预测轨迹"平滑"的关键
        smoothness_loss = torch.mean(acceleration ** 2)
        
        return smoothness_loss
