import torch
import torch.nn as nn
import torchvision.models as models

class model_v3(nn.Module):
    """
    Model V3: Physics-Informed Soft Robot Self-Modeling
    
    设计逻辑:
    1. Static-Dynamic Decoupling: 使用第一帧提取静态外观特征 (Identity)，LSTM提取动态形变特征。
    2. Skip Connection: 将静态特征直接传递给 Decoder，减轻时序网络的重建压力。
    3. Physics Loss Interface: 提供二阶导数计算接口，用于压制预测抖动。
    
    输入 (Input):
    - x: [Batch, Seq, 1, 64, 64] 灰度图像序列 (或 3 通道)
    - q: [Batch, Seq, Actuator_Dim] 驱动参数序列 (如压力值)
    
    输出 (Output):
    - y_pred: [Batch, Seq, 1, 64, 64] 预测的重建图像序列
    - latent_seq: [Batch, Seq, Hidden_Dim] LSTM 输出的隐变量序列，用于计算 Physics Loss
    """
    def __init__(self, n_inputs=1, n_outputs=4, input_channel=1, hidden_size=512):
        super(model_v3, self).__init__()
        
        # 1. Visual Encoder (特征提取)
        resnet = models.resnet18(pretrained=True)
        # 修改第一层以适配输入通道
        if input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1]) # 输出 [B, 512, 1, 1]
        self.vis_dim = 512

        # 2. Actuator Encoder (驱动参数编码)
        self.actuator_encoder = nn.Sequential(
            nn.Linear(n_outputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.act_dim = 64

        # 3. Dynamic Core (基于 LSTM 的动力学学习)
        self.lstm = nn.LSTM(
            input_size=self.vis_dim + self.act_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.hidden_size = hidden_size

        # 4. Visual Decoder (带 Skip Connection 的重建)
        # 输入拼接了 LSTM 输出(动态) 和 第一帧特征(静态)
        self.decoder_input = nn.Linear(hidden_size + self.vis_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channel, 4, 2, 1), # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, x, q):
        b, s, c, h, w = x.shape
        
        # A. 提取每一帧的特征
        x_reshaped = x.view(b * s, c, h, w)
        vis_feats = self.visual_encoder(x_reshaped).view(b, s, -1) # [B, S, 512]
        
        q_reshaped = q.view(b * s, -1)
        act_feats = self.actuator_encoder(q_reshaped).view(b, s, -1) # [B, S, 64]

        # B. 核心改进：锁定第一帧作为静态形态先验 (Identity)
        # 这保证了无论怎么动，机器人的基础结构是稳定的
        static_identity = vis_feats[:, 0, :].unsqueeze(1).repeat(1, s, 1) # [B, S, 512]

        # C. 动力学演化
        combined_in = torch.cat([vis_feats, act_feats], dim=-1)
        dynamic_latent, _ = self.lstm(combined_in) # [B, S, hidden_size]

        # D. 融合静态与动态特征进行解码
        # 这是为了解决重建效果差的问题，Decoder 始终拥有第一帧的高清结构信息
        dec_in = torch.cat([dynamic_latent, static_identity], dim=-1) # [B, S, hidden+512]
        dec_in_flat = dec_in.view(b * s, -1)
        
        z = self.decoder_input(dec_in_flat).view(-1, 256, 4, 4)
        y_pred = self.decoder(z)
        
        return y_pred.view(b, s, c, h, w), dynamic_latent

    def get_physics_loss(self, latent_seq):
        """
        计算隐变量在时间轴上的二阶导数 (加速度)
        用于惩罚不自然的跳变，实现 PINN 的平滑约束
        """
        if latent_seq.shape[1] < 3:
            return torch.tensor(0.0).to(latent_seq.device)
            
        # v = z_t - z_{t-1}
        velocity = latent_seq[:, 1:] - latent_seq[:, :-1]
        # a = v_t - v_{t-1}
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # 惩罚加速度的平均平方和
        return torch.mean(acceleration ** 2)