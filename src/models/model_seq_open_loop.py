import torch
import torch.nn as nn
import torchvision.models as models

class RecurrentPhysicsModel(nn.Module):
    """
    Model V4: Open-Loop Physics-Informed Self-Modeling
    
    核心改进:
    1. True Self-Modeling: 动力学演化 (LSTM) 仅依赖驱动参数 (q) 和初始状态 (x_0)。
       推理阶段不需要后续帧的视觉信息。
    2. Initial State Conditioning: 将第一帧的特征作为 Context 持续注入 LSTM。
    3. PINN Interface: 保留物理约束接口。
    
    输入:
    - x: [B, S, C, H, W] (训练时用整个序列算 Loss，但推理只用 x[:, 0])
    - q: [B, S, Act_Dim] 驱动序列
    
    输出:
    - y_pred: [B, S, C, H, W]
    - latent_seq: [B, S, Hidden]
    """
    def __init__(self, input_channel=1, act_dim=2, hidden_size=512, pretrained=False):
        super(RecurrentPhysicsModel, self).__init__()
        
        # 1. Visual Encoder (只用于第一帧)
        # 使用 ResNet18 提取静态特征
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
            
        if input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1]) # [B, 512, 1, 1]
        self.vis_dim = 512

        # 2. Actuator Encoder
        self.actuator_encoder = nn.Sequential(
            nn.Linear(act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.act_dim = 64

        # 3. Dynamic Core (Open-Loop LSTM)
        # 输入: 当前动作特征 (64) + 初始状态特征 (512, 作为Context)
        # 这样设计让模型知道"我是谁(初始状态)"以及"我在做什么(动作)"
        self.lstm = nn.LSTM(
            input_size=self.act_dim + self.vis_dim, 
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.hidden_size = hidden_size

        # 4. Visual Decoder
        # 输入: LSTM隐变量(动态) + 初始特征(静态)
        self.decoder_input = nn.Linear(hidden_size + self.vis_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4->8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16->32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, input_channel, 4, 2, 1), # 32->64
            nn.Sigmoid()
        )

    def forward(self, x, q):
        b, s, c, h, w = x.shape
        
        # --- Stage A: 初始状态感知 (Initial State Perception) ---
        # 我们只取序列的第一帧 x[:, 0] 进行编码
        # 这符合自建模设定：只知道初始状态
        x_0 = x[:, 0] # [B, C, H, W]
        static_feat = self.visual_encoder(x_0).view(b, 512) # [B, 512]
        
        # --- Stage B: 驱动指令编码 (Actuation Encoding) ---
        # 编码整个动作序列
        q_reshaped = q.view(b * s, -1)
        act_feats = self.actuator_encoder(q_reshaped).view(b, s, -1) # [B, S, 64]
        
        # --- Stage C: 开环动力学演化 (Open-Loop Dynamics) ---
        # 关键修改:
        # LSTM 的输入不包含未来的视觉帧，而是由 (当前动作, 初始状态) 组成
        # 1. 扩展 static_feat 以匹配序列长度
        static_feat_seq = static_feat.unsqueeze(1).repeat(1, s, 1) # [B, S, 512]
        
        # 2. 拼接输入: [动作序列, 初始状态序列]
        # 模型在每一步都能看到"初始状态"，从而推断相对于初始状态的形变
        lstm_input = torch.cat([act_feats, static_feat_seq], dim=-1) # [B, S, 64+512]
        
        # 3. 演化
        dynamic_latent, _ = self.lstm(lstm_input) # [B, S, Hidden]
        
        # --- Stage D: 解码 (Decoding) ---
        # 融合 动态特征(LSTM) 和 静态特征(ResNet)
        dec_in = torch.cat([dynamic_latent, static_feat_seq], dim=-1) # [B, S, Hidden+512]
        
        dec_in_flat = dec_in.view(b * s, -1)
        z = self.decoder_input(dec_in_flat).view(-1, 256, 4, 4)
        y_pred = self.decoder(z)
        
        return y_pred.view(b, s, c, h, w), dynamic_latent

    def get_physics_loss(self, latent_seq):
        """PINN 二阶导数平滑约束"""
        if latent_seq.shape[1] < 3:
            return torch.tensor(0.0, device=latent_seq.device)
        vel = latent_seq[:, 1:] - latent_seq[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        return torch.mean(acc ** 2)
    
    