import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# --- [全局 GPU 配置] ---
CUDA_DEVICE = 2  # <--- 在这里修改 GPU 编号
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

# 复用现有的模型定义
from src.models import FBV_SM, PositionalEncoder
# [修改] 仅导入渲染相关函数，get_rays 使用本地修复版
from func import prepare_chunks
from src.utils.rendering import OM_rendering

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# ==========================================
# 1. 核心工具函数 (已修复设备和类型问题)
# ==========================================

def get_rays(height: int, width: int, focal_length: torch.Tensor):
    """
    [修复版] get_rays
    1. 修复了 device 不匹配问题 (torch.arange 默认在 CPU)。
    2. 修复了 dtype 不匹配问题 (rotation_matrix 默认为 Long)。
    """
    # 确保网格生成的 tensor 与 focal_length 在同一个设备上
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=focal_length.device),
        torch.arange(height, dtype=torch.float32, device=focal_length.device),
        indexing='ij')

    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    
    rays_d = directions
    # 保持与原代码一致的相机原点偏移 [1, 0, 0]
    rays_o = torch.tensor([1, 0, 0], dtype=torch.float32, device=focal_length.device).expand(directions.shape)

    # 调整坐标系方向
    rays_d_clone = rays_d.clone()
    rays_d[..., 0], rays_d[..., 2] = rays_d_clone[..., 2].clone(), rays_d_clone[..., 0].clone()

    # [关键修复] 显式指定 dtype=torch.float32，防止报错 "baddbmm_cuda not implemented for Long"
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]], 
                                   device=focal_length.device, 
                                   dtype=torch.float32) # <--- 修复点
    
    # 调整维度以支持广播 (1, 1, 3, 3)
    rotation_matrix = rotation_matrix[None, None]

    rays_d = torch.matmul(rays_d, rotation_matrix)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    return rays_o, rays_d


def get_rays(H, W, focal, eye, center, up):
    """
    根据相机参数 (Eye, Center, Up) 生成世界坐标系下的射线。
    完全匹配 PyVista/VTK 的渲染逻辑。
    """
    # 1. 确保输入是 Tensor 并且在正确的设备上
    device = focal.device
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)

    # 2. 构建相机坐标系 (Camera Coordinate System)
    # Forward (z_cam): 相机看向的方向 (OpenGL标准中相机看向 -Z，所以 Z_cam 指向背后)
    # 但为了计算射线方便，我们先算 view_dir
    view_dir = center - eye
    view_dir = view_dir / torch.norm(view_dir)
    
    # Right (x_cam)
    right = torch.cross(view_dir, up)
    right = right / torch.norm(right)
    
    # True Up (y_cam)
    true_up = torch.cross(right, view_dir)
    true_up = true_up / torch.norm(true_up)
    
    # 3. 生成相机平面上的像素网格
    # PyTorch 的 meshgrid 'ij' 对应 (Height, Width)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 4. 计算局部射线方向 (Camera Space Directions)
    # 假设图像中心对应光轴，像素坐标 (u, v)
    # x = (i - W/2) / focal
    # y = -(j - H/2) / focal  (注意：图像y轴向下，但3D世界y轴通常向上，这里需匹配渲染器的UV方向)
    # PyVista/VTK 渲染出来的图片，数组索引(0,0)通常是左上角
    # 在相机空间中：
    # +X 是右，+Y 是上，-Z 是前
    # 像素 i (0->W) 是从左到右 -> 对应 +X
    # 像素 j (0->H) 是从上到下 -> 对应 -Y
    
    dir_x = (i - W * 0.5) / focal
    dir_y = -(j - H * 0.5) / focal # 翻转Y轴以匹配图像坐标系
    dir_z = torch.ones_like(dir_x) # 假设看向前方 (单位距离)
    
    # 5. 将局部射线变换到世界坐标系 (Rotation)
    # World_Dir = dir_x * Right + dir_y * True_Up + dir_z * View_Dir
    # [W, H, 3]
    rays_d = (
        dir_x[..., None] * right + 
        dir_y[..., None] * true_up + 
        dir_z[..., None] * view_dir
    )
    
    # 归一化方向向量
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # 6. 射线原点 (World Origin)
    # 所有射线都从相机位置出发
    rays_o = eye.expand_as(rays_d)
    
    # 展平为 [N, 3]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    return rays_o, rays_d


def soft_sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True):
    """
    适配软体机器人的采样函数 (移除刚体关节变换)
    """
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # P = O + t * D
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals

def Robust_Mask_Rendering(raw, z_vals):
    """
    修正版：去掉了无穷远的背景墙
    """
    # 1. 计算采样点之间的距离 delta
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    
    # [修改点]：不要用 1e10！用最后一段的平均步长代替
    # 这样光线穿过最后一点后，如果没碰到物体，就会穿出去（变成透明）
    last_dist = dists[..., -1:]
    dists = torch.cat([dists, last_dist], -1)
    
    # 2. 处理密度 (Density)
    sigma = nn.functional.softplus(raw[..., 1]) 
    
    # 3. 计算 Alpha
    alpha = 1.0 - torch.exp(-sigma * dists)
    
    # 4. 累积权重
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # 5. 计算最终 Mask
    acc_map = torch.sum(weights, -1)
    
    return acc_map

def soft_model_forward(rays_o, rays_d, near, far, model, action, chunksize, n_samples=192):
    """
    适配软体机器人的前向传播
    """
    query_points, z_vals = soft_sample_stratified(rays_o, rays_d, near, far, n_samples)
    
    # 扩展 action 以匹配采样点数量
    # query_points: [N_rays, N_samples, 3]
    # action: [DOF] -> [N_rays, N_samples, DOF]
    action_expanded = action.view(1, 1, -1).expand(query_points.shape[0], query_points.shape[1], -1)
    
    # 拼接输入: (X, Y, Z, Torque...)
    model_input = torch.cat((query_points, action_expanded), dim=-1)
    
    batches = prepare_chunks(model_input, chunksize=chunksize)
    predictions = []
    
    for batch in batches:
        # batch 已经在 device 上了 (如果在 prepare_chunks 前放过去)，
        # 但为了保险，再次 .to(device)
        batch = batch.to(device)
        predictions.append(model(batch))
    
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # rgb_map, _ = OM_rendering(raw)
    rgb_map = Robust_Mask_Rendering(raw, z_vals)
    
    return {'rgb_map': rgb_map}

def load_soft_data(data_dir):
    """加载并合并软体机器人训练所需的序列 `.npz` 数据。

    Args:
        data_dir: 包含 `images` 与 `actions` 字段的目录。

    Returns:
        images: 图像数组，形状 (N, H, W[, C])。
        actions: 动作数组，形状 (N, action_dim)。
        focal: 焦距（float）。
    """
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    
    print(f"Found {len(files)} data files. Loading...")
    
    all_images = []
    all_actions = []
    focal = None
    
    for f in sorted(files):
        data = np.load(f)
        all_images.append(data['images'])
        all_actions.append(data['actions'])
        if focal is None and 'focal' in data:
            focal = data['focal']
            
    images = np.concatenate(all_images, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    # 如果没存 focal，给一个默认值
    if focal is None or focal == 1.0:
        height, width = images.shape[1:3]
        # 假设 FOV ~42度
        approx_focal = 0.5 * width / np.tan(0.5 * 30 * np.pi / 180)
        focal = approx_focal
        print(f"Warning: Using calculated focal length: {focal}")
    
    return images, actions, float(focal)

# ==========================================
# 2. 训练主逻辑
# ==========================================

def train():
    """软体机器人神经场基线训练主循环。

    主要流程：
        加载数据 -> 动作归一化 -> 构建模型 -> 生成射线 ->
        随机采样帧监督渲染 -> 反向优化。

    Returns:
        无返回值；会在 `train_log_soft/*` 写入模型与可视化结果。
    """
    # --- 参数配置 ---
    DATA_DIR = "data/sequence_data"
    LOG_DIR = os.path.join("train_log", "train_log_soft", "experiment_3")
    os.makedirs(os.path.join(LOG_DIR, "image"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "model"), exist_ok=True)
    
    N_ITERS = 50000         
    LR = 5e-4               
    DISPLAY_RATE = 500      
    SAVE_RATE = 2000        
    
    # --- 加载数据 ---
    images_np, actions_np, focal_val = load_soft_data(DATA_DIR)
    
    # 动作归一化
    action_max = np.max(np.abs(actions_np))
    if action_max > 0:
        actions_np = actions_np / action_max
        print(f"Actions normalized by max value: {action_max}")
    else:
        action_max = 1.0
        
    np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [action_max])

    # 转 Tensor
    images = torch.from_numpy(images_np).float().to(device)
    actions = torch.from_numpy(actions_np).float().to(device)
    focal = torch.tensor(focal_val).float().to(device)
    
    num_samples = len(images)
    H, W = images.shape[1], images.shape[2]
    DOF = actions.shape[1]
    
    print(f"Data Loaded: {num_samples} frames, {H}x{W}, DOF={DOF}")
    
    idx = np.arange(num_samples)
    split = int(0.9 * num_samples)
    train_idx = idx[:split]
    test_idx = idx[split:]
    
    # --- 初始化模型 ---
    d_input = 3 + DOF 
    
    encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)
    model = FBV_SM(encoder=encoder, d_input=d_input, d_filter=128, output_size=2)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)
    
    # 预计算射线
    # 这些参数必须与您在 elastica_env.py / collect_remote.py 中使用的一模一样！
    CAM_EYE = (1.5, 0.0, 0.5)
    CAM_CENTER = (0.0, 0.0, 0.25)
    CAM_UP = (0.0, 0.0, 1.0)
    
    print(f"Generating rays for Camera: Eye={CAM_EYE}, Center={CAM_CENTER}")
    
    # 调用新函数
    # 注意：`get_rays` 内部会自动处理 device，只要传入的 focal 在 GPU 上
    rays_o, rays_d = get_rays(H, W, focal, CAM_EYE, CAM_CENTER, CAM_UP)
    
    # 确保射线在正确的设备上
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    # 调整近远平面 (Near/Far)
    # 相机(1.5, 0, 0.5) 到 原点(0,0,0) 距离约 1.58
    # 机器人活动范围大概在半径 0.5 左右
    # 所以 Near=1.0, Far=2.5 是比较合理的范围，能包裹住机器人
    NEAR = 0.5
    FAR = 2.5
    
    print(">>> Start Training...")
    
    # --- 训练循环 ---
    for i in trange(N_ITERS):
        model.train()
        
        target_idx = np.random.choice(train_idx)
        target_img = images[target_idx].reshape(-1)
        target_action = actions[target_idx]
        
        outputs = soft_model_forward(
            rays_o.reshape(-1, 3), 
            rays_d.reshape(-1, 3), 
            NEAR, FAR, 
            model, 
            target_action, 
            chunksize=4096*8 
        )
        
        pred_img = outputs['rgb_map']
        loss = torch.nn.functional.mse_loss(pred_img, target_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- 验证 ---
        if i % DISPLAY_RATE == 0:
            model.eval()
            with torch.no_grad():
                val_idx = np.random.choice(test_idx)
                val_action = actions[val_idx]
                val_gt = images[val_idx].cpu().numpy()
                
                val_out = soft_model_forward(
                    rays_o.reshape(-1, 3), 
                    rays_d.reshape(-1, 3), 
                    NEAR, FAR, 
                    model, 
                    val_action, 
                    chunksize=4096*8
                )
                
                val_pred = val_out['rgb_map'].reshape(H, W).cpu().numpy()
                val_loss = np.mean((val_pred - val_gt)**2)
                
                scheduler.step(val_loss)
                
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title(f"GT (Iter {i})")
                plt.imshow(val_gt, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title(f"Pred (Loss: {val_loss:.5f})")
                plt.imshow(val_pred, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(LOG_DIR, "image", f"step_{i:05d}.png"))
                plt.close()

        if i % SAVE_RATE == 0:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", f"model_{i:05d}.pt"))
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

    print("Training Finished.")

if __name__ == "__main__":
    train()
