import torch

def get_rays(H, W, focal, eye, center, up, device='cpu'):
    """根据针孔相机参数生成每个像素对应射线。

    Args:
        H: 图像高度。
        W: 图像宽度。
        focal: 相机焦距 (tensor)。
        eye: 相机位置 (tuple or tensor)。
        center: 注视点 (tuple or tensor)。
        up: 上方向向量 (tuple or tensor)。
        device: 计算设备。

    Returns:
        (rays_o, rays_d)，展平后均为 (H*W, 3)。
    """
    focal = torch.tensor(float(focal), dtype=torch.float32, device=device)
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)

    view_dir = center - eye
    view_dir = view_dir / torch.norm(view_dir)
    
    right = torch.linalg.cross(view_dir, up)
    right = right / torch.norm(right)
    
    true_up = torch.linalg.cross(right, view_dir)
    true_up = true_up / torch.norm(true_up)
    
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='ij')
    dir_x = (i - W * 0.5) / focal
    dir_y = -(j - H * 0.5) / focal
    dir_z = torch.ones_like(dir_x)
    rays_d = (dir_x[..., None] * right + 
              dir_y[..., None] * true_up + 
              dir_z[..., None] * view_dir)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = eye.expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)