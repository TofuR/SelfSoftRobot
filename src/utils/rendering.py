import torch
import torch.nn as nn


# =============================================================================
# 体渲染函数
# =============================================================================

def OM_rendering(raw: torch.Tensor):
    """不透明度调制的渲染实现（从 func.py 提取并复用）。

    Args:
        raw: 网络原始输出，形状 (N_rays, N_samples, 2)，通道为 [visibility, density]

    Returns:
        render_img: (N_rays,) 累积像素值
        alpha: (N_rays, N_samples) 每点权重
    """
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha * raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=1)
    return render_img, alpha


def OM_rendering_split_output(raw: torch.Tensor):
    """与 `OM_rendering` 类似但返回可见性通道，便于调试。"""
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha * raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=1)
    visibility = raw[..., 0]
    return render_img, alpha, visibility


def robust_mask_rendering(raw, z_vals):
    """修正版 Mask 渲染：去掉无穷远背景墙，适合开放场景。

    与 OM_rendering 的区别在于使用采样点间距而非固定步长计算 alpha。

    Args:
        raw: 网络原始输出，(N_rays, N_samples, 2)。
        z_vals: 采样深度，(N_rays, N_samples)。

    Returns:
        acc_map: (N_rays,) 累积不透明度。
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    last_dist = dists[..., -1:]
    dists = torch.cat([dists, last_dist], -1)

    sigma = nn.functional.softplus(raw[..., 1])
    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device),
                    1. - alpha + 1e-10], -1), -1
    )[:, :-1]
    acc_map = torch.sum(weights, -1)
    return acc_map


# =============================================================================
# 射线采样函数
# =============================================================================

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True):
    """沿射线均匀分层采样。

    Args:
        rays_o: 射线原点，(N_rays, 3)。
        rays_d: 射线方向，(N_rays, 3)。
        near: 近平面距离。
        far: 远平面距离。
        n_samples: 每条射线采样点数。
        perturb: 是否添加随机扰动。

    Returns:
        pts: 采样点，(N_rays, n_samples, 3)。
        z_vals: 采样深度，(N_rays, n_samples)。
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
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


# =============================================================================
# 批量渲染辅助
# =============================================================================

def make_z_vals(near: float, far: float, n_samples: int, rays_n: int, device='cpu'):
    """返回扩展到每条射线的 z_vals, 形状 (rays_n, n_samples)。"""
    t_vals = torch.linspace(0., 1., n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    return z_vals.expand(rays_n, n_samples)


def pts_from_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, z_vals: torch.Tensor):
    """根据射线原点/方向与 z_vals 生成采样点，返回 (N_rays, n_samples, 3)。"""
    return rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)


def prepare_render_inputs(pts_chunk, curr_bs, current_state, current_action):
    """把单块 pts 扩展为模型输入格式。

    Args:
        pts_chunk: (chunk, n_samples, 3)
        curr_bs: batch size
        current_state: (B, Hidden)
        current_action: (B, ActionDim)

    Returns:
        pts_in: (B*chunk, n_samples, 3)
        state_in: (B*chunk, Hidden)
        act_in: (B*chunk, ActionDim)
    """
    n_rays_chunk = pts_chunk.shape[0]
    n_samples = pts_chunk.shape[1]

    pts_in = pts_chunk.unsqueeze(0).expand(curr_bs, -1, -1, -1).reshape(-1, n_samples, 3)
    state_in = current_state.unsqueeze(1).expand(-1, n_rays_chunk, -1).reshape(-1, current_state.shape[-1])
    act_in = current_action.unsqueeze(1).expand(-1, n_rays_chunk, -1).reshape(-1, current_action.shape[-1])

    return pts_in, state_in, act_in


def render_raw_with_model(model, pts_in, state_in, act_in, n_samples):
    """调用 model 获取 raw 输出，兼容常见模型接口。

    优先使用 `model.forward_rendering(pts, state, action)`，
    否则尝试 `model.query_field(pts, state)`。
    """
    if hasattr(model, 'forward_rendering'):
        return model.forward_rendering(pts_in.reshape(-1, n_samples, 3), state_in, act_in)
    elif hasattr(model, 'query_field'):
        return model.query_field(pts_in.reshape(-1, n_samples, 3), state_in)
    else:
        raise RuntimeError('Model does not expose forward_rendering or query_field')
