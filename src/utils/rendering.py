import torch
import torch.nn as nn

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


# --- Rendering helpers moved from src/training/renderer_utils.py ---
def make_z_vals(near: float, far: float, n_samples: int, rays_n: int, device='cpu'):
    """返回扩展到每条射线的 z_vals, 形状 (rays_n, n_samples)。"""
    t_vals = torch.linspace(0., 1., n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * (t_vals)
    return z_vals.expand(rays_n, n_samples)


def pts_from_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, z_vals: torch.Tensor):
    """根据射线原点/方向与 z_vals 生成采样点，返回 (N_rays, n_samples, 3)。"""
    return rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)


def prepare_render_inputs(pts_chunk: torch.Tensor, curr_bs: int, current_state: torch.Tensor, current_action: torch.Tensor):
    """把单块 pts 扩展为模型输入格式。

    pts_chunk: (chunk, n_samples, 3)
    返回: pts_in, state_in, act_in，形状分别为
        pts_in: (B*chunk, n_samples, 3)
        state_in: (B*chunk, Hidden)
        act_in: (B*chunk, ActionDim)
    """
    n_rays_chunk = pts_chunk.shape[0]
    n_samples = pts_chunk.shape[1]

    # pts_in: (B, chunk, n_samples, 3) -> (B*chunk, n_samples, 3)
    pts_in = pts_chunk.unsqueeze(0).expand(curr_bs, -1, -1, -1).reshape(-1, n_samples, 3)

    # state_in: (B, Hidden) -> (B, chunk, Hidden) -> (B*chunk, Hidden)
    state_in = current_state.unsqueeze(1).expand(-1, n_rays_chunk, -1).reshape(-1, current_state.shape[-1])

    # act_in: (B, ActionDim) -> (B, chunk, ActionDim) -> (B*chunk, ActionDim)
    act_in = current_action.unsqueeze(1).expand(-1, n_rays_chunk, -1).reshape(-1, current_action.shape[-1])

    return pts_in, state_in, act_in


def render_raw_with_model(model, pts_in: torch.Tensor, state_in: torch.Tensor, act_in: torch.Tensor, n_samples: int):
    """调用 model 获取 raw 输出，兼容常见模型接口。

    优先使用 `model.forward_rendering(pts, state, action)`，否则尝试 `model.query_field(pts, state)`。
    返回 raw_out 张量，调用方需根据模型语义处理 raw_out。
    """
    if hasattr(model, 'forward_rendering'):
        return model.forward_rendering(pts_in.reshape(-1, n_samples, 3), state_in, act_in)
    elif hasattr(model, 'query_field'):
        return model.query_field(pts_in.reshape(-1, n_samples, 3), state_in)
    else:
        raise RuntimeError('Model does not expose forward_rendering or query_field')

# --- End moved helpers ---
