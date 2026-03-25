import torch
from src.utils.rendering import OM_rendering
from src.utils.rendering import (
    make_z_vals,
    pts_from_rays,
    prepare_render_inputs,
    render_raw_with_model,
)

def run_batch_rendering_nerf(
    model, 
    batch_actions, 
    rays_o_full, rays_d_full, 
    near, far, 
    n_samples, 
    is_train=True, 
    device='cpu'):
    """NeRF 风格的批量渲染。

    Args:
        model: NeRF 模型
        batch_actions: (B, Seq, action_dim)
        rays_o_full, rays_d_full: 全图射线
        near, far, n_samples: 采样参数
        is_train: 是否训练模式
        device: 设备

    Returns:
        pred_img, loss_phy, ray_indices
    """
    curr_bs = batch_actions.shape[0]
    latent_seq = model.get_physics_state(batch_actions)
    current_state = latent_seq[:, -1, :]
    current_action = batch_actions[:, -1, :]
    
    loss_phy = model.compute_smoothness_loss(latent_seq) if is_train else 0.0
    
    if is_train:
        ray_indices = torch.randint(0, rays_o_full.shape[0], (1024,), device=device)
        rays_o = rays_o_full[ray_indices]
        rays_d = rays_d_full[ray_indices]
    else:
        rays_o = rays_o_full
        rays_d = rays_d_full
    
    z_vals = make_z_vals(near, far, n_samples, rays_o.shape[0], device=device)
    pts = pts_from_rays(rays_o, rays_d, z_vals)
    
    all_rgb = []
    chunk_size = 2048
    for i in range(0, rays_o.shape[0], chunk_size):
        pts_chunk = pts[i : i+chunk_size]
        # prepare inputs
        pts_in, state_in, act_in = prepare_render_inputs(pts_chunk, curr_bs, current_state, current_action)

        # model-specific call
        raw_out = render_raw_with_model(model, pts_in, state_in, act_in, n_samples)

        rgb, _ = OM_rendering(raw_out)
        rgb = rgb.view(curr_bs, pts_chunk.shape[0])
        all_rgb.append(rgb)
    
    pred_img = torch.cat(all_rgb, dim=1)
    return pred_img, loss_phy, ray_indices if is_train else None

def run_full_rendering_nerf(
    model, 
    input_seq, 
    rays_o_full, rays_d_full, 
    near, far, 
    n_samples, 
    device='cpu'):
    """全图渲染用于验证。

    Args:
        model: 模型
        input_seq: (1, seq_len, action_dim)
        ...

    Returns:
        pred_flat: (H*W,)
    """
    with torch.no_grad():
        latent_seq = model.get_physics_state(input_seq)
        current_state = latent_seq[:, -1, :]
        
        chunk_size = 2048
        n_rays = rays_o_full.shape[0]
        all_rgb = []
        
        t_vals = torch.linspace(0., 1., n_samples, device=device)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        
        for i in range(0, n_rays, chunk_size):
            rays_o = rays_o_full[i : i+chunk_size]
            rays_d = rays_d_full[i : i+chunk_size]
            n_chunk = rays_o.shape[0]

            z_vals_chunk = make_z_vals(near, far, n_samples, n_chunk, device=device)
            pts = pts_from_rays(rays_o, rays_d, z_vals_chunk)

            # prepare inputs for model.query_field (repeat state)
            state_rep = current_state.repeat_interleave(n_chunk, dim=0)

            density = None
            if hasattr(model, 'query_field'):
                density = model.query_field(pts, state_rep)
            else:
                # fallback: try generic renderer path
                pts_in, state_in, act_in = prepare_render_inputs(pts, 1, current_state, torch.zeros_like(state_rep[:, :1]))
                raw_out = render_raw_with_model(model, pts_in, state_in, act_in, n_samples)
                # raw_out may be (B*chunk, n_samples, 2) -> separate density
                density = raw_out[..., -1:]

            rgb_fake = torch.ones_like(density).repeat(1, 1, 3)
            raw = torch.cat([rgb_fake, density], dim=-1)
            rgb, _ = OM_rendering(raw)
            rgb = rgb.mean(dim=-1, keepdim=True)
            all_rgb.append(rgb)
        
        return torch.cat(all_rgb, dim=0).squeeze()