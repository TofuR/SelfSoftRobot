"""ode_solver.py — ODE 积分器，用于 Flow Matching 推理。

训练时不需要 ODE 积分（直接用 flow matching loss），
推理时从高斯噪声 X₀ 积分速度场得到预测点云 X₁。

提供两种积分器：
  - euler_solve: 一阶 Euler 法，简单快速
  - rk4_solve:   四阶 Runge-Kutta，精度更高、步数更少

两种均支持额外关键字参数（如 action），透传给 velocity_net。
"""

import torch


def euler_solve(velocity_net, x0, cond, n_steps=50, return_trajectory=False, **kwargs):
    """Euler 法 ODE 积分。

    从 x0 出发，以 dt = 1/n_steps 为步长，
    迭代 x_{t+dt} = x_t + dt * u_theta(x_t, t | cond, **kwargs)。

    Args:
        velocity_net: 可调用对象，签名 (x_t, t, cond, **kwargs) → velocity。
        x0:    (B, N, 3) 起始点云（通常从 N(0, sigma^2 I) 采样）。
        cond:  (B, cond_dim) 条件向量（由 MultiScaleEMA 编码）。
        n_steps: 积分步数。
        return_trajectory: 是否返回完整轨迹（用于可视化）。
        **kwargs: 额外参数透传给 velocity_net（如 action）。

    Returns:
        若 return_trajectory=False: (B, N, 3) 最终点云。
        若 return_trajectory=True:  (B, n_steps+1, N, 3) 完整轨迹。
    """
    dt = 1.0 / n_steps
    x = x0
    trajectory = [x.detach()] if return_trajectory else None

    for step in range(n_steps):
        t = torch.full((x.shape[0], 1), step * dt, device=x.device, dtype=x.dtype)
        v = velocity_net(x, t, cond, **kwargs)
        x = x + dt * v
        if return_trajectory:
            trajectory.append(x.detach())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)  # (B, n_steps+1, N, 3)
    return x


def rk4_solve(velocity_net, x0, cond, n_steps=20, return_trajectory=False, **kwargs):
    """四阶 Runge-Kutta ODE 积分。

    比 Euler 法精度更高，相同精度下可用更少步数。

    Args:
        velocity_net: 可调用对象，签名 (x_t, t, cond, **kwargs) → velocity。
        x0:    (B, N, 3) 起始点云。
        cond:  (B, cond_dim) 条件向量。
        n_steps: 积分步数（通常 Euler 的 1/2 ~ 1/3 即可）。
        return_trajectory: 是否返回完整轨迹。
        **kwargs: 额外参数透传给 velocity_net（如 action）。

    Returns:
        若 return_trajectory=False: (B, N, 3) 最终点云。
        若 return_trajectory=True:  (B, n_steps+1, N, 3) 完整轨迹。
    """
    dt = 1.0 / n_steps
    x = x0
    trajectory = [x.detach()] if return_trajectory else None

    for step in range(n_steps):
        t_val = step * dt

        t1 = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
        t_half = torch.full((x.shape[0], 1), t_val + dt / 2,
                            device=x.device, dtype=x.dtype)
        t_next = torch.full((x.shape[0], 1), t_val + dt,
                            device=x.device, dtype=x.dtype)

        k1 = velocity_net(x, t1, cond, **kwargs)
        k2 = velocity_net(x + dt / 2 * k1, t_half, cond, **kwargs)
        k3 = velocity_net(x + dt / 2 * k2, t_half, cond, **kwargs)
        k4 = velocity_net(x + dt * k3, t_next, cond, **kwargs)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if return_trajectory:
            trajectory.append(x.detach())

    if return_trajectory:
        return torch.stack(trajectory, dim=1)
    return x
