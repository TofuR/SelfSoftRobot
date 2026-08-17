"""与当前 checkpoint 键完全兼容的精简 OpenLoop 推理模型。

这里只保留部署前向路径。当前可用 checkpoint 使用 fractional encoder；遇到其他
encoder 会由 loader 明确拒绝，避免加载随机或不完整权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FractionalMemory(nn.Module):
    def __init__(self, action_dim: int, n_orders: int = 4, window_size: int = 20,
                 hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_orders = n_orders
        self.window_size = window_size
        self.raw_alphas = nn.Parameter(torch.logit(torch.linspace(0.2, 0.8, n_orders)))
        self.order_weights = nn.Parameter(torch.ones(n_orders))
        input_dim = n_orders * action_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

    @property
    def alphas(self):
        return torch.sigmoid(self.raw_alphas)

    @staticmethod
    def _weights(alpha, length: int):
        values = [torch.ones(1, device=alpha.device, dtype=alpha.dtype)]
        for index in range(1, length):
            values.append(values[-1] * (index - 1 - alpha) / index)
        return torch.cat(values)

    def build_weight_cache(self, length: int, device=None, dtype=None) -> None:
        """规划期预计算 GL 权重为常量。alpha 冻结 → 对动作梯度逐位不变(B10)。

        cache key:(length, device, dtype, n_orders, raw_alphas._version, raw_alphas.data_ptr)。
        失效条件:_version 递增(任何对 alpha 的 in-place 改动)/ data_ptr 变化 / length/device/dtype 变化。
        禁止:缓存后二次归一化(引入 ~1e-8 相对误差)、把 order_weights 折进缓存
        (它是 Parameter,改变乘法结合顺序浮点下不相等)、把 4 次 einsum 合成一次
        (改变 reduction 顺序)。不能用 register_buffer(会进 state_dict,撞 loader 的
        严格校验)—— 用普通属性。
        """
        self._cached_weights = {}
        self._cache_key = (int(length), str(device), str(dtype),
                           int(self.n_orders), int(self.raw_alphas._version),
                           int(self.raw_alphas.data_ptr()))
        alpha = self.alphas.detach()
        for index in range(self.n_orders):
            weights = self._weights(alpha[index], length)
            weights = weights / (weights.abs().sum() + 1e-8)
            self._cached_weights[index] = weights.detach()

    def invalidate_weight_cache(self) -> None:
        self._cached_weights = {}
        self._cache_key = None

    def forward(self, action_window):
        _, length, _ = action_window.shape
        features = []
        for index in range(self.n_orders):
            cached = getattr(self, "_cached_weights", None)
            key = getattr(self, "_cache_key", None)
            valid = (cached and key and key ==
                     (int(length), str(action_window.device), str(action_window.dtype),
                      int(self.n_orders), int(self.raw_alphas._version),
                      int(self.raw_alphas.data_ptr())))
            if valid:
                weights = cached[index].to(device=action_window.device,
                                           dtype=action_window.dtype)
            else:
                weights = self._weights(self.alphas[index], length)
                weights = weights / (weights.abs().sum() + 1e-8)
            value = torch.einsum("k,bkd->bd", weights, action_window)
            features.append(value * self.order_weights[index])
        current = action_window[:, -1, :]
        velocity = (current - action_window[:, -2, :]
                    if length >= 2 else torch.zeros_like(current))
        return self.state_mlp(torch.cat((*features, current, velocity), dim=-1))


class OpenLoopTransitionModel(nn.Module):
    def __init__(self, action_dim: int, n_nodes: int, hidden_dim: int = 128,
                 window_size: int = 20, n_orders: int = 4, z_dim: int = 16):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.z_dim = int(z_dim)
        self.register_buffer("pc_center", torch.zeros(1, 1, 3))
        self.register_buffer("pc_scale", torch.ones(1, 1, 3))
        self.register_buffer("action_norm_factor", torch.tensor(1.0))
        self.register_buffer("open_loop_mode", torch.tensor(True))
        self.temporal = FractionalMemory(action_dim, n_orders, window_size, hidden_dim)
        self.z_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.z_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, z_dim))
        self.z_cell = nn.GRUCell(hidden_dim + 3 * n_nodes, z_dim)
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(6 * n_nodes, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self.delta_scale_max = 1.0
        self.init_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def encode(self, action_window):
        return self.temporal(action_window)

    def init_z_from_action(self, action_window):
        return self.z_init(self.encode(action_window))

    def forward(self, action_window, prev_skeleton=None,
                prev_prev_skeleton=None, prev_z=None):
        device = action_window.device
        batch_size = action_window.shape[0]
        condition = self.encode(action_window)
        if prev_z is None:
            latent = self.z_init(condition)
        else:
            flattened = (torch.zeros(batch_size, 3 * self.n_nodes, device=device)
                         if prev_skeleton is None else
                         prev_skeleton.reshape(batch_size, -1))
            latent = self.z_cell(torch.cat((condition, flattened), dim=-1), prev_z)
        latent_projection = self.z_proj(latent)
        if prev_skeleton is None:
            hidden = self.init_hidden(condition)
        else:
            velocity = (torch.zeros_like(prev_skeleton) if prev_prev_skeleton is None
                        else prev_skeleton - prev_prev_skeleton)
            hidden = self.state_encoder(torch.cat((
                prev_skeleton.reshape(batch_size, -1),
                velocity.reshape(batch_size, -1)), dim=-1))
        positions = torch.linspace(-1, 1, self.n_nodes, device=device)
        position_embedding = self.z_embed(positions.view(self.n_nodes, 1))
        sequence = ((condition + latent_projection).unsqueeze(1) +
                    position_embedding.unsqueeze(0))
        output, _ = self.gru(sequence, hidden.unsqueeze(0))
        scale = torch.clamp(self.delta_scale, max=self.delta_scale_max)
        delta = scale * torch.tanh(self.delta_head(output))
        skeleton = delta if prev_skeleton is None else prev_skeleton + delta
        return {"skeleton": skeleton, "latent_z": latent}
