"""本地可微 OpenLoop rollout，不依赖项目根目录。"""

from __future__ import annotations

import torch


def window_torch(buffer, index: int, window_size: int):
    start = index - window_size + 1
    if start >= 0:
        return buffer[start:index + 1]
    padding = torch.zeros((-start, buffer.shape[1]), device=buffer.device,
                          dtype=buffer.dtype)
    return torch.cat((padding, buffer[0:index + 1]), dim=0)


def plan_rollout(model, buffer, start_index: int, horizon: int,
                 window_size: int, initial_state):
    state = initial_state
    previous = initial_state
    latent = model.init_z_from_action(
        window_torch(buffer, start_index, window_size).unsqueeze(0))
    predictions = []
    for offset in range(1, horizon + 1):
        action_window = window_torch(
            buffer, start_index + offset, window_size).unsqueeze(0)
        output = model(action_window, state, previous, latent)
        prediction = output["skeleton"]
        latent = output["latent_z"]
        predictions.append(prediction.squeeze(0))
        previous, state = state, prediction
    return torch.stack(predictions, dim=0)
