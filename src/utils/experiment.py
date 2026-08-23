"""实验管理工具：自动编号、目录创建、GIF 生成。"""

import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_experiment(base_dir, config=None, *, prefix="exp", now=None,
                      announce=True):
    """创建带日期和当日递增序号的实验目录。

    命名规则: ``{prefix}_{YYYYMMDD}_{idx:03d}``。
    同一 ``base_dir`` 和日期下的序号连续递增；目录创建使用原子 ``mkdir``，
    并发启动时会自动选择下一个序号。

    Args:
        base_dir: 模型级日志目录，如 "train_log/train_mstnf"。
        config: 超参数字典，保存为 config.json。
        prefix: 目录前缀。通用训练使用 ``exp``，真实流水线试次使用 ``trial``。
        now: 可选 datetime，供可复现测试注入固定时间。
        announce: 是否打印创建出的目录。

    Returns:
        exp_dir: 实验目录路径。
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError(f"非法实验目录前缀: {prefix!r}")
    current = now or datetime.now()
    date_str = current.strftime("%Y%m%d")
    escaped = re.escape(prefix)
    name_pattern = re.compile(rf"^{escaped}_{date_str}_(\d+)$")

    os.makedirs(base_dir, exist_ok=True)
    max_idx = -1
    for name in os.listdir(base_dir):
        match = name_pattern.match(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    idx = max_idx + 1
    while True:
        exp_name = f"{prefix}_{date_str}_{idx:03d}"
        exp_dir = os.path.join(base_dir, exp_name)
        try:
            os.mkdir(exp_dir)
            break
        except FileExistsError:
            idx += 1

    if config is not None:
        save_config(exp_dir, config)

    if announce:
        print(f"Experiment: {exp_dir}")
    return exp_dir


def save_config(exp_dir, config):
    """保存超参数到 config.json。"""
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path


def save_gif(exp_dir, filename, frames_pred, frames_gt, epoch_idx,
             action_curves=None, fps=10, skip=1):
    """生成 GT vs Pred 对比 GIF。

    Args:
        exp_dir: 实验目录。
        filename: 文件名，如 "epoch_01.gif"。
        frames_pred: 预测帧列表，每帧 (H, W) numpy 数组。
        frames_gt: GT 帧列表，每帧 (H, W) numpy 数组。
        epoch_idx: 当前 epoch 编号（用于标题）。
        action_curves: 可选，(T, D) 动作曲线数组。
        fps: 帧率。
        skip: 帧降采样步长。
    """
    frames_pred = frames_pred[::skip]
    frames_gt = frames_gt[::skip]
    n_frames = min(len(frames_pred), len(frames_gt))

    has_actions = action_curves is not None
    n_cols = 3 if has_actions else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 2:
        ax_gt, ax_pred = axes
    else:
        ax_gt, ax_pred, ax_act = axes

    im_gt = ax_gt.imshow(frames_gt[0], cmap='gray', vmin=0, vmax=1)
    ax_gt.set_title("GT"); ax_gt.axis('off')

    im_pred = ax_pred.imshow(frames_pred[0], cmap='gray', vmin=0, vmax=1)
    ax_pred.set_title(f"Pred (Ep {epoch_idx})"); ax_pred.axis('off')

    artists = [im_gt, im_pred]
    if has_actions:
        actions_skipped = action_curves[::skip]
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
        for d in range(action_curves.shape[1]):
            ax_act.plot(action_curves[::skip, d], color=colors[d % len(colors)],
                        alpha=0.5, label=f'Act {d}')
        vline = ax_act.axvline(x=0, color='r')
        ax_act.legend(fontsize=8); ax_act.set_title("Action")
        artists.append(vline)

    def update(frame):
        im_gt.set_data(frames_gt[frame])
        im_pred.set_data(frames_pred[frame])
        if has_actions:
            vline.set_xdata([frame, frame])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    save_path = os.path.join(exp_dir, "vis", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close()
    return save_path
