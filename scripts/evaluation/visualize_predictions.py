"""visualize_predictions.py — 模型预测可视化 CLI。

子命令:
  compare   — GT vs Pred 骨架对比
  animate   — 骨架序列动画 (GIF)
  live      — 实时仿真对比
  render    — 2D 渲染图像对比

用法:
  # GT vs Pred 骨架对比
  python scripts/evaluation/visualize_predictions.py compare \
      --checkpoint train_log/train_ms_scnf/001/phase2/model/best_model.pt \
      --data_dir data/sequence_data_3d

  # 骨架序列动画
  python scripts/evaluation/visualize_predictions.py animate \
      --checkpoint ... --data_dir data/sequence_data_3d \
      --save_path output/animation.gif

  # 实时仿真对比
  python scripts/evaluation/visualize_predictions.py live \
      --checkpoint ... --n_steps 50

  # 2D 渲染对比
  python scripts/evaluation/visualize_predictions.py render \
      --checkpoint ... --data_dir data/sequence_data_3d \
      --n_samples 5 --save_dir output/renders
"""

import os
import sys

CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# compare — GT vs Pred 骨架对比
# =============================================================================

def cmd_compare(args):
    from src.utils.model_loader import load_model
    from src.utils.skeleton_viz import (plot_skeleton_3d, plot_comparison_grid,
                                         plot_error_along_arm, print_metrics)
    from src.data.dataset import SoftSequenceDataset

    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']

    ds = SoftSequenceDataset(args.data_dir, seq_len=info['window_size'],
                             return_3d=True)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True)

    batch = next(iter(loader))
    action_window = batch[0].to(device)
    gt_pos = batch[-1].numpy()  # (B, 3, N)

    with torch.no_grad():
        pred_dict = model.predict_skeleton(action_window)

    pred_skeletons = pred_dict['fine'].cpu().numpy()

    # 单帧对比
    os.makedirs(args.save_dir, exist_ok=True)
    all_errors = []
    for i in range(min(args.n_samples, pred_skeletons.shape[0])):
        pred = pred_skeletons[i]  # (N, 3)
        gt = gt_pos[i].T          # (N, 3)

        errors = np.linalg.norm(pred - gt, axis=1)
        all_errors.append(errors)

        print_metrics(pred, gt, label=f"Sample {i}")

        plot_skeleton_3d(pred, gt=gt, title=f'Sample {i}: GT vs Pred',
                         save_path=os.path.join(args.save_dir, f'compare_{i:03d}.png'),
                         show=False)

    # 误差分布
    mean_errors = np.mean(all_errors, axis=0)
    plot_error_along_arm(mean_errors, title='Average Node-wise Error',
                          save_path=os.path.join(args.save_dir, 'error_distribution.png'),
                          show=False)

    # 多帧网格
    preds = [pred_skeletons[i] for i in range(pred_skeletons.shape[0])]
    gts = [gt_pos[i].T for i in range(gt_pos.shape[0])]
    plot_comparison_grid(preds, gts,
                          save_path=os.path.join(args.save_dir, 'comparison_grid.png'),
                          show=False)

    print(f"\nResults saved to {args.save_dir}/")


# =============================================================================
# animate — 骨架序列动画
# =============================================================================

def cmd_animate(args):
    from src.utils.model_loader import load_model
    from src.utils.skeleton_viz import animate_skeleton_sequence, print_metrics
    from src.data.dataset import SoftSequenceDataset

    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']

    ds = SoftSequenceDataset(args.data_dir, seq_len=info['window_size'],
                             return_3d=True)

    # 取单个序列文件，按时间顺序遍历
    n_frames = min(args.n_frames, len(ds))
    pred_seq = []
    gt_seq = []
    actions_seq = []

    for i in range(n_frames):
        sample = ds[i]
        action_window = sample[0].unsqueeze(0).to(device)
        gt_pos = sample[-1].numpy().T  # (N, 3)
        raw_action = action_window[0, -1].cpu().numpy() * ds.norm_factor

        with torch.no_grad():
            pred_dict = model.predict_skeleton(action_window)

        pred = pred_dict['fine'][0].cpu().numpy()
        pred_seq.append(pred)
        gt_seq.append(gt_pos)
        actions_seq.append(raw_action)

    pred_seq = np.stack(pred_seq)
    gt_seq = np.stack(gt_seq)
    actions_seq = np.stack(actions_seq)

    print(f"\nSequence: {n_frames} frames")
    print_metrics(pred_seq.reshape(-1, 31, 3)[-1:], gt_seq.reshape(-1, 31, 3)[-1:],
                  label="Last frame")

    save_path = args.save_path or 'output/skeleton_animation.gif'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    animate_skeleton_sequence(pred_seq, gt_seq=gt_seq, actions=actions_seq,
                               save_path=save_path, fps=args.fps)
    print(f"Animation saved to {save_path}")


# =============================================================================
# live — 实时仿真对比
# =============================================================================

def cmd_live(args):
    from src.utils.model_loader import load_model

    info = load_model(args.checkpoint, device=device)
    model = info['model']
    norm_factor = info['norm_factor']
    window_size = info['window_size']
    action_dim = info['action_dim']

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from elastica_env import ContinuousSoftArmEnv

    env = ContinuousSoftArmEnv(dt=1e-4)

    history = np.zeros((1, window_size, action_dim))
    sim_steps_per_action = args.sim_steps

    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(121, projection='3d')
    ax_act = fig.add_subplot(122)

    gt_line, = ax3d.plot([], [], [], 'b-o', linewidth=3, markersize=4, label='GT')
    pred_line, = ax3d.plot([], [], [], 'r-o', linewidth=2, markersize=3, label='Pred')
    ax3d.set_xlim(-0.3, 0.3)
    ax3d.set_ylim(-0.3, 0.3)
    ax3d.set_zlim(0, 0.6)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend()
    ax3d.set_title('Live Simulation')

    act_history = []
    act_lines = []
    colors = ['red', 'green']
    for d in range(action_dim):
        l, = ax_act.plot([], [], color=colors[d], label=f'torque_{d}')
        act_lines.append(l)
    ax_act.legend()
    ax_act.set_xlabel('Step')
    ax_act.set_ylabel('Torque')
    ax_act.set_title('Driving Actions')

    print(f"Running {args.n_steps} steps of live simulation...")
    for step in range(args.n_steps):
        # 随机动作
        action = np.random.uniform(-0.3, 0.3, size=action_dim)
        env.set_action(action)
        for _ in range(sim_steps_per_action):
            env.step(steps=1)

        # GT
        _, _, positions, _ = env.get_observation_3d()
        gt = positions.T  # (31, 3)

        # 模型预测
        act_norm = action / norm_factor
        history = np.roll(history, -1, axis=1)
        history[0, -1] = act_norm
        action_tensor = torch.from_numpy(history).float().to(device)

        with torch.no_grad():
            pred_dict = model.predict_skeleton(action_tensor)
            pred = pred_dict['fine'][0].cpu().numpy()

        # 更新图
        gt_line.set_data(gt[:, 0], gt[:, 1])
        gt_line.set_3d_properties(gt[:, 2])
        pred_line.set_data(pred[:, 0], pred[:, 1])
        pred_line.set_3d_properties(pred[:, 2])

        act_history.append(action)
        act_arr = np.array(act_history)
        for d, l in enumerate(act_lines):
            l.set_data(np.arange(len(act_arr)), act_arr[:, d])
        ax_act.set_xlim(0, max(10, len(act_arr)))
        ax_act.set_ylim(-0.5, 0.5)

        ax3d.set_title(f'Step {step+1}/{args.n_steps}')
        plt.pause(0.01)

        if step % 10 == 0:
            err = np.linalg.norm(pred - gt, axis=1).mean()
            print(f"  Step {step:3d}: MNE={err:.6f}m")

    plt.show()


# =============================================================================
# render — 2D 渲染图像对比
# =============================================================================

def cmd_render(args):
    from src.utils.model_loader import load_model
    from src.data.dataset import SoftSequenceDataset
    from src.utils.rendering import sample_stratified, OM_rendering
    from src.utils.camera import get_rays

    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']

    if info['model_type'] != 'ms_scnf' or info['phase'] != 2:
        print("Warning: render requires MS-SCNF phase 2 model. Results may be inaccurate.")

    ds = SoftSequenceDataset(args.data_dir, seq_len=info['window_size'],
                             return_3d=False)

    cam_params = ds.get_camera_params()
    if cam_params is None:
        from src.config.params import get_camera_params
        cam_params = get_camera_params()

    H, W = ds.H, ds.W
    focal = ds.focal if hasattr(ds, 'focal') else cam_params['focal']
    near = cam_params.get('near', 0.5)
    far = cam_params.get('far', 1.5)
    n_samples = cam_params.get('n_samples', 64)

    rays_o, rays_d = get_rays(H, W, focal,
                               cam_params['eye'], cam_params['center'], cam_params['up'],
                               device=device)

    os.makedirs(args.save_dir, exist_ok=True)

    for i in range(min(args.n_samples, len(ds))):
        sample = ds[i]
        action_window = sample[0].unsqueeze(0).to(device)
        gt_img = sample[1].numpy().reshape(H, W)

        with torch.no_grad():
            pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples,
                                             perturb=False)
            raw = model(pts.unsqueeze(0), action_window)
            pred_img, _ = OM_rendering(raw.squeeze(0))
            pred_img = pred_img.reshape(H, W).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(gt_img, cmap='gray')
        ax1.set_title('GT')
        ax1.axis('off')
        ax2.imshow(pred_img, cmap='gray')
        ax2.set_title('Pred')
        ax2.axis('off')

        mse = np.mean((gt_img - pred_img) ** 2)
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        fig.suptitle(f'Sample {i} | MSE={mse:.6f} | PSNR={psnr:.2f}dB')

        save_path = os.path.join(args.save_dir, f'render_{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [{i}] MSE={mse:.6f} PSNR={psnr:.2f}dB → {save_path}")

    print(f"\nRendered images saved to {args.save_dir}/")


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="模型预测可视化工具")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--data_dir", type=str, default="data/sequence_data_3d",
                        help="数据目录")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (默认自动)")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # compare
    p_compare = subparsers.add_parser("compare", help="GT vs Pred 骨架对比")
    p_compare.add_argument("--n_samples", type=int, default=8, help="采样数量")
    p_compare.add_argument("--save_dir", type=str, default="output/compare", help="保存目录")

    # animate
    p_animate = subparsers.add_parser("animate", help="骨架序列动画")
    p_animate.add_argument("--n_frames", type=int, default=100, help="帧数")
    p_animate.add_argument("--save_path", type=str, default=None, help="GIF 保存路径")
    p_animate.add_argument("--fps", type=int, default=10, help="帧率")

    # live
    p_live = subparsers.add_parser("live", help="实时仿真对比")
    p_live.add_argument("--n_steps", type=int, default=50, help="仿真步数")
    p_live.add_argument("--sim_steps", type=int, default=500,
                         help="每动作的仿真积分步数")

    # render
    p_render = subparsers.add_parser("render", help="2D 渲染对比")
    p_render.add_argument("--n_samples", type=int, default=5, help="采样数量")
    p_render.add_argument("--save_dir", type=str, default="output/renders", help="保存目录")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    globals()[f'cmd_{args.command}'](args)


if __name__ == "__main__":
    main()
