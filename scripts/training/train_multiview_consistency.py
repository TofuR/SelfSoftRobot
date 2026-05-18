"""train_multiview_consistency.py — 多视角一致性约束训练入口（方案 B）。

在 Plan A（多视角渲染 + 深度监督）基础上增加跨视角一致性约束和重投影约束。

用法:
    # CMSTNF 多视角一致性训练
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk

    # CMSTNF 多视角一致性训练（含深度）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk --depth

    # 覆盖参数
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk \
        --lr 1e-4 --n_epochs 300 --w_consist 0.1 --w_reproj 0.2

支持模型: mstnf, cmstnf, smooth_cmstnf
数据格式: 新版 (N,V,H,W) 数组格式 或 旧版 images_front/side 格式均可
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.params import load_config
from src.utils.config_utils import resolve_config
from src.utils.experiment import create_experiment
from src.data.dataset_multiview_depth import MultiViewDepthDataset
from src.training.trainer_multiview_consistency import MultiViewConsistencyTrainer


def create_model(model_type, action_dim, config):
    """根据模型类型创建模型实例。"""
    temp_cfg = config["temporal"]
    model_cfg = config["model"]

    if model_type == "mstnf":
        from src.models.model_mstnf import MSTNFModel
        return MSTNFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
        )
    elif model_type == "cmstnf":
        from src.models.model_cmstnf import CMSTNFModel
        return CMSTNFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
            d_filter=model_cfg["d_filter"],
            n_freqs=model_cfg["n_freqs"],
        )
    elif model_type == "smooth_cmstnf":
        raise ValueError("smooth_cmstnf 已归档到 docs/archived/smooth_cmstnf/，如需使用请恢复文件")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def multiview_collate(batch):
    """自定义 collate: images_list 和 depths_list 是 list of tensors。"""
    action_windows = torch.stack([b[0] for b in batch])

    n_views = len(batch[0][1])
    images_list = [torch.stack([b[1][v] for b in batch]) for v in range(n_views)]

    depths_list = None
    if batch[0][2] is not None:
        depths_list = [torch.stack([b[2][v] for b in batch]) for v in range(n_views)]

    positions = None
    if batch[0][3] is not None:
        positions = torch.stack([b[3] for b in batch])

    action_window_next = None
    images_next_list = None
    if len(batch[0]) > 4 and batch[0][4] is not None:
        action_window_next = torch.stack([b[4] for b in batch])
        images_next_list = [torch.stack([b[5][v] for b in batch]) for v in range(n_views)]

    return (action_windows, images_list, depths_list,
            positions, action_window_next, images_next_list)


def train(args):
    defaults = load_config("training")
    config = resolve_config(defaults, {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "optimization.batch_size": args.batch_size,
        "multiview.w_depth": args.w_depth,
        "multiview.w_consist": args.w_consist,
        "multiview.w_reproj": args.w_reproj,
        "multiview.n_rays_per_view": args.n_rays_per_view,
        "multiview.n_samples": args.n_samples,
        "multiview.use_depth_guided_sampling": args.depth_guided,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_cfg = config["optimization"]
    mv_cfg = config["multiview"]
    temp_cfg = config["temporal"]

    # ── 数据集 ──
    print(f"\n加载数据: {args.data_dir}")
    full_ds = MultiViewDepthDataset(
        args.data_dir,
        seq_len=temp_cfg["window_size"],
        return_depth=args.depth,
        return_pairs=True,
    )
    cam_system = full_ds.cam_system
    n_views = full_ds.n_views
    action_dim = full_ds.action_dim

    # 训练/验证分割
    n_total = len(full_ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=opt_cfg["batch_size"],
        shuffle=True, num_workers=4, collate_fn=multiview_collate)

    print(f"  训练: {n_train}, 验证: {n_val}, 视角: {n_views}, 动作维度: {action_dim}")
    cam_system.summary()

    # ── 模型 ──
    model = create_model(args.model, action_dim, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {args.model}, 参数量: {n_params:,}")

    # ── 训练器 ──
    trainer = MultiViewConsistencyTrainer(model, cam_system, device, config=config)

    # ── 实验日志 ──
    config_dict = {
        "model": args.model, "action_dim": action_dim,
        "data": args.data_dir, "n_views": n_views,
        "use_depth": args.depth,
        "plan": "B",
        "training": {"lr": opt_cfg["lr"], "batch_size": opt_cfg["batch_size"],
                      "n_epochs": opt_cfg["n_epochs"]},
        "multiview": mv_cfg,
    }
    exp_dir = create_experiment(
        os.path.join("train_log", f"train_multiview_consistency_{args.model}"), config_dict)

    # ── 训练循环 ──
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=opt_cfg["scheduler_patience"])

    n_epochs = opt_cfg["n_epochs"]
    best_val_loss = float("inf")

    print(f"\n>>> 开始训练 (方案B: 多视角一致性约束): {n_epochs} epochs, "
          f"{'含深度监督' if args.depth else '无深度'}")
    print(f"    w_consist={mv_cfg.get('w_consist', 0.05):.3f}, "
          f"w_reproj={mv_cfg.get('w_reproj', 0.1):.3f}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_details = {"recon": 0.0, "depth": 0.0, "consist": 0.0, "reproj": 0.0, "smooth": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for batch in pbar:
            (action_window, images_list, depths_list,
             _, action_window_next, images_next_list) = batch

            losses = trainer.train_step(
                action_window, images_list, depths_list,
                action_window_next=action_window_next,
                images_next_list=images_next_list,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            epoch_loss += losses["total"].item()
            for k in epoch_details:
                if k in losses:
                    epoch_details[k] += losses[k].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.5f}",
                "recon": f"{losses['recon'].item():.4f}",
                "dep": f"{losses['depth'].item():.4f}",
                "reproj": f"{losses['reproj'].item():.4f}",
                "cons": f"{losses['consist'].item():.4f}",
            })

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        n_val_samples = min(30, len(val_ds))
        with torch.no_grad():
            val_indices = np.random.choice(len(val_ds), n_val_samples, replace=False)
            for vi in val_indices:
                batch = val_ds[vi]
                aw = batch[0].unsqueeze(0)
                imgs = [img.unsqueeze(0) for img in batch[1]]
                deps = None
                if batch[2] is not None:
                    deps = [d.unsqueeze(0) for d in batch[2]]
                v_losses = trainer.train_step(aw, imgs, deps)
                val_loss += v_losses["total"].item()
        val_loss /= max(n_val_samples, 1)
        scheduler.step(val_loss)

        avg_train = epoch_loss / max(n_batches, 1)
        avg_recon = epoch_details["recon"] / max(n_batches, 1)
        avg_depth = epoch_details["depth"] / max(n_batches, 1)
        avg_consist = epoch_details["consist"] / max(n_batches, 1)
        avg_reproj = epoch_details["reproj"] / max(n_batches, 1)
        print(f"Epoch {epoch} | Train: {avg_train:.5f} "
              f"(recon={avg_recon:.4f}, depth={avg_depth:.4f}, "
              f"consist={avg_consist:.4f}, reproj={avg_reproj:.4f}) | "
              f"Val: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── 保存 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, "model", "best_model.pt"))
            print(f"  -> best_model.pt saved (val={val_loss:.5f})")

        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, "model", f"model_epoch{epoch:04d}.pt"))

    # ── 最终保存 ──
    torch.save(model.state_dict(), os.path.join(exp_dir, "model", "final_model.pt"))
    np.savetxt(os.path.join(exp_dir, "action_norm_factor.txt"), [full_ds.norm_factor])
    print(f"\n训练完成! 日志: {exp_dir}, Best val: {best_val_loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="多视角一致性约束训练 (方案B)")
    parser.add_argument("--model", type=str, default="mstnf",
                        choices=["mstnf", "cmstnf", "smooth_cmstnf"])
    parser.add_argument("--data_dir", type=str, default="data/exp7_multiview")
    parser.add_argument("--depth", action="store_true", help="启用深度监督")
    parser.add_argument("--depth-guided", action="store_true",
                        help="启用深度引导采样 (coarse-to-fine)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--w_depth", type=float, default=None, help="深度loss权重")
    parser.add_argument("--w_consist", type=float, default=None, help="跨视角一致性权重")
    parser.add_argument("--w_reproj", type=float, default=None, help="重投影一致性权重")
    parser.add_argument("--n_rays_per_view", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
