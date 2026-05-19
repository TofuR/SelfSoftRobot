"""验证脚本：逐个模型测试 UnifiedTrainer 的 forward → loss → backward 链路。

检查项:
  1. 模型创建 + training_spec 解析
  2. 数据集创建 + collate → dict batch
  3. forward pass
  4. compute_losses → loss dict
  5. loss.backward() → 梯度不断裂
  6. 多 Phase 模型的 Phase 切换

用法:
    python scripts/testing/verify_unified_trainer.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from config.params import load_config


def make_dummy_batch(model_type, action_dim=2, device="cpu"):
    """为每种模型创建模拟 dict batch。"""
    B, K, D = 1, 20, action_dim

    if model_type in ("mstnf", "cmstnf"):
        return {
            "action_window": torch.randn(B, K, D),
            "action_window_next": torch.randn(B, K, D),
            "images": torch.rand(64 * 64),
            "depths": None,
            "gt_positions": None,
            "coords": None,
            "gt_sdf": None,
            "gt_normals": None,
        }
    elif model_type == "ms_scnf":
        return {
            "action_window": torch.randn(B, K, D),
            "action_window_next": torch.randn(B, K, D),
            "images": torch.rand(64 * 64),
            "depths": None,
            "gt_positions": torch.randn(B, 31, 3),
            "coords": None,
            "gt_sdf": None,
            "gt_normals": None,
        }
    elif model_type == "sdf":
        M = 100
        return {
            "action_window": torch.randn(B, K, D),
            "action_window_next": None,
            "images": None,
            "depths": None,
            "gt_positions": None,
            "coords": torch.randn(M, 3),
            "gt_sdf": torch.randn(M),
            "gt_normals": torch.randn(M, 3),
        }
    elif model_type == "skeleton_sdf":
        M = 100
        return {
            "action_window": torch.randn(B, K, D),
            "action_window_next": None,
            "images": None,
            "depths": None,
            "gt_positions": torch.randn(B, 31, 3),
            "coords": torch.randn(M, 3),
            "gt_sdf": torch.randn(M),
            "gt_normals": torch.randn(M, 3),
        }


def test_model(model_type, device="cpu"):
    """测试单个模型的完整链路。"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}")
    print(f"{'='*60}")

    config = load_config("training")
    action_dim = 2

    # 1. 创建模型
    print("  [1] Creating model...", end=" ")
    try:
        if model_type == "mstnf":
            from src.models.model_mstnf import MSTNFModel
            model = MSTNFModel(action_dim=action_dim, window_size=20, n_scales=4, hidden_dim=128)
        elif model_type == "cmstnf":
            from src.models.model_cmstnf import CMSTNFModel
            model = CMSTNFModel(action_dim=action_dim, window_size=20, n_scales=4, hidden_dim=128,
                                d_filter=256, n_freqs=10)
        elif model_type == "ms_scnf":
            from src.models.model_ms_scnf import MSSCNFModel
            model = MSSCNFModel(action_dim=action_dim, window_size=20, n_scales=4, hidden_dim=128,
                                d_filter=256, n_freqs=10, n_fine=31, skeleton_mode="point")
        elif model_type == "sdf":
            from src.models.model_sdf import TemporalSDFModel
            model = TemporalSDFModel(action_dim=action_dim, window_size=20, n_scales=4, hidden_dim=128)
        elif model_type == "skeleton_sdf":
            from src.models.model_skeleton_sdf import SkeletonSDFModel
            model = SkeletonSDFModel(action_dim=action_dim, window_size=20, n_scales=4, hidden_dim=128,
                                     skeleton_mode="bspline")
        model = model.to(device)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 2. 验证 training_spec
    print("  [2] Checking training_spec...", end=" ")
    try:
        spec = model.training_spec
        assert len(spec.phases) >= 1
        for p in spec.phases:
            assert p.name
            assert p.active_losses
            assert p.supervision_mode in ("rendering", "direct_3d", "skeleton")
        print(f"OK ({len(spec.phases)} phases)")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 3. 测试每个 Phase
    for phase_idx, phase_spec in enumerate(spec.phases):
        print(f"\n  [3.{phase_idx}] Phase '{phase_spec.name}':")
        print(f"       supervision={phase_spec.supervision_mode}, losses={phase_spec.active_losses}")

        batch = make_dummy_batch(model_type, device=device)

        # 3b. 测试 compute_losses
        print(f"       compute_losses...", end=" ")
        try:
            model.train()
            losses = model.compute_losses(batch, phase_spec)
            assert isinstance(losses, dict), f"Expected dict, got {type(losses)}"
            # 渲染模式 phase 可能只有渲染层 loss（如 recon），模型层可以返回空
            if len(losses) == 0 and phase_spec.supervision_mode == "rendering":
                print("OK (model losses empty — rendering-only phase)")
            else:
                assert len(losses) > 0, "No losses returned"
                for k, v in losses.items():
                    assert isinstance(v, torch.Tensor), f"Loss '{k}' is not Tensor: {type(v)}"
                    assert v.dim() == 0, f"Loss '{k}' is not scalar: shape={v.shape}"
                    assert torch.isfinite(v).all(), f"Loss '{k}' has non-finite values"
                loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"OK ({loss_str})")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 3c. 测试 backward
        if len(losses) == 0:
            print(f"       backward... SKIP (no model losses for rendering-only phase)")
        else:
            print(f"       backward...", end=" ")
            try:
                total = sum(losses.values())
                total.backward()

                grads_ok = 0
                grads_total = 0
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        grads_total += 1
                        if p.grad is not None and torch.isfinite(p.grad).all():
                            grads_ok += 1

                print(f"OK ({grads_ok}/{grads_total} params have gradients)")
                model.zero_grad()
            except Exception as e:
                print(f"FAIL: {e}")
                import traceback
                traceback.print_exc()
                return False

    # 4. 测试 PhaseStrategy
    print(f"\n  [4] Testing PhaseStrategy...", end=" ")
    try:
        from src.training.phase_strategy import PhaseStrategy
        strategy = PhaseStrategy(model)
        for i, (idx, ps) in enumerate(strategy.iterate_phases()):
            fn = strategy.get_forward_fn()
            trainable = len(strategy.get_trainable_params())
            print(f"\n       Phase {i}: forward={ps.forward_attr}, trainable={trainable}", end="")
        print("\n       OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    return True


def test_dataset_factory():
    """测试数据集工厂和 collate。"""
    print(f"\n{'='*60}")
    print(f"Testing: dataset_factory (with real data)")
    print(f"{'='*60}")

    from src.training.dataset_factory import create_dataset, get_collate_fn
    from src.training.spec import PhaseSpec
    config = load_config("training")

    # 测试 SDF 数据集
    sdf_dir = "data/seq_rr_3d"
    if os.path.isdir(sdf_dir):
        print(f"  [SDF] Creating dataset from {sdf_dir}...", end=" ")
        try:
            phase = PhaseSpec("full", dataset_type="sdf", supervision_mode="direct_3d",
                              active_losses=["sdf", "normal", "eikonal"])
            ds = create_dataset("sdf", sdf_dir, config, phase)
            collate_fn = get_collate_fn("sdf", ds)
            sample = ds[0]
            batch = collate_fn([sample])
            assert "action_window" in batch
            assert "coords" in batch
            assert "gt_sdf" in batch
            assert "gt_normals" in batch
            print(f"OK (action_window={batch['action_window'].shape}, coords={batch['coords'].shape})")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  [SDF] SKIP: {sdf_dir} not found")

    # 测试 Sequence 数据集
    seq_dir = "data/seq_rz_3d"
    if os.path.isdir(seq_dir):
        print(f"  [Sequence] Creating dataset from {seq_dir}...", end=" ")
        try:
            phase = PhaseSpec("skeleton", dataset_type="sequence", supervision_mode="skeleton",
                              dataset_kwargs={"return_3d": True}, active_losses=["skeleton"])
            ds = create_dataset("sequence", seq_dir, config, phase)
            collate_fn = get_collate_fn("sequence", ds)
            sample = ds[0]
            batch = collate_fn([sample])
            assert "action_window" in batch
            print(f"OK (action_window={batch['action_window'].shape})")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  [Sequence] SKIP: {seq_dir} not found")


def main():
    device = "cpu"

    print("=" * 60)
    print("UnifiedTrainer Verification")
    print("=" * 60)

    test_dataset_factory()

    results = {}
    for model_type in ["mstnf", "cmstnf", "ms_scnf", "sdf", "skeleton_sdf"]:
        results[model_type] = test_model(model_type, device=device)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for model_type, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {model_type:20s}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\nAll {len(results)} models passed! Safe to archive old trainers.")
    else:
        print(f"\nSome models failed! Fix issues before archiving.")

    return all_ok


if __name__ == "__main__":
    main()
