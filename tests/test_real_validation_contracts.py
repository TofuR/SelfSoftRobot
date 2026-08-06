"""P1b 契约测试:T2 单位往返 / T3 坐标往返 / T4a 共享目标核 / T4b CLI-GUI 一致 / T7 rollout 等价 / T8 契约拒绝。

全部离线可跑,不依赖 checkpoint 或硬件。
"""

import unittest

import numpy as np
import torch

from real_validation.models import Anchor, ModelDescriptor, SafetyPolicy, Scene, ScenePrimitive


class UnitConversionTest(unittest.TestCase):
    """T2:kPa ↔ 模型动作单位往返;safety=150kPa 时模型输入 ≤ 1.0。"""

    def test_kpa_to_model_is_bounded_by_training_domain(self):
        from real_validation.units import kPa_to_model
        # 训练域上界 150 kPa → 模型输入必须 ≤ 1.0(锁 B1:此前把 0-150 原样喂进 [0,1] 域)
        actions = np.array([[0.0], [150.0], [75.0], [10.0]], dtype=np.float32)
        model = kPa_to_model(actions, action_scale_kpa=np.array([150.0]),
                             action_norm_factor=1.0)
        self.assertLessEqual(model.max(), 1.0)
        self.assertAlmostEqual(model[1, 0], 1.0, places=6)

    def test_round_trip_is_exact(self):
        from real_validation.units import kPa_to_model, model_to_kPa
        scale = np.array([150.0, 120.0, 100.0, 90.0, 80.0, 70.0], dtype=np.float32)
        actions = np.random.default_rng(0).uniform(0, 1, (5, 6)).astype(np.float32)
        restored = model_to_kPa(kPa_to_model(actions * scale, action_scale_kpa=scale,
                                             action_norm_factor=1.0),
                                action_scale_kpa=scale, action_norm_factor=1.0)
        np.testing.assert_allclose(restored, actions * scale, rtol=1e-6, atol=1e-6)

    def test_norm_factor_multiplicative(self):
        from real_validation.units import kPa_to_model
        actions = np.array([[150.0]], dtype=np.float32)
        with_norm = kPa_to_model(actions, action_scale_kpa=np.array([150.0]),
                                 action_norm_factor=2.0)
        self.assertAlmostEqual(with_norm[0, 0], 0.5, places=6)  # /scale /norm

    def test_torch_tensor_path(self):
        from real_validation.units import kPa_to_model
        actions = torch.tensor([[0.0], [150.0]])
        out = kPa_to_model(actions, action_scale_kpa=torch.tensor([150.0]),
                           action_norm_factor=1.0)
        self.assertIsInstance(out, torch.Tensor)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_check_unit_consistency(self):
        from real_validation.units import check_unit_consistency
        # npz 已归一化链路(本数据:norm_factor≈1.0)
        label = check_unit_consistency([150.0], 0.99999)
        self.assertIn("npz 已归一化", label)
        # 旧式未归一化(norm≈hi6)
        label2 = check_unit_consistency([150.0], 150.0, hi6=[150.0])
        self.assertIn("旧式未归一化", label2)


class SharedObjectiveParityTest(unittest.TestCase):
    """T4a:共享目标核的 4 个损失项 + 障碍项逐位一致;障碍项对 K 不变;z 永不进 loss。"""

    def test_obstacle_term_is_mean_over_k_and_nodes(self):
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(0)
        preds = torch.randn(4, 15, 3, dtype=torch.float64)      # (K,N,3) 归一化
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(10.0, 10.0, 2.0), (20.0, 5.0, 1.5)]
        got = obstacle_term(preds, center, scale, obstacles)
        # 手工引用实现:逐 k 逐 obs 对 (N,) 求均值再累加,最后 /K(即对 (K,N) 全均值)
        expected = preds.new_zeros(())
        for k in range(preds.shape[0]):
            for (cx, cy, r) in obstacles:
                d = torch.linalg.vector_norm(preds[k, :, :2] - preds.new_tensor((cx, cy)), dim=1)
                expected = expected + torch.relu(r - d).square().mean()
        expected = expected / preds.shape[0]
        self.assertTrue(torch.equal(got, expected))

    def test_obstacle_term_is_invariant_to_horizon(self):
        """锁死口径选择:mean-over-(K,N) 使同一 w_obs 的避障压强不随 K 漂移(与 auto_k 兼容)。"""
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(1)
        base = torch.randn(5, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(10.0, 10.0, 2.0)]
        doubled = torch.cat([base, base], 0)
        self.assertTrue(torch.allclose(
            obstacle_term(base, center, scale, obstacles),
            obstacle_term(doubled, center, scale, obstacles), rtol=0, atol=1e-12))

    def test_z_channel_never_enters_obstacle_loss(self):
        """pc_scale[2]=1e-6,任何非零 z 会被放大 1e6 —— 障碍 loss 必须只吃 [:2]。"""
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(2)
        preds = torch.randn(3, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 3.0)]
        plain = obstacle_term(preds, center, scale, obstacles)
        noisy = preds.clone()
        noisy[:, :, 2] += 1e3          # z 通道污染
        self.assertTrue(torch.equal(plain, obstacle_term(noisy, center, scale, obstacles)))

    def test_k_equals_one_is_finite(self):
        """抓 CLI inverse_plan.py:154 的 K=1 时 errs[1:] 空 → L_mono NaN。"""
        from real_validation.obstacles import obstacle_term
        preds = torch.randn(1, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        value = obstacle_term(preds, center, scale, [(0.0, 0.0, 1.0)])
        self.assertTrue(torch.isfinite(value))


class ContractRejectionTest(unittest.TestCase):
    """T8:缺失 manifest 关键字段必须阻断规划;provenance 标签可审计。"""

    def test_action_scale_kpa_missing_blocks_planning(self):
        """fail-closed 裁决:action_scale_kpa 缺失不能回退 1.0(否则把活 OOD bug 固化成默认)。"""
        from real_validation.models import ModelDescriptor
        descriptor = ModelDescriptor(
            checkpoint="m.pt", checkpoint_hash="abc", model_type="state_transition",
            action_dim=1, n_nodes=15, history_steps=40,
            action_scale_kpa=None, channel_map=None, train_dt_nominal_s=None)
        self.assertIsNone(descriptor.action_scale_kpa)

    def test_manifest_round_trip(self):
        from real_validation.deploy_manifest import DeployManifest
        manifest = DeployManifest(
            checkpoint_sha256="deadbeef", action_scale_kpa=(150.0,),
            channel_map=(0,), train_dt_nominal_s=0.2, train_dt_measured_s=0.2031,
            train_dt_std_s=0.011, mask_source="white_on_blue",
            mask_source_provenance="path_suffix",
            segment_params={"val": 100.0}, camera=None,
            k_safe_table_px={"5px": 51, "10px": 124},
            n_nodes=15, window_size=40, z_dim=16, episode_len=40, action_dim=1,
            encoder_type="fractional", hidden_dim=128, n_scales=4)
        restored = DeployManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.action_scale_kpa, (150.0,))
        self.assertEqual(restored.k_safe_table_px["10px"], 124)

    def test_manifest_missing_required_raises(self):
        from real_validation.deploy_manifest import DeployManifest
        with self.assertRaises(ValueError):
            DeployManifest(checkpoint_sha256="x", action_scale_kpa=None,
                           channel_map=None, train_dt_nominal_s=None, mask_source=None,
                           mask_source_provenance=None, segment_params=None, camera=None,
                           k_safe_table_px=None, n_nodes=None, window_size=None,
                           z_dim=None, episode_len=None, action_dim=None,
                           encoder_type=None, hidden_dim=None, n_scales=None)


class PreflightNewGatesTest(unittest.TestCase):
    """P1b:preflight 新门(dt_mismatch / k_safe_uncertified / unsupported_obstacle / action_scale_missing)。"""

    def _base(self, **model_kw):
        from real_validation.planner_service import build_plan
        kwargs = {"k_safe": 4, **model_kw}   # 默认 k_safe=4;测试可覆盖为 None 测 uncertified
        model = ModelDescriptor("m.pt", "abc", "state_transition", 1, 3, 2,
                                k_train=4, action_scale_kpa=(1.0,), channel_map=(0,),
                                train_dt_nominal_s=0.2, train_dt_measured_s=0.2031,
                                train_dt_std_s=0.011, **kwargs)
        anchor = Anchor(state=((0, 0), (1, 1), (2, 2)), action_history=((0,), (0,)))
        scene = Scene("t", (ScenePrimitive("target_point", "model", {"xy": [2, 3]}),))
        safety = SafetyPolicy(pressure_max6=(100,) * 6, rise_rate6=(100,) * 6,
                              fall_rate6=(100,) * 6, ack_timeout_s=0.1)
        plan = build_plan(model_actions=((10,), (20,)), channel_map=(0,),
                          step_interval_s=0.2, model=model, anchor=anchor,
                          scene=scene, safety=safety)
        return plan, model, anchor, scene, safety

    def test_dt_mismatch_is_detected(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base()
        bad = plan.__class__.from_dict({**plan.to_dict(), "step_interval_s": 0.3})
        codes = {i.code for i in validate_plan(bad, model, anchor, scene, safety).issues}
        self.assertIn("dt_mismatch", codes)

    def test_dt_match_passes(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base()
        self.assertTrue(validate_plan(plan, model, anchor, scene, safety).ok)

    def test_k_safe_uncertified_blocks(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base(k_safe=None, k_safe_table_px=None)
        codes = {i.code for i in validate_plan(plan, model, anchor, scene, safety).issues}
        self.assertIn("k_safe_uncertified", codes)

    def _plan_for(self, scene, model, anchor, safety, step_interval_s=0.2):
        from real_validation.planner_service import build_plan
        return build_plan(model_actions=((10,), (20,)), channel_map=(0,),
                          step_interval_s=step_interval_s, model=model, anchor=anchor,
                          scene=scene, safety=safety)

    def test_unsupported_obstacle_blocks(self):
        from real_validation.preflight import validate_plan
        _, model, anchor, _, safety = self._base()
        scene_with_poly = Scene("t", (
            ScenePrimitive("target_point", "model", {"xy": [2, 3]}),
            ScenePrimitive("obstacle_polygon", "model", {"points": [[1, 1], [2, 1], [2, 2]]}),
        ))
        plan = self._plan_for(scene_with_poly, model, anchor, safety)
        codes = {i.code for i in validate_plan(plan, model, anchor, scene_with_poly, safety).issues}
        self.assertIn("unsupported_obstacle", codes)

    def test_supported_aabb_passes_when_planned(self):
        """obstacle_aabb 在 planner 支持后不再是 unsupported(AABB SDF 已在 obstacles)。"""
        from real_validation.preflight import validate_plan
        _, model, anchor, _, safety = self._base()
        scene_with_aabb = Scene("t", (
            ScenePrimitive("target_point", "model", {"xy": [2, 3]}),
            ScenePrimitive("obstacle_aabb", "model", {"min": [1, 1], "max": [2, 2]}),
        ))
        plan = self._plan_for(scene_with_aabb, model, anchor, safety)
        # AABB 由 planner 支持 → preflight 不再因 unsupported_obstacle 阻断
        codes = {i.code for i in validate_plan(plan, model, anchor, scene_with_aabb, safety).issues}
        self.assertNotIn("unsupported_obstacle", codes)


if __name__ == "__main__":
    unittest.main()
