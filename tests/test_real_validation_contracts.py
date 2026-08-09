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
        # 全 0 落在障碍圆内 → 穿透非零(修 B4:障碍此前放太远,relu(r-d) 恒 0 空转)
        preds = torch.zeros(4, 15, 3, dtype=torch.float64)      # (K,N,3) 归一化
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
        got = obstacle_term(preds, center, scale, obstacles)
        # 手工引用实现:每个 obs 对 (K,N) 全均值求和(即 Σ_obs mean(K,N),不再 /K)
        expected = preds.new_zeros(())
        for (cx, cy, r) in obstacles:
            d = torch.linalg.vector_norm(preds[:, :, :2] - preds.new_tensor((cx, cy)), dim=2)
            expected = expected + torch.relu(r - d).square().mean()
        self.assertTrue(torch.equal(got, expected))
        self.assertGreater(got.item(), 0.0)      # 非空转:障碍与数据确实重叠

    def test_obstacle_term_is_invariant_to_horizon(self):
        """锁 B4:mean-over-(K,N) 使同一 w_obs 的避障压强不随 K 漂移(与 auto_k 兼容)。

        全 0 穿透值(非零 loss);base(K=5)与 doubled(K=10)必须完全相等 ——
        旧 double-divide 会给出 2 倍关系,此断言直接卡死。
        """
        from real_validation.obstacles import obstacle_term
        base = torch.zeros(5, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 1.0)]
        doubled = torch.cat([base, base], 0)     # K=10
        base_value = obstacle_term(base, center, scale, obstacles)
        doubled_value = obstacle_term(doubled, center, scale, obstacles)
        self.assertGreater(base_value.item(), 0.0)   # 穿透非零
        self.assertTrue(torch.equal(base_value, doubled_value))

    def test_z_channel_never_enters_obstacle_loss(self):
        """pc_scale[2]=1e-6,任何非零 z 会被放大 1e6 —— 障碍 loss 必须只吃 [:2]。"""
        from real_validation.obstacles import obstacle_term
        preds = torch.zeros(3, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 1.0)]
        plain = obstacle_term(preds, center, scale, obstacles)
        self.assertGreater(plain.item(), 0.0)    # 障碍重叠 → 非零基线
        noisy = preds.clone()
        noisy[:, :, 2] += 1e3                    # z 通道污染
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


class RolloutEquivalenceTest(unittest.TestCase):
    """T7:runtime/rollout.plan_rollout 与 src 侧 rollout 同输入逐元素相等(CPU)。"""

    def test_rollout_matches_reference_implementation(self):
        from real_validation.runtime.rollout import plan_rollout as wb_rollout
        from real_validation.runtime.model import OpenLoopTransitionModel

        torch.manual_seed(0)
        model = OpenLoopTransitionModel(1, 3, hidden_dim=8, window_size=4,
                                        n_orders=2, z_dim=4).eval()
        buffer = torch.randn(10, 1)
        start_index = 5
        horizon = 3
        s_init = torch.randn(1, 3, 3)

        def reference(buffer_t, t_start, K, window_size, s):
            s_prev = s
            aw0 = buffer_t[t_start - window_size + 1:t_start + 1].unsqueeze(0)
            if aw0.shape[1] < window_size:
                pad = torch.zeros((1, window_size - aw0.shape[1], 1))
                aw0 = torch.cat([pad, aw0], 1)
            z = model.init_z_from_action(aw0)
            preds = []
            for k in range(1, K + 1):
                aw = buffer_t[t_start + k - window_size + 1:t_start + k + 1].unsqueeze(0)
                if aw.shape[1] < window_size:
                    pad = torch.zeros((1, window_size - aw.shape[1], 1))
                    aw = torch.cat([pad, aw], 1)
                out = model(aw, s, s_prev, z)
                s_pred = out["skeleton"]
                z = out["latent_z"]
                preds.append(s_pred.squeeze(0))
                s_prev, s = s, s_pred
            return torch.stack(preds, 0)

        with torch.no_grad():
            expected = reference(buffer, start_index, horizon, 4, s_init)
            got = wb_rollout(model, buffer, start_index, horizon, 4, s_init)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-6))


class CliGuiConsistencyTest(unittest.TestCase):
    """T4b:共享目标核在 CLI 与 GUI 之间逐位一致。

    前置:两侧 norm_factor 必须相等 —— CLI 走 model_loader,找不到 action_norm_factor.txt
    时会静默回落 1.0(model_loader.py:73),不断言就可能"权重相同、norm_factor 不同"假通过。
    """

    def test_shared_objective_matches_across_call_sites(self):
        from real_validation.obstacles import cli_obstacle_loss, obstacle_term
        # 全 0 落在障碍圆内 → 非零 penalty,两个调用点必须逐位一致
        preds = torch.zeros(4, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 1.0)]
        via_cli = cli_obstacle_loss(preds, center, scale, obstacles)
        via_shared = obstacle_term(preds, center, scale, obstacles, reduce="mean")
        self.assertGreater(via_cli.item(), 0.0)   # 非空转
        self.assertTrue(torch.equal(via_cli, via_shared))


class LiveAnchorTest(unittest.TestCase):
    """P3 核心:从相机帧建 anchor(分割→骨架→质量门→归一化→Anchor)。"""

    def _frame(self):
        from tests.test_perception_parity import synthetic_bgr_scene
        bgr, bg = synthetic_bgr_scene()
        # synthetic 场景的臂顶行 = 24 > 质量门控 max_top_row=20 → 上移 10 行使臂顶≤14
        shifted = np.zeros_like(bgr)
        shifted[:-10] = bgr[10:]
        bg_shifted = np.zeros_like(bg)
        bg_shifted[:-10] = bg[10:]
        return shifted, bg_shifted

    def _stub_model(self):
        return type("Geometry", (), {
            "pc_center": torch.zeros(3), "pc_scale": torch.ones(3),
        })()

    def _anchor(self, **overrides):
        from real_validation.live_anchor import anchor_from_camera_frame
        from real_validation.perception.segmentation import segment_white_on_blue
        bgr, bg = self._frame()
        mask = segment_white_on_blue(bgr, bg)
        area = float(mask.sum())
        model = self._stub_model()
        action_history = ((0.1, 0.2, 0.3),) * 40
        kwargs = dict(
            bgr=bgr, background_gray=bg, segment_params={}, n_nodes=15, model=model,
            action_history=action_history, area_median_px=area, frame_ref="test#0")
        kwargs.update(overrides)
        return anchor_from_camera_frame(**kwargs)

    def test_healthy_frame_yields_anchor(self):
        anchor, quality, skeleton = self._anchor()
        self.assertIsNotNone(anchor, f"quality={quality.verdict} {quality.reasons}")
        self.assertEqual(quality.verdict, "ok")
        self.assertEqual(len(anchor.state), 15)
        self.assertEqual(len(anchor.action_history), 40)
        self.assertEqual(len(anchor.action_history[0]), 3)   # action_dim=3
        self.assertEqual(anchor.action_units, "model_normalized")
        self.assertEqual(anchor.state_space, "model_normalized")
        self.assertIn("verdict", anchor.quality)
        self.assertEqual(skeleton.shape, (15, 2))

    def test_normalization_matches_manual(self):
        import numpy as np
        anchor, _, _ = self._anchor()
        state = np.asarray(anchor.state)
        # 归一化 = (px - 0) / 1(stub pc_center=0, pc_scale=1)→ 直接 = 像素
        bgr, bg = self._frame()
        from real_validation.perception.segmentation import segment_white_on_blue
        from real_validation.perception.skeleton import extract_skeleton_2d
        mask = segment_white_on_blue(bgr, bg)
        sk, _ = extract_skeleton_2d(mask, 15, tip_fix=True, return_info=True)
        np.testing.assert_allclose(state, sk[:, :2], atol=1e-6)

    def test_reject_frame_yields_none(self):
        anchor, quality, _ = self._anchor(area_median_px=1.0)   # 面积比巨大 → reject
        self.assertIsNone(anchor)
        self.assertEqual(quality.verdict, "reject")

    def test_prev_state_is_normalized(self):
        # prev_skeleton 必须是像素骨架(与当前骨架同空间,node_step 才小不误拒)
        from real_validation.perception.segmentation import segment_white_on_blue
        from real_validation.perception.skeleton import extract_skeleton_2d
        bgr, bg = self._frame()
        mask = segment_white_on_blue(bgr, bg)
        sk, _ = extract_skeleton_2d(mask, 15, tip_fix=True, return_info=True)
        anchor, quality, _ = self._anchor(prev_skeleton=sk)
        self.assertIsNotNone(anchor, f"quality={quality.verdict} {quality.reasons}")
        self.assertIsNotNone(anchor.prev_state)
        # prev_state 也是归一化空间:stub pc_center=0, pc_scale=1 → ≈像素
        self.assertTrue(np.allclose(np.asarray(anchor.prev_state), sk[:, :2], atol=1e-6))


class WarmupTest(unittest.TestCase):
    """P3:冷启动动作序列 + 6 通道展开。"""

    def test_warmup_actions_shape_and_bounds(self):
        from real_validation.warmup import warmup_actions
        seq = warmup_actions(3, 40)
        self.assertEqual(seq.shape, (40, 3))
        self.assertTrue(np.all(seq >= 0.0) and np.all(seq <= 1.0))
        self.assertGreater(seq.max(), 0.5)     # ramp 到 0.8

    def test_triangle_covers_load_and_unload(self):
        from real_validation.warmup import warmup_actions
        seq = warmup_actions(1, 60, kind="triangle")
        # 升段后再降段:存在 v 先升后降
        peak = seq.max()
        self.assertGreater(peak, 0.5)
        last = seq[-1, 0]
        self.assertLess(last, peak)           # 回落后小于峰值

    def test_expand_to_6ch(self):
        from real_validation.warmup import expand_to_6ch
        actions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        expanded = expand_to_6ch(actions, (0, 1, 2))
        self.assertEqual(expanded.shape, (2, 6))
        np.testing.assert_allclose(expanded[0], [0.1, 0.2, 0.3, 0, 0, 0])
        with self.assertRaises(ValueError):
            expand_to_6ch(actions, (0, 1))    # channel_map 长度≠action_dim


class AutoKTest(unittest.TestCase):
    """B17:step_budget 从学到的 delta_scale 现算;select_k_by_gap 纯函数。"""

    def test_select_k_by_gap_clamps(self):
        from real_validation.planning.auto_k import select_k_by_gap
        self.assertEqual(select_k_by_gap(10.0, 4.0, 4, 40), 4)   # ceil(10/4)=3 < k_min → clamp 到 4
        self.assertEqual(select_k_by_gap(200.0, 4.0, 4, 40), 40)  # ceil(50) > k_max → 40
        self.assertEqual(select_k_by_gap(16.0, 4.0, 4, 40), 4)
        with self.assertRaises(ValueError):
            select_k_by_gap(10.0, 4.0, 40, 4)                     # k_min > k_max

    def test_step_budget_uses_learned_delta_scale(self):
        """delta_scale 初值 0.1 → budget ≈ 0.1×pc_scale,不是 1.0×pc_scale(B17)。"""
        from real_validation.planning.auto_k import step_budget_px
        from real_validation.runtime.model import OpenLoopTransitionModel
        model = OpenLoopTransitionModel(1, 3, hidden_dim=8, window_size=4,
                                        n_orders=2, z_dim=4)
        model.pc_scale.data = torch.tensor([[1.0, 1.0, 1.0]])   # 归一化 → px
        self.assertAlmostEqual(model.delta_scale.item(), 0.1)     # 可学参数初值 0.1
        budget = step_budget_px(model)
        self.assertAlmostEqual(budget, 0.1, places=6)             # 0.1×1.0,不是 1.0
        model.delta_scale.data.fill_(0.7)
        self.assertAlmostEqual(step_budget_px(model), 0.7, places=6)

    def test_gap_point_subtracts_radius(self):
        from real_validation.planning.auto_k import gap_px_point
        self.assertEqual(gap_px_point((10.0, 10.0), (13.0, 10.0), 2.0), 1.0)
        self.assertEqual(gap_px_point((10.0, 10.0), (11.0, 10.0), 2.0), 0.0)  # 圆内


if __name__ == "__main__":
    unittest.main()
