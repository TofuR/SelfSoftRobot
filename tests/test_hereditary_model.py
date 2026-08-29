"""HereditaryOperatorModel 契约级测试（trainer 兼容性 + 迟滞语义）。

覆盖:
  - trainer 接口契约（forward 签名/返回槽、init_z_from_action、spec）
  - 状态不变量（打包往返、单次消费、episode rollout == 手工步进）
  - 迟滞语义（历史依赖存在、静态项无记忆、恒输入不漂移）

运行: python -m unittest tests.test_hereditary_model -v
"""

import unittest

import torch

from src.models.model_hereditary_operator import HereditaryOperatorModel


def _model(C=2, N=15, K=10, seed=0):
    torch.manual_seed(seed)
    return HereditaryOperatorModel(
        action_dim=C, n_nodes=N, window_size=K,
        n_play=4, n_maxwell=3, dt=0.1,
    )


def _window(B, K, C, seed=0):
    """(B, K, C) 合成动作窗口。"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, K, C, generator=g)


def _rollout(model, aws):
    """按 trainer 模式 rollout（z 线程化），返回 (最终 skeleton, 最终 packed state)。"""
    z = model.init_z_from_action(aws[:, 0])
    skel = None
    for t in range(aws.shape[1]):
        out = model.forward(aws[:, t], None, None, z)
        z = out["latent_z"]
        skel = out["skeleton"]
    return skel, z


class TestTrainerContract(unittest.TestCase):
    def test_forward_signature_and_slots(self):
        """forward(aw, s_prev, s_prev_prev, z) → {skeleton, latent_z}。"""
        model = _model()
        aw = _window(3, 10, 2)
        s = torch.randn(3, 15, 3)
        z = torch.randn(3, model.operator_state_dim)
        out = model.forward(aw, s, s, z)
        self.assertEqual(out["skeleton"].shape, (3, 15, 3))
        self.assertEqual(out["latent_z"].shape, (3, model.operator_state_dim))

    def test_prev_skeleton_ignored(self):
        """无骨架反馈: prev_skeleton 取值不影响输出（gt/open_loop 等价）。"""
        model = _model()
        aw = _window(2, 10, 2)
        z = model.init_z_from_action(aw)
        s1 = torch.zeros(2, 15, 3)
        s2 = torch.randn(2, 15, 3)
        out1 = model.forward(aw, s1, s1, z)
        out2 = model.forward(aw, s2, s2, z)
        self.assertTrue(torch.allclose(out1["skeleton"], out2["skeleton"]))

    def test_training_spec_episode_mode(self):
        """spec 声明 episode 模式（trainer 据此走序列路径）。"""
        spec = HereditaryOperatorModel.training_spec
        phase = spec.phases[0]
        self.assertTrue(phase.use_episode_mode)
        self.assertEqual(phase.dataset_type, "state_transition")
        self.assertEqual(phase.supervision_mode, "spatial_sequence")
        self.assertIn("skeleton", phase.active_losses)

    def test_state_dict_roundtrip(self):
        """state_dict 拷贝到另一实例后输出一致。"""
        model = _model()
        aw = _window(2, 10, 2)
        z = model.init_z_from_action(aw)
        out = model.forward(aw, None, None, z)

        model2 = _model(seed=99)  # 不同初始化
        model2.load_state_dict(model.state_dict())
        out2 = model2.forward(aw, None, None, z)
        self.assertTrue(torch.allclose(out["skeleton"], out2["skeleton"],
                                       atol=1e-6))

    def test_set_normalization_and_denormalize(self):
        """set_normalization + predict_skeleton 反归一化链路。"""
        model = _model()
        center = torch.tensor([1.0, 2.0, 0.0])
        scale = torch.tensor([10.0, 20.0, 1.0])
        model.set_normalization(center, scale, action_norm_factor=1.0)
        aw = _window(1, 10, 2)
        skel = model.predict_skeleton(aw)          # 物理坐标
        self.assertEqual(skel.shape, (1, 15, 3))
        self.assertTrue(torch.isfinite(skel).all())


class TestStateInvariant(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        model = _model()
        p = torch.randn(2, 2, 4)
        h = torch.randn(2, 2, 3)
        p2, h2 = model._unpack_state(model._pack_state(p, h))
        self.assertTrue(torch.allclose(p, p2) and torch.allclose(h, h2))

    def test_single_consumption_of_current_action(self):
        """init_z 烧入 K−1 步后，forward 消费 a_t 恰一次:
        等价于从静息直接烧入整个窗口（同一输入序列，单次计入）。"""
        model = _model()
        aw = _window(1, 10, 2)
        z = model.init_z_from_action(aw)           # 烧入 aw[:, :-1]
        out = model.forward(aw, None, None, z)     # step 用 aw[:, -1]

        # 手工参照: 静息烧入全部 K 步（含最后一步，同一序列）
        p, h = model._burn_in(aw)
        p_out, h_out = model._unpack_state(out["latent_z"])
        self.assertTrue(torch.allclose(p_out, p, atol=1e-6))
        self.assertTrue(torch.allclose(h_out, h, atol=1e-6))

    def test_episode_rollout_matches_manual_stepping(self):
        """trainer 的 z 线程化 rollout == 算子逐库手工步进。"""
        model = _model()
        T, K, C, B = 6, 10, 2, 2
        # (B, T, K, C) 逐 episode 步窗口（forward 只消费各窗口末元素 a_t）
        g = torch.Generator().manual_seed(3)
        aws = torch.rand(B, T, K, C, generator=g)
        _, z_final = _rollout(model, aws)

        # 手工: 烧入首窗口前 K−1 步，再逐步消费各窗口末元素
        p = model.play.init_state(B, aws.device)
        h = model.maxwell.init_state(B, aws.device)
        for k in range(K - 1):
            e = model.drive(aws[:, 0, k])
            p, _ = model.play.step(p, e)
            h = model.maxwell.step(h, e)
        for t in range(T):
            e = model.drive(aws[:, t, -1])
            p, _ = model.play.step(p, e)
            h = model.maxwell.step(h, e)

        p_out, h_out = model._unpack_state(z_final)
        self.assertTrue(torch.allclose(p_out, p, atol=1e-6))
        self.assertTrue(torch.allclose(h_out, h, atol=1e-6))


class TestHysteresisSemantics(unittest.TestCase):
    def test_hold_stability_no_drift(self):
        """恒输入下骨架渐近收敛到不动点，无漂移（v1 电平-当-增量回归）。

        物理上这是粘弹蠕变: Maxwell 亏量按 exp 衰减 → 步差单调不增
        且趋于零，但非一步到位（τ 大的模态持续蠕变）。用小 τ 网格
        （≤1s）验证精确收敛;漂移 bug 的特征是步差不衰减甚至线性增长。
        """
        model = _model()
        with torch.no_grad():  # 换小 τ 网格，300 步内亏量 < 1e-8
            model.maxwell.taus.copy_(
                torch.tensor([0.3, 0.5, 1.0]))
            model.maxwell.decays.copy_(
                torch.exp(-model.dt.item() / model.maxwell.taus))
        K, C = 10, 2
        aw = torch.full((1, K, C), 0.5)
        z = model.init_z_from_action(aw)
        prev, prev_diff = None, None
        for step in range(300):
            out = model.forward(aw, None, None, z)
            z = out["latent_z"]
            if prev is not None:
                diff = (out["skeleton"] - prev).norm().item()
                if prev_diff is not None:
                    # 步差单调不增（无漂移/无振荡发散）;
                    # 容差覆盖 float32 噪声底（~1e-8 量级）
                    self.assertLessEqual(diff, prev_diff + 1e-7)
                prev_diff = diff
            prev = out["skeleton"]
        self.assertLess(prev_diff, 1e-5)   # 渐近收敛到不动点

    def test_history_dependence_exists(self):
        """迟滞存在性: 同一当前动作、不同历史 → play 状态必不同。

        两条路径都终止于 0.5，但 A 曾上冲到 0.8（reversal 记忆），
        B 单调上升（无极值记忆）——PI 状态可分。注意不能把终点设为
        全局最大值（最终上冲会抹平历史）。
        """
        model = _model()
        K, C = 10, 2
        # 路径 A: 0.1 → 0.8 → 0.5（先冲高再回落）
        path_a = torch.cat([
            torch.linspace(0.1, 0.8, K // 2),
            torch.linspace(0.8, 0.5, K - K // 2)]).view(1, K, 1).repeat(1, 1, C)
        # 路径 B: 0.1 → 0.5 单调上升
        path_b = torch.linspace(0.1, 0.5, K).view(1, K, 1).repeat(1, 1, C)
        out_a = model.forward(path_a)
        out_b = model.forward(path_b)
        p_a, _ = model._unpack_state(out_a["latent_z"])
        p_b, _ = model._unpack_state(out_b["latent_z"])
        self.assertFalse(torch.allclose(p_a, p_b))

    def test_static_term_history_free(self):
        """公理 1 构造验证: 关闭算子与残差后，输出与历史无关。"""
        model = _model()
        with torch.no_grad():
            model.play.raw_weights.fill_(-20.0)    # softplus → ≈0
            model.maxwell.weights.zero_()
            model.residual_scale.zero_()
        K, C = 10, 2
        path_a = torch.linspace(0.1, 0.8, K).view(1, K, 1).repeat(1, 1, C)
        path_b = torch.cat([
            torch.linspace(0.9, 0.2, K // 2),
            torch.linspace(0.2, 0.8, K - K // 2)]).view(1, K, 1).repeat(1, 1, C)
        out_a = model.forward(path_a)
        out_b = model.forward(path_b)
        self.assertTrue(torch.allclose(out_a["skeleton"], out_b["skeleton"],
                                       atol=1e-6))

    def test_spectral_report_structure(self):
        """hysteresis_report 给出可定量作图的 (网格, 权重) 对。"""
        model = _model()
        rep = model.hysteresis_report()
        C, J, M = 2, 4, 3
        self.assertEqual(rep["play_weights"].shape, (C, J))
        self.assertEqual(rep["play_thresholds"].shape, (J,))
        self.assertEqual(rep["maxwell_weights"].shape, (C, M))
        self.assertEqual(rep["maxwell_taus"].shape, (M,))
        self.assertTrue(torch.all(rep["play_weights"] >= 0))
        self.assertAlmostEqual(rep["dt"], 0.1, places=6)  # float32 buffer


if __name__ == "__main__":
    unittest.main()
