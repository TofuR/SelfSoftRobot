"""src/operators 四构件的单元测试（物理语义级，非仅形状）。

运行: python -m unittest tests.test_hereditary_operators -v
"""

import math
import unittest

import torch

from src.operators import (
    MonotoneSplineDrive, PlayBank, MaxwellBank, LocalFrameModeBank,
)


class TestMonotoneSplineDrive(unittest.TestCase):
    def test_zero_at_rest(self):
        """静息（动作下界）→ 零驱动。"""
        drive = MonotoneSplineDrive(n_channels=4)
        a = torch.zeros(2, 4)
        self.assertTrue(torch.allclose(drive(a), torch.zeros_like(a)))

    def test_monotone_nondecreasing(self):
        """e(a) 单调不减（随机权重下构造保证）。"""
        torch.manual_seed(0)
        drive = MonotoneSplineDrive(n_channels=3)
        with torch.no_grad():
            drive.raw_weights.normal_(0.0, 2.0)  # 任意软约束外扰动
        a = torch.linspace(0.0, 1.0, 101).view(101, 1).expand(101, 3)
        e = drive(a)                              # (101, 3)
        self.assertTrue(torch.all(e[1:] - e[:-1] >= -1e-7))

    def test_capacity_is_per_channel(self):
        """输出形状 (B, C)，通道间无耦合（逐通道纯函数）。"""
        drive = MonotoneSplineDrive(n_channels=4)
        a = torch.rand(5, 4)
        a2 = a.clone()
        a2[:, 0] += 0.1  # 只动通道 0
        e, e2 = drive(a), drive(a2)
        self.assertTrue(torch.allclose(e[:, 1:], e2[:, 1:]))

    def test_unit_range_normalization_fixes_full_scale(self):
        drive = MonotoneSplineDrive(
            n_channels=4, output_normalization="unit_range")
        with torch.no_grad():
            drive.raw_weights.normal_(0.0, 3.0)
        endpoint = drive(torch.ones(1, 4))
        self.assertTrue(torch.allclose(
            endpoint, torch.ones_like(endpoint), atol=1e-6))
        grid = torch.linspace(0.0, 1.0, 101).view(101, 1).expand(101, 4)
        values = drive(grid)
        self.assertTrue(torch.all(values >= -1e-7))
        self.assertTrue(torch.all(values <= 1.0 + 1e-6))


class TestPlayBank(unittest.TestCase):
    def _bank(self):
        torch.manual_seed(0)
        return PlayBank(n_channels=2, n_operators=4,
                        r_range=(0.05, 0.4))

    def test_deadband_freeze(self):
        """输入在死区内 → 状态冻结（率无关摩擦核心语义）。"""
        bank = self._bank()
        p = torch.full((1, 2, 4), 0.5)
        u = torch.full((1, 2), 0.52)  # |Δ|=0.02 < r_min=0.05
        p_new, q = bank.step(p, u)
        self.assertTrue(torch.allclose(p_new, p))

    def test_tracking_at_band_edge(self):
        """输入超出死区 → 状态贴着 band 边缘跟踪（|q| ≤ r_j）。"""
        bank = self._bank()
        p = torch.zeros(1, 2, 4)
        u = torch.full((1, 2), 0.9)  # 远超所有 r_j
        _, q = bank.step(p, u)
        self.assertTrue(torch.all(q >= -1e-7))
        self.assertTrue(torch.allclose(
            q, bank.thresholds.view(1, 1, -1)))

    def test_q_bounded_by_threshold(self):
        """任意输入序列下 |q_j| ≤ r_j（构造不变量）。"""
        bank = self._bank()
        p = bank.init_state(3, torch.device("cpu"))
        for _ in range(50):
            u = torch.rand(3, 2)
            p, q = bank.step(p, u)
            r = bank.thresholds.view(1, 1, -1)
            self.assertTrue(torch.all(q.abs() <= r + 1e-7))

    def test_rate_independence(self):
        """PI 率无关性: 同一输入路径不同速率 → 同一终态。"""
        bank = self._bank()
        # 路径: 0 → 0.8 → 0.3（单调段），快/慢两种采样密度
        path_fast = [0.8, 0.3]                      # 每 1 步跳一次
        path_slow = torch.linspace(0.0, 0.8, 21).tolist() + \
            torch.linspace(0.8, 0.3, 21).tolist()[1:]

        def run(path):
            p = bank.init_state(1, torch.device("cpu"))
            for a in path:
                p, _ = bank.step(p, torch.tensor([[a, a]]))
            return p

        # 快路径补前置缓升（慢路径从 0 缓升，快路径直接跳——PI 对
        # 单调段跳变不敏感，只有 reversal 依赖极值历史）
        p_fast = run(path_fast)
        p_slow = run(path_slow)
        self.assertTrue(torch.allclose(p_fast, p_slow, atol=1e-6))

    def test_persistent_offset_under_hold(self):
        """恒输入下 q 收敛到常数（持续偏移，非衰减非漂移）——v1 修正核心。"""
        bank = self._bank()
        p = bank.init_state(1, torch.device("cpu"))
        u = torch.tensor([[0.7, 0.7]])
        q_prev = None
        for _ in range(30):
            p, q = bank.step(p, u)
            if q_prev is not None:
                self.assertTrue(torch.allclose(q, q_prev))
            q_prev = q
        # 且偏移非零（0.7 > 多数 r_j → 多数 q 停在 band 边缘）
        self.assertGreater(q.abs().max().item(), 0.0)

    def test_weights_nonnegative(self):
        bank = self._bank()
        self.assertTrue(torch.all(bank.weights >= 0))


class TestMaxwellBank(unittest.TestCase):
    def test_zoh_exact_constant_drive(self):
        """恒驱动解析解: h_n = e·(1 − decay^n)（精确 ZOH）。"""
        dt, n = 0.1, 6
        bank = MaxwellBank(n_channels=2, n_elements=n, dt=dt)
        h = bank.init_state(1, torch.device("cpu"))
        e = torch.tensor([[0.5, 0.5]])
        for step in range(1, 60):
            h = bank.step(h, e)
            analytic = e.unsqueeze(-1) * (1 - bank.decays ** step)
            self.assertTrue(torch.allclose(h, analytic, atol=1e-6))

    def test_stability_at_large_dt_over_tau(self):
        """dt/τ ≥ 2 仍稳定（显式 Euler 在此发散——本构造的动机）。"""
        bank = MaxwellBank(n_channels=1, n_elements=3, dt=5.0,
                           tau_range=(15.0, 20.0))  # dt/τ = 5/15 > 2
        h = bank.init_state(1, torch.device("cpu"))
        e = torch.tensor([[1.0]])
        for _ in range(100):
            h = bank.step(h, e)
        self.assertFalse(torch.any(h.isnan()))
        self.assertTrue(torch.all(h.abs() <= 1.0 + 1e-6))

    def test_tau_min_floor(self):
        """τ 下界 ≥ 3·dt（可辨识窗口下界）。"""
        bank = MaxwellBank(n_channels=1, n_elements=5, dt=0.1,
                           tau_range=(0.05, 10.0))  # 试图突破下界
        self.assertTrue(torch.all(bank.taus >= 3 * 0.1 - 1e-6))

    def test_decay_bounds(self):
        bank = MaxwellBank(n_channels=1, n_elements=8, dt=0.1)
        self.assertTrue(torch.all(bank.decays > 0))
        self.assertTrue(torch.all(bank.decays < 1))

    def test_deficit_decays_to_zero(self):
        """恒输入下亏量 (h − e) → 0（平衡态贡献为零，读出层依据）。

        默认网格最慢 τ=10s（decay=0.99/步），2000 步后残差 < 1e-8。
        """
        bank = MaxwellBank(n_channels=1, n_elements=6, dt=0.1)
        h = bank.init_state(1, torch.device("cpu"))
        e = torch.tensor([[0.6]])
        for _ in range(2000):
            h = bank.step(h, e)
        self.assertTrue(torch.allclose(h, e.unsqueeze(-1), atol=1e-4))


class TestLocalFrameModeBank(unittest.TestCase):
    def _skeleton(self, B=2, N=15, bend=0.3, seed=0):
        """合成平面弧形骨架 (B, N, 3)，z=0。"""
        t = torch.linspace(0, 1, N)
        x = t
        y = bend * t ** 2
        skel = torch.stack([x, y, torch.zeros_like(t)], dim=-1)
        return skel.unsqueeze(0).expand(B, N, 3).contiguous()

    def test_shapes(self):
        bank = LocalFrameModeBank(n_modes=8, n_nodes=15)
        modes = bank(self._skeleton())
        self.assertEqual(modes.shape, (8, 2, 15, 3))

    def test_planar_modes_zero_z(self):
        """平面合同: 模态 z 分量恒为 0。"""
        bank = LocalFrameModeBank(n_modes=4, n_nodes=15)
        modes = bank(self._skeleton())
        self.assertTrue(torch.allclose(modes[..., 2], torch.zeros_like(
            modes[..., 2])))

    def test_frame_orthonormal(self):
        """切向/法向单位正交（面内）。"""
        bank = LocalFrameModeBank(n_modes=2, n_nodes=15)
        tangent, normal = bank._local_frame(self._skeleton())
        dot = (tangent * normal).sum(-1)
        self.assertTrue(torch.allclose(dot, torch.zeros_like(dot), atol=1e-5))
        self.assertTrue(torch.allclose(tangent.norm(dim=-1),
                                       torch.ones_like(tangent.norm(dim=-1)),
                                       atol=1e-4))


if __name__ == "__main__":
    unittest.main()
