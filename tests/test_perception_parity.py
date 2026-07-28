"""感知迁移的行为冻结测试。

_legacy_* 是迁移前 src/utils/skeleton_2d.py 的逐行副本，作为永久对照基线。任何对
real_validation/perception/ 的修改若改变输出，本测试立即失败。

为什么用内联冻结副本而不是"薄壳 vs 实现"对比：薄壳改造后
`src.utils.skeleton_2d.extract_skeleton_2d is real_validation...extract_skeleton_2d`
成立，任何这类比对都是恒真式。

为什么用合成 mask：仓库的 mask 目录被 gitignore，CI 与他人 clone 拿不到；合成输入
覆盖全部代码分支，真实 mask 那一组用 skipUnless 作可选增强。
"""

import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 冻结的旧实现
def _legacy_perpendicular_tip_fix(skeleton, binary_img, n_points):
    sk = skeleton.astype(np.float64)
    if n_points < 5 or np.abs(sk).max() == 0:
        return skeleton
    ys, xs = np.where(binary_img > 0.5)
    if len(xs) < 10:
        return skeleton
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    far = sk[min(max(2, int(0.25 * n_points)), n_points - 1)]
    near = sk[min(max(1, int(0.10 * n_points)), n_points - 1)]
    seg = near - far
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return skeleton
    d = seg / L
    proj = (pts - far) @ d
    w = float(binary_img.sum(1).max())
    slab = proj >= proj.max() - 0.4 * w
    if int(slab.sum()) < 3:
        return skeleton
    node0 = pts[slab].mean(0)
    sk[0] = node0
    a = sk[min(3, n_points - 1)]
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32)


def _legacy_extract_skeleton_2d(binary_img, n_points=31, tip_fix=False):
    H, W = binary_img.shape
    coords = []
    for row in range(H - 1, -1, -1):
        white_cols = np.where(binary_img[row] > 0.5)[0]
        if len(white_cols) > 0:
            coords.append([white_cols.mean(), float(row)])
    if len(coords) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    coords = np.array(coords, dtype=np.float32)
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)
    target_lens = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lens, cum_len, coords[:, 0])
    resampled[:, 1] = np.interp(target_lens, cum_len, coords[:, 1])
    if tip_fix:
        resampled = _legacy_perpendicular_tip_fix(resampled, binary_img, n_points)
    return resampled


# ---------------------------------------------------------------- 合成 mask
def synthetic_masks():
    """覆盖全部代码路径的确定性合成 mask，返回 [(name, (H,W) uint8), ...]。"""
    cases = []

    cases.append(("empty", np.zeros((64, 48), np.uint8)))

    one_row = np.zeros((64, 48), np.uint8)
    one_row[40, 20:26] = 1
    cases.append(("single_row", one_row))

    straight = np.zeros((120, 80), np.uint8)
    straight[20:110, 36:44] = 1
    cases.append(("straight_tube", straight))

    bent = np.zeros((120, 80), np.uint8)
    for row in range(20, 110):
        left = 34 + int(round(0.12 * (row - 20) ** 1.35 / 10.0))
        bent[row, left:left + 9] = 1
    cases.append(("bent_tube", bent))

    # 倾斜末端帽:node0=tip 在图像**底部**，故越往下越窄、且向一侧偏
    # （这正是 _perpendicular_tip_fix 要修的形状——逐行质心对倾斜 cap 做水平切片
    #   会把 node0 落到角落而非中点）
    tilted_cap = bent.copy()
    for offset in range(6):
        row = 109 - offset                      # offset=0 是最底行(最窄)
        tilted_cap[row, :] = 0
        base_left = 34 + int(round(0.12 * (row - 20) ** 1.35 / 10.0))
        width = max(1, 4 + offset)              # 底部 4 → 上方 9，向上变宽
        left = base_left + (5 - offset)         # 底部右偏最多，形成倾斜
        tilted_cap[row, left:left + width] = 1
    cases.append(("tilted_cap", tilted_cap))

    tiny = np.zeros((64, 48), np.uint8)
    tiny[30:33, 20:23] = 1
    cases.append(("tiny_blob", tiny))

    edge = np.zeros((120, 80), np.uint8)
    edge[10:118, 0:7] = 1
    cases.append(("touching_edge", edge))

    return cases


class SkeletonParityTest(unittest.TestCase):
    def test_matches_frozen_reference_on_synthetic_masks(self):
        from real_validation.perception.skeleton import extract_skeleton_2d

        for name, mask in synthetic_masks():
            for n_points in (15, 31):
                for tip_fix in (False, True):
                    tag = f"{name} n={n_points} tip_fix={tip_fix}"
                    expected = _legacy_extract_skeleton_2d(mask, n_points, tip_fix=tip_fix)
                    actual = extract_skeleton_2d(mask, n_points, tip_fix=tip_fix)
                    self.assertEqual(actual.shape, expected.shape, tag)
                    self.assertEqual(actual.dtype, expected.dtype, tag)
                    self.assertTrue(np.array_equal(actual, expected), tag)

    def test_batch_matches_frozen_reference(self):
        from real_validation.perception.skeleton import batch_extract_skeleton_2d

        masks = np.stack([mask for _, mask in synthetic_masks()
                          if mask.shape == (120, 80)])
        expected = np.stack([_legacy_extract_skeleton_2d(m, 15, tip_fix=True)
                             for m in masks])
        actual = batch_extract_skeleton_2d(masks, 15, tip_fix=True)
        self.assertTrue(np.array_equal(actual, expected))

    def test_tip_fix_default_stays_false(self):
        """compare_skeleton_methods.py 的 5 处调用靠这个默认值充当 M0 未修基线。"""
        import inspect

        from real_validation.perception.skeleton import (
            batch_extract_skeleton_2d, extract_skeleton_2d,
        )
        for function in (extract_skeleton_2d, batch_extract_skeleton_2d):
            default = inspect.signature(function).parameters["tip_fix"].default
            self.assertIs(default, False, function.__name__)

    def test_returns_bare_ndarray(self):
        """masks_to_transition_npz.py:91 有 `T, N, _ = sk2d.shape`，返回元组会立刻 ValueError。"""
        from real_validation.perception.skeleton import (
            batch_extract_skeleton_2d, extract_skeleton_2d,
        )
        mask = dict(synthetic_masks())["bent_tube"]
        self.assertIsInstance(extract_skeleton_2d(mask, 15, tip_fix=True), np.ndarray)
        self.assertIsInstance(batch_extract_skeleton_2d(mask[None], 15), np.ndarray)

    def test_shim_reexports_same_objects(self):
        import src.utils.skeleton_2d as shim
        from real_validation.perception import skeleton as canonical

        self.assertIs(shim.extract_skeleton_2d, canonical.extract_skeleton_2d)
        self.assertIs(shim.batch_extract_skeleton_2d, canonical.batch_extract_skeleton_2d)
        self.assertIs(shim._perpendicular_tip_fix, canonical._perpendicular_tip_fix)
        self.assertTrue(callable(shim.project_3d_to_2d))
        self.assertTrue(callable(shim.compute_2d_skeleton_loss))

    def test_shim_still_exports_torch_helpers(self):
        """_smoke_triangulation.py:76 把 ImportError 吞进 except，漏 re-export 会静默失活。"""
        import src.utils.skeleton_2d as shim

        for name in ("project_3d_to_2d", "compute_2d_skeleton_loss"):
            self.assertIn(name, shim.__all__)
            self.assertTrue(callable(getattr(shim, name)))


class TipFixObservabilityTest(unittest.TestCase):
    """B13:tip_fix 的门控原先是静默跳过,在线质量门控需要"是否生效"信号。"""

    def _info(self, mask, n_points=15, tip_fix=True):
        from real_validation.perception.skeleton import extract_skeleton_2d
        return extract_skeleton_2d(mask, n_points, tip_fix=tip_fix, return_info=True)

    def test_applied_on_tilted_cap(self):
        _, info = self._info(dict(synthetic_masks())["tilted_cap"])
        self.assertTrue(info["tip_fix_requested"])
        self.assertTrue(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "applied")

    def test_skip_reason_too_few_points(self):
        _, info = self._info(dict(synthetic_masks())["bent_tube"], n_points=4)
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "n_points_lt_5")

    def test_skip_reason_too_few_foreground(self):
        mask = np.zeros((40, 20), np.uint8)
        mask[10:13, 8:11] = 1
        self.assertLess(int(mask.sum()), 10)
        _, info = self._info(mask)
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "foreground_lt_10")

    def test_not_requested_reports_reason(self):
        _, info = self._info(dict(synthetic_masks())["bent_tube"], tip_fix=False)
        self.assertFalse(info["tip_fix_requested"])
        self.assertFalse(info["tip_fix_applied"])
        self.assertEqual(info["tip_fix_reason"], "not_requested")

    def test_empty_mask_reports_zero_skeleton(self):
        _, info = self._info(dict(synthetic_masks())["empty"])
        self.assertEqual(info["tip_fix_reason"], "zero_skeleton")
        self.assertEqual(info["n_valid_rows"], 0)

    def test_return_info_false_keeps_legacy_return_type(self):
        from real_validation.perception.skeleton import extract_skeleton_2d
        result = extract_skeleton_2d(dict(synthetic_masks())["bent_tube"], 15, tip_fix=True)
        self.assertIsInstance(result, np.ndarray)

    def test_info_values_match_bare_result(self):
        """return_info=True 的骨架必须与 return_info=False 逐位相同(同一计算)。"""
        from real_validation.perception.skeleton import extract_skeleton_2d
        mask = dict(synthetic_masks())["tilted_cap"]
        bare = extract_skeleton_2d(mask, 15, tip_fix=True)
        with_info, _ = extract_skeleton_2d(mask, 15, tip_fix=True, return_info=True)
        self.assertTrue(np.array_equal(bare, with_info))


REAL_MASKS = REPO / "real_capture/data/derived/seq_20260627_163921/masks"


@unittest.skipUnless(REAL_MASKS.is_dir(), "真实 mask 目录不存在（已 gitignore）")
class SkeletonParityOnRealMasksTest(unittest.TestCase):
    def test_matches_frozen_reference_on_50_real_masks(self):
        import cv2

        from real_validation.perception.skeleton import extract_skeleton_2d

        files = sorted(REAL_MASKS.glob("*.png"))
        self.assertGreater(len(files), 50)
        for index in np.linspace(0, len(files) - 1, 50).astype(int):
            path = files[index]
            mask = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            expected = _legacy_extract_skeleton_2d(mask, 15, tip_fix=True)
            actual = extract_skeleton_2d(mask, 15, tip_fix=True)
            self.assertTrue(np.array_equal(actual, expected), path.name)


if __name__ == "__main__":
    unittest.main()
