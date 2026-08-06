"""感知探针的离线验收。

合成帧写进临时目录 → 探针必须产出 overlay/timing/quality 三件产物。
真实数据的验收另有一个 skipUnless 测试（数据已 gitignore，CI 跑不到）。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "real_validation" / "perception_probe.py"


def _write_synthetic_sequence(directory: Path, count: int = 4):
    """写 count 帧合成 BGR PNG + 一张中值背景，返回 (frames_dir, background_path)。"""
    import cv2
    from tests.test_perception_parity import synthetic_bgr_scene

    frames_dir = directory / "cam0"
    frames_dir.mkdir(parents=True)
    frame, bg_gray = synthetic_bgr_scene()
    for index in range(count):
        shifted = np.roll(frame, index, axis=1)
        cv2.imwrite(str(frames_dir / f"{index:05d}.png"), shifted)
    background = directory / "bg_median.png"
    cv2.imwrite(str(background), bg_gray)
    return frames_dir, background


class ProbeTest(unittest.TestCase):
    def test_probe_produces_overlay_timing_and_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames_dir, background = _write_synthetic_sequence(root)
            out = root / "probe"
            completed = subprocess.run(
                [sys.executable, str(PROBE), "--source", "dir",
                 "--frames-dir", str(frames_dir), "--background", str(background),
                 "--n-points", "15", "--frames", "3", "--out", str(out)],
                cwd=REPO, capture_output=True, text=True, timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertTrue((out / "overlay.png").is_file())
            timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
            for key in ("segment_ms", "skeleton_ms", "quality_ms", "total_ms"):
                self.assertIn(key, timing)
                self.assertIn("mean", timing[key])
                self.assertIn("p90", timing[key])
                self.assertGreater(timing[key]["mean"], 0.0)
            self.assertEqual(timing["n_frames"], 3)

            records = [json.loads(line) for line in
                       (out / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertIn(record["verdict"], {"ok", "degraded", "reject"})
                self.assertIn("mask_area_ratio", record)

    def test_source_is_required(self):
        completed = subprocess.run([sys.executable, str(PROBE)], cwd=REPO,
                                   capture_output=True, text=True, timeout=120)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--source", completed.stderr)

    def test_no_builtin_default_frames_dir(self):
        """探针不得内置仓库路径默认值（否则在 PC 上会指向不存在的目录并静默失败）。"""
        source = PROBE.read_text(encoding="utf-8")
        self.assertNotIn("real_capture/data", source)
        self.assertNotIn("seq_20260627", source)


REAL_CAM0 = REPO / "real_capture/data/raw/seq_20260627_163921/cam0"
REAL_BG = REPO / "real_capture/data/derived/seq_20260627_163921/bg_median.png"
REAL_META = REPO / "real_capture/data/derived/seq_20260627_163921/segment_meta.json"


@unittest.skipUnless(REAL_CAM0.is_dir() and REAL_BG.is_file() and REAL_META.is_file(),
                     "真实采集数据不存在（已 gitignore；只能在服务器/本机验收）")
class ProbeOnRealDataTest(unittest.TestCase):
    def test_probe_runs_on_real_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "probe"
            completed = subprocess.run(
                [sys.executable, str(PROBE), "--source", "dir",
                 "--frames-dir", str(REAL_CAM0), "--background", str(REAL_BG),
                 "--segment-params", str(REAL_META), "--n-points", "15",
                 "--frames", "6", "--out", str(out)],
                cwd=REPO, capture_output=True, text=True, timeout=600)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
            # 采集节拍 ~5 fps（action_interval_s=0.2）→ 单帧总耗时必须远低于 200 ms
            self.assertLess(timing["total_ms"]["p90"], 200.0,
                            f"单帧 p90 {timing['total_ms']['p90']:.1f} ms 超过采集节拍预算")
            records = [json.loads(line) for line in
                       (out / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 6)


if __name__ == "__main__":
    unittest.main()
