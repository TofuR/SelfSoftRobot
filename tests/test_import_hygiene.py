"""import 卫生：锁死 real_validation 反向被 src/ 依赖时的最小依赖面。

反向 import 之所以可行，全靠 real_validation/__init__.py 的传递闭包恰好是
stdlib-only —— 这份纯净性没有任何机制保护：任何人往 __init__.py 加一行
`from .model_runtime import ModelRuntime`，离线数据准备脚本就会开始依赖 torch；
往 perception/__init__.py 加 eager import，只要骨架的脚本就被迫装 cv2+scipy
（反面教材：src/data/real/__init__.py:10 就是 eager 绝对 import）。

必须用子进程 —— 同进程里其它测试早已 import torch，会把本测试变成假阴性。
"""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FORBIDDEN = ("torch", "PyQt5", "pyqtgraph", "cv2", "scipy", "matplotlib")
# numpy 不在禁列 —— perception.skeleton 本来就需要 numpy。


def _leaked(statement: str) -> list[str]:
    script = textwrap.dedent(f"""
        import sys
        {statement}
        forbidden = {FORBIDDEN!r}
        leaked = sorted({{name.split('.')[0] for name in sys.modules
                         if name.split('.')[0] in forbidden}})
        print(",".join(leaked))
    """)
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                               capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise AssertionError(f"子进程失败:\n{completed.stderr}")
    payload = completed.stdout.strip()
    return payload.split(",") if payload else []


class ImportHygieneTest(unittest.TestCase):
    def test_package_root_is_stdlib_only(self):
        self.assertEqual(_leaked("import real_validation"), [])

    def test_perception_package_has_no_side_effects(self):
        self.assertEqual(_leaked("import real_validation.perception"), [])

    def test_skeleton_module_needs_no_cv2_or_scipy(self):
        self.assertEqual(
            _leaked("from real_validation.perception.skeleton import "
                    "extract_skeleton_2d, batch_extract_skeleton_2d"), [])

    def test_segmentation_shim_import_never_raises(self):
        # src/data/real/__init__.py:10 是 eager 绝对 import，capture_to_npz.py 与
        # inspect_capture.py 都会走到；薄壳 import-time 绝不能 raise。
        completed = subprocess.run(
            [sys.executable, "-c",
             "import src.data.real; import src.utils.skeleton_2d; print('ok')"],
            cwd=REPO, capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
