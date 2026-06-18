"""train_gt_transition.py — 薄封装：转调 train_transition.py --mode gt（向后兼容）。

gt 与 open_loop 已合并到 train_transition.py（同一网络，仅 teacher_forcing 不同）。
本脚本保留以兼容现有命令与文档引用，行为等价于 `train_transition.py --mode gt`。
所有原参数（--data_dir/--n_epochs/--episode_len/--z_dim/--encoder/--dense_step_weight 等）透传。
"""

import os
import sys

# 让同目录的 train_transition 可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transition import main  # noqa: E402

if __name__ == "__main__":
    # --mode 放最后：argparse "后值优先"，保证封装强制 gt（即使用户误传 --mode）
    main(sys.argv[1:] + ["--mode", "gt"])
