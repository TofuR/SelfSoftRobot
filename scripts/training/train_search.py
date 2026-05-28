"""train_search.py — 超参数搜索：生成命令并依次执行 train_unified.py。

设计原则:
  - 零代码重复：直接调用 train_unified.py 作为子进程
  - 可中断续跑：--resume 跳过已完成实验（检测 exp_dir 存在且含 best_model.pt）
  - 可只生成不执行：--dry_run 只打印命令，方便手动选择/编辑
  - 汇总表格：--summarize 扫描实验目录的 config.json 提取结果

用法:
    # 搜索学习率（生成 4 个命令并执行）
    python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rz_c6_sk \
        --search lr=1e-4,3e-4,1e-3,3e-3

    # 只打印命令不执行（手动复制粘贴或保存为 .sh）
    python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rz_c6_sk \
        --search lr=1e-4,3e-4,1e-3,3e-3 --dry_run

    # 中断后续跑（跳过已有 best_model.pt 的实验）
    python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rz_c6_sk \
        --search lr=1e-4,3e-4,1e-3,3e-3 --resume

    # 多参数网格搜索（2×3=6 组）
    python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rz_c6_sk \
        --search lr=1e-4,1e-3 --search batch_size=2,4,8

    # 汇总已完成的搜索结果
    python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rz_c6_sk \
        --search lr=1e-4,3e-4,1e-3,3e-3 --summarize

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_search.py ...

搜索参数与 train_unified.py 的 CLI 参数名一致:
    lr, batch_size, n_epochs, phase1_epochs, phase2_epochs,
    skeleton_mode, w_skeleton_fine, w_skeleton_medium, w_skeleton_coarse,
    n_freqs, d_filter, deform_n_freqs, n_rays, n_samples, chunk_size
"""

import os
import sys
import json
import argparse
import itertools
import subprocess
import glob
from datetime import datetime  # noqa: F401 — used in summarize filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── 搜索参数名 → train_unified.py 的 CLI 参数名 + 类型 ──
PARAM_MAP = {
    "lr":                ("--lr", float),
    "batch_size":        ("--batch_size", int),
    "n_epochs":          ("--n_epochs", int),
    "phase1_epochs":     ("--phase1_epochs", int),
    "phase2_epochs":     ("--phase2_epochs", int),
    "skeleton_mode":     ("--skeleton_mode", str),
    "n_freqs":           ("--n_freqs", int),
    "d_filter":          ("--d_filter", int),
    "deform_n_freqs":    ("--deform_n_freqs", int),
    "n_rays":            ("--n_rays", int),
    "n_samples":         ("--n_samples", int),
    "chunk_size":        ("--chunk_size", int),
}

# 训练脚本路径
TRAINER = os.path.join(os.path.dirname(__file__), "train_unified.py")


def parse_search_specs(specs):
    """解析 --search name=val1,val2,... 为参数组合列表。

    Returns:
        list[dict]: 每个元素是 {param_name: typed_value}。
    """
    search_axes = []
    for spec in specs:
        name, vals_str = spec.split("=")
        name = name.strip()
        if name not in PARAM_MAP:
            raise ValueError(f"Unknown param '{name}', choose from: {list(PARAM_MAP)}")
        cli_flag, dtype = PARAM_MAP[name]
        vals = [dtype(v.strip()) for v in vals_str.split(",")]
        search_axes.append((name, cli_flag, vals))

    combos = []
    for combo_vals in itertools.product(*[a[2] for a in search_axes]):
        combo = {}
        for (name, cli_flag, _), val in zip(search_axes, combo_vals):
            combo[name] = val
        combos.append(combo)
    return combos


def build_command(args, combo):
    """构建一条 train_unified.py 命令。"""
    cmd = [sys.executable, TRAINER,
           "--model", args.model,
           "--data_dir", args.data_dir]
    if args.canonical_data_dir:
        cmd += ["--canonical_data_dir", args.canonical_data_dir]
    if args.multiview:
        cmd.append("--multiview")
    if args.depth:
        cmd.append("--depth")
    if args.consistency:
        cmd.append("--consistency")
    if args.num_workers:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.phase:
        cmd += ["--phase", str(args.phase)]

    # 全局覆盖参数（搜索参数优先，跳过已搜索的）
    search_names = set(combo.keys())
    if args.lr is not None and "lr" not in search_names:
        cmd += ["--lr", str(args.lr)]
    if args.n_epochs is not None and "n_epochs" not in search_names:
        cmd += ["--n_epochs", str(args.n_epochs)]
    if args.batch_size is not None and "batch_size" not in search_names:
        cmd += ["--batch_size", str(args.batch_size)]

    # 本次搜索参数
    for name, val in combo.items():
        cli_flag = PARAM_MAP[name][0]
        cmd += [cli_flag, str(val)]

    return cmd


def is_completed(args, combo):
    """检查某个组合是否已完成（实验目录存在且含 best_model.pt）。"""
    tag = f"{args.model}_search"
    search_dir = os.path.join("train_log", tag)
    if not os.path.exists(search_dir):
        return False

    for exp_dir in sorted(glob.glob(os.path.join(search_dir, "exp_*"))):
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        has_best = (os.path.exists(os.path.join(exp_dir, "phase_skeleton", "model", "best_model.pt"))
                    or os.path.exists(os.path.join(exp_dir, "phase_joint", "model", "best_model.pt"))
                    or os.path.exists(os.path.join(exp_dir, "model", "best_model.pt"))
                    or os.path.exists(os.path.join(exp_dir, "phase_canonical", "model", "best_model.pt"))
                    or os.path.exists(os.path.join(exp_dir, "phase_deformation", "model", "best_model.pt")))
        if has_best:
            with open(config_path) as f:
                cfg = json.load(f)
            match = True
            for name, val in combo.items():
                cfg_key = f"search_{name}"
                if cfg.get(cfg_key) != val:
                    match = False
                    break
            if match:
                return True
    return False


def summarize(args, combos):
    """扫描实验目录，汇总结果。"""
    tag = f"{args.model}_search"
    search_dir = os.path.join("train_log", tag)
    if not os.path.exists(search_dir):
        print("No search experiments found.")
        return

    results = []
    for exp_dir in sorted(glob.glob(os.path.join(search_dir, "exp_*"))):
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path) as f:
            cfg = json.load(f)

        # 提取搜索参数
        search_params = {}
        for key in cfg:
            if key.startswith("search_"):
                search_params[key[7:]] = cfg[key]

        # 提取每个 phase 的 best_loss
        phases_info = {}
        for phase in cfg.get("phases", []):
            if phase.get("trained"):
                phases_info[phase["name"]] = phase.get("best_loss", "N/A")

        results.append({
            "exp_dir": exp_dir,
            "params": search_params,
            "phases": phases_info,
        })

    if not results:
        print("No completed experiments found.")
        return

    # 打印表格
    print(f"\n{'='*80}")
    print(f"搜索结果汇总 ({len(results)} 组)")
    print(f"{'='*80}")
    header = f"{'idx':>4}  {'params':45s}"
    # 动态添加 phase 列
    all_phases = set()
    for r in results:
        all_phases.update(r["phases"].keys())
    for p in sorted(all_phases):
        header += f"  {p + '_loss':>12s}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "(default)"
        line = f"{i:>4}  {params_str:45s}"
        for p in sorted(all_phases):
            v = r["phases"].get(p, "-")
            line += f"  {str(v):>12s}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="超参数搜索（调用 train_unified.py）",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        choices=["mstnf", "cmstnf", "ms_scnf", "sdf", "skeleton_sdf"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--canonical_data_dir", type=str, default=None)
    parser.add_argument("--search", type=str, action="append", required=True,
                        help="搜索参数，格式: name=val1,val2,...  可多次 --search")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--consistency", action="store_true")
    # 全局覆盖（与 train_unified.py 一致）
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
    # 搜索控制
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令不执行")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已完成的实验")
    parser.add_argument("--summarize", action="store_true",
                        help="只输出已有搜索结果的汇总表")
    args = parser.parse_args()

    combos = parse_search_specs(args.search)

    # ── 汇总模式 ──
    if args.summarize:
        summarize(args, combos)
        return

    # ── 打印搜索方案 ──
    print(f"模型: {args.model}")
    print(f"数据: {args.data_dir}")
    print(f"搜索方案: {len(combos)} 个组合")
    for i, c in enumerate(combos, 1):
        desc = ", ".join(f"{k}={v}" for k, v in c.items())
        print(f"  [{i}] {desc}")
    print()

    # ── 生成并执行命令 ──
    completed, skipped, failed = 0, 0, 0
    for i, combo in enumerate(combos, 1):
        desc = ", ".join(f"{k}={v}" for k, v in combo.items())

        if args.resume and is_completed(args, combo):
            print(f"[{i}/{len(combos)}] SKIP (已完成): {desc}")
            skipped += 1
            continue

        cmd = build_command(args, combo)
        cmd_str = " ".join(cmd)

        if args.dry_run:
            print(f"[{i}/{len(combos)}] {cmd_str}")
            continue

        print(f"\n[{i}/{len(combos)}] {desc}")
        print(f"  命令: {cmd_str}")
        print("-" * 60)

        try:
            result = subprocess.run(cmd, cwd=os.getcwd())
            if result.returncode == 0:
                completed += 1
                print("  -> 完成")
            else:
                failed += 1
                print(f"  -> 失败 (exit code {result.returncode})")
        except KeyboardInterrupt:
            print(f"\n  中断! 剩余 {len(combos) - i} 个未执行。")
            print(f"  用 --resume 续跑: python {' '.join(sys.argv)} --resume")
            break
        except Exception as e:
            failed += 1
            print(f"  -> 异常: {e}")

    # ── 简要汇总 ──
    if not args.dry_run:
        print(f"\n{'='*40}")
        print(f"完成: {completed}, 跳过: {skipped}, 失败: {failed}")
        if completed > 0 or skipped > 0:
            print(f"\n查看汇总: python {' '.join(sys.argv[:])} --summarize")


if __name__ == "__main__":
    main()
