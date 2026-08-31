"""plot_hysteresis_spectrum.py — HereditaryOperatorModel 迟滞谱可视化。

读 checkpoint → hysteresis_report() → 画两组"迟滞指纹":
  ① PI 密度 μ_c(r_j) = b_{c,j} / r_j（率无关/摩擦型迟滞的谱）
  ② Maxwell 谱 |v_{c,k}| vs τ_k（log 轴，率相关/粘弹型迟滞的谱）
外加静态增益（驱动样条斜率）与残差幅度，构成模型的完整定量读出。

用法:
  python scripts/evaluation/plot_hysteresis_spectrum.py \
      --checkpoint train_log/hereditary/exp_.../phase_hereditary/model/best_model.pt \
      [--out output/spectrum]

输出: <out>/hysteresis_spectrum.png + spectrum_summary.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.utils.model_loader import load_model  # noqa: E402


def main(argv=None):
    pa = argparse.ArgumentParser(description="迟滞谱可视化（PI 密度 + Maxwell 谱）")
    pa.add_argument("--checkpoint", required=True, help="best_model.pt")
    pa.add_argument("--out", default=None,
                    help="输出目录（默认 <ckpt 上三级>/spectrum，即 exp 根下）")
    args = pa.parse_args(argv)

    info = load_model(args.checkpoint, device="cpu")
    model = info["model"]
    assert type(model).__name__ == "HereditaryOperatorModel", (
        f"checkpoint 不是 HereditaryOperatorModel: {type(model).__name__}")

    rep = model.hysteresis_report()
    r = rep["play_thresholds"].numpy()          # (J,)
    b = rep["play_weights"].numpy()             # (C, J)
    taus = rep["maxwell_taus"].numpy()          # (M,)
    v = rep["maxwell_weights"].numpy()          # (C, M)
    C = b.shape[0]
    saved_cfg = info.get("saved_config") or {}
    view = (saved_cfg.get("action_view") or {}).get("model_action_channels")
    ch_labels = [f"ch{c} (raw{view[c]})" if view else f"ch{c}"
                 for c in range(C)]

    if args.out is None:
        exp_root = os.path.dirname(os.path.dirname(
            os.path.dirname(args.checkpoint)))   # .../phase_X/model → exp 根
        args.out = os.path.join(exp_root, "spectrum")
    os.makedirs(args.out, exist_ok=True)

    # ── 画图: 上 PI 密度，下 Maxwell 谱 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    mu = b / r[None, :]                          # PI 密度 μ(r) ∝ b/r
    for c in range(C):
        axes[0].plot(r, mu[c] / (mu[c].max() + 1e-12), "o-",
                     label=ch_labels[c])
        axes[1].plot(taus, v[c], "s-", label=ch_labels[c])
    # 图内标签用英文（matplotlib 默认字体缺 CJK 字形;中文摘要在 txt 里）
    axes[0].set_xlabel("play threshold r")
    axes[0].set_ylabel("normalized PI density mu(r) ~ b/r")
    axes[0].set_title("Rate-independent (friction-type) hysteresis spectrum")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Maxwell time constant tau (s)")
    axes[1].set_ylabel("spectral amplitude |v|")
    axes[1].set_title(f"Rate-dependent (viscoelastic) hysteresis spectrum "
                      f"(dt={rep['dt']:.2f}s)")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("HereditaryOperatorModel hysteresis fingerprint")
    fig.tight_layout()
    png = os.path.join(args.out, "hysteresis_spectrum.png")
    fig.savefig(png, dpi=150)

    # ── 文本摘要（定量读出，论文表格直接可用）──
    lines = [
        f"checkpoint: {args.checkpoint}",
        f"dt = {rep['dt']:.4f} s",
        f"burnin_mode = {saved_cfg.get('burnin_mode', 'rest (pre-F1)')}"
        f"  tau_max = {saved_cfg.get('tau_max', 'n/a')}s",
        "（F3 注意: rest 烧入 + tau_max>episode 时域的旧谱含伪静态 aliasing，"
        "定量结论须以 equilibrium 烧入的重训谱为准）",
        "",
        "PI play 密度 μ_c(r_j) = b/r（率无关迟滞容量分布）:",
        f"  r grid: {np.array2string(r, precision=3)}",
    ]
    for c in range(C):
        lines.append(f"  {ch_labels[c]}: b={np.array2string(b[c], precision=3)}")
    lines += [
        "",
        "Maxwell 谱 |v_c(τ_k)|（率相关迟滞容量分布）:",
        f"  tau grid (s): {np.array2string(taus, precision=2)}",
    ]
    for c in range(C):
        lines.append(f"  {ch_labels[c]}: |v|={np.array2string(v[c], precision=3)}")
    # 谱质量的粗判据（设计文档 E1 的谱版）
    total_play = b.sum()
    total_maxwell = v.sum()
    if total_play + total_maxwell > 0:
        play_share = total_play / (total_play + total_maxwell)
        lines += [
            "",
            f"play/maxwell 容量占比: {play_share:.2f} / {1 - play_share:.2f}",
            "（读法: play 份额高 → 迟滞以率无关摩擦为主；",
            "  Maxwell 份额高 → 以粘弹蠕变为主；E1 变速率实验做最终裁决）",
        ]
    txt = os.path.join(args.out, "spectrum_summary.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"spectrum → {png}")
    print(f"summary  → {txt}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
