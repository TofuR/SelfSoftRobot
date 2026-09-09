#!/usr/bin/env python3
"""Plot completed sequential feedback benchmarks without rerunning inference."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True, help="new report directory")
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    if not (run / "COMPLETE").exists():
        parser.error("benchmark must be complete")
    summaries = json.loads((run / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    names = {"torch_b": "Original B", "batched_b": "Batched B",
             "fast_b": "Analytic B", "cached_a": "Cached A"}
    colors = ["#9c537c", "#d98b26", "#147e89", "#4864b0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    table = []
    for i, summary in enumerate(summaries):
        method = summary["method"]
        with (run / method / "per_frame.csv").open() as source:
            rows = [r for r in csv.DictReader(source) if int(r["horizon"]) > 0]
        axes[0].plot([int(r["horizon"]) for r in rows],
                     [float(r["feedback_ms"]) for r in rows],
                     label=names[method], color=colors[i % len(colors)])
        timing = summary["timing"]["feedback_ms"]
        axes[1].plot([timing["p50"], timing["max"]], [i, i], color=colors[i % len(colors)], lw=3)
        axes[1].scatter([timing["p50"], timing["p95"], timing["max"]], [i]*3,
                        color=colors[i % len(colors)], s=[25, 65, 25])
        axes[1].annotate(f'{timing["p50"]:.1f} / {timing["p95"]:.1f} / {timing["max"]:.1f}',
                         (timing["p95"], i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        table.append(f'<tr><td>{html.escape(names[method])}</td>'
                     f'<td>{timing["p50"]:.1f}</td><td>{timing["p95"]:.1f}</td>'
                     f'<td>{timing["max"]:.1f}</td><td>{timing["over_100ms"]}/{timing["samples"]}</td>'
                     f'<td>{summary["suffix_updates"]}/{timing["samples"]}</td></tr>')
    for limit, style in ((100, "--"), (200, ":")):
        axes[0].axhline(limit, color="#666666", ls=style, lw=1)
        axes[1].axvline(limit, color="#666666", ls=style, lw=1)
    axes[0].set(yscale="log", xlabel="Remaining horizon (steps)", ylabel="Feedback computation (ms)",
                title="Sequential replay; thresholds at 100 and 200 ms")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[1].set(xscale="log", xlabel="Feedback computation (ms)",
                yticks=range(len(summaries)), yticklabels=[names[s["method"]] for s in summaries],
                title="P50 / P95 / maximum (ms)", ylim=(-.6, len(summaries)-.3))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(out / "latency.png", dpi=180)
    plt.close(fig)
    encoded = base64.b64encode((out / "latency.png").read_bytes()).decode()
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>部分观测反馈计算耗时</title><style>body{max-width:1100px;margin:32px auto;font-family:system-ui;line-height:1.7}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}img{width:100%}</style>
<h1>部分观测反馈计算耗时</h1><p>同一模型、80 帧高变化窗口、固定 56×56 像素方块。
每种方法独立串行运行，统计预热后 79 次具有剩余动作的完整反馈计算。</p>
<p>计时从内存中的原图开始，包括方块覆盖、预测、边缘提取、状态校正、后缀修订及压力约束检查。
不包含初始规划、A 预计算、相机曝光/图像年龄、磁盘读取、诊断绘图和阀门通信。</p>
<table><tr><th>方法</th><th>P50 ms</th><th>P95 ms</th><th>最大 ms</th><th>≥100ms</th><th>后缀接受</th></tr>'''
    page += "".join(table) + f'</table><img alt="反馈耗时对照" src="data:image/png;base64,{encoded}">'
    page += '''<p>A 包括在线投影、真实非线性校验；缓存失效或候选未下降时保留旧计划。
接受次数不是物理控制成功率。录制图像不响应修订压力，本实验不能给出真实闭环精度。</p></html>'''
    (out / "report.html").write_text(page)
    (out / "source.json").write_text(json.dumps({"run": str(run.resolve())}, indent=2)+"\n")
    (out / "COMPLETE").touch()


if __name__ == "__main__":
    main()
