"""三腔道激励协议生成器 → 6 列 actions6.csv(供 Replay 采集)。

软体臂多腔道自建模需要覆盖三种激励:
  1. 单腔道正交基:每个腔道独立 ramp→hold→释放(识别基本作用方向)
  2. 成对组合:两腔同向/反向(同段腔道竞争、耦合)
  3. 三腔协同:三腔同向/一升一降(跨段协同)

输出 actions6.csv(t_sec, c0..c5),与 real_capture 的 Replay 模式兼容
(valve_control.load_action_sequence:6 列、时间严格递增)。未驱动通道恒 0。
压力变化保持连续(每步增量受限),rise/fall 上限由采集端 limiter 执行时再限。

Usage:
  python scripts/real/gen_3chamber_excitation.py \
      --channels 0,1,2 --hi6 150,150,150 --dt 0.2 \
      --hold 10 --ramp 15 --out actions6.csv
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _ramp(lo, hi, steps):
    """线性 0→1 序列(含端点),长度 steps。"""
    if steps <= 1:
        return [1.0]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def _segment_rise_hold_fall(hi, ramp_steps, hold_steps):
    """单腔:0→hi(ramp)→hi(hold)→0(ramp)→0(hold)。返回 [0..1] 归一化序列。"""
    return (_ramp(0.0, 1.0, ramp_steps)
            + [1.0] * hold_steps
            + _ramp(1.0, 0.0, ramp_steps)
            + [0.0] * hold_steps)


def _segment_dual(hi_a, hi_b, mode, ramp_steps, hold_steps):
    """成对:同向(同升同降)或反向(一个升一个降)。返回 [(a,b), ...] 归一化。"""
    single = _segment_rise_hold_fall(1.0, ramp_steps, hold_steps)
    if mode == "same":
        return [(v, v) for v in single]
    if mode == "opposite":
        return [(v, 1.0 - v) for v in single]
    raise ValueError(mode)


def _segment_triple(mode, ramp_steps, hold_steps):
    """三腔:同向 / ch0 升 ch1 降 ch2 中 / ch0 降 ch1 升 ch2 中。返回 [(a,b,c), ...]。"""
    single = _segment_rise_hold_fall(1.0, ramp_steps, hold_steps)
    if mode == "same":
        return [(v, v, v) for v in single]
    if mode == "a_up_b_down":
        return [(v, 1.0 - v, 0.5) for v in single]
    if mode == "a_down_b_up":
        return [(1.0 - v, v, 0.5) for v in single]
    raise ValueError(mode)


def generate(channels, hi6, dt, ramp_steps, hold_steps):
    """返回 [(t_sec, c0..c5)]。hi6: 每通道 kPa 上限(长度=6)。"""
    chs = list(channels)
    rows = []
    t = 0.0

    def append(values6):
        nonlocal t
        rows.append([f"{t:.6f}"] + [f"{v:.4f}" for v in values6])
        t += dt

    # 静息:三腔 0 起步(Replay 从静息开始,避免首拍跳变)
    append([0.0] * 6)

    # 阶段 1:单腔正交基
    for ch in chs:
        seg = _segment_rise_hold_fall(hi6[ch], ramp_steps, hold_steps)
        for v in seg:
            values = [0.0] * 6
            values[ch] = v
            append(values)

    # 阶段 2:成对组合
    for i, a in enumerate(chs):
        for b in chs[i + 1:]:
            for mode in ("same", "opposite"):
                seg = _segment_dual(hi6[a], hi6[b], mode, ramp_steps, hold_steps)
                for (va, vb) in seg:
                    values = [0.0] * 6
                    values[a] = va * hi6[a]
                    values[b] = vb * hi6[b]
                    append(values)

    # 阶段 3:三腔协同
    for mode in ("same", "a_up_b_down", "a_down_b_up"):
        seg = _segment_triple(mode, ramp_steps, hold_steps)
        for (va, vb, vc) in seg:
            values = [0.0] * 6
            values[chs[0]] = va * hi6[chs[0]]
            values[chs[1]] = vb * hi6[chs[1]]
            values[chs[2]] = vc * hi6[chs[2]]
            append(values)

    # 收尾:归零
    append([0.0] * 6)
    return rows


def main():
    parser = argparse.ArgumentParser(description="生成三腔道激励 actions6.csv")
    parser.add_argument("--channels", default="0,1,2", help="驱动通道,逗号分隔")
    parser.add_argument("--hi6", default="150,150,150", help="每腔道上限 kPa,逗号分隔")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--ramp", type=int, default=15, help="每段 ramp 步数")
    parser.add_argument("--hold", type=int, default=10, help="每段保持步数")
    parser.add_argument("--out", default="actions6.csv")
    args = parser.parse_args()

    channels = [int(c) for c in args.channels.split(",") if c.strip()]
    hi6_vals = [float(v) for v in args.hi6.split(",") if v.strip()]
    if len(channels) != len(hi6_vals):
        parser.error("--channels 与 --hi6 长度必须一致(每通道一个上限)")
    if not channels:
        parser.error("至少需要一个驱动通道")

    hi6 = [0.0] * 6
    for ch, v in zip(channels, hi6_vals):
        if not (0 <= ch <= 5):
            parser.error(f"通道 {ch} 超出 0..5")
        hi6[ch] = v

    rows = generate(channels, hi6, args.dt, args.ramp, args.hold)

    with open(args.out, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_sec", "c0", "c1", "c2", "c3", "c4", "c5"])
        writer.writerows(rows)

    # 自检:时间严格递增、值在范围内、末行归零
    times = [float(r[0]) for r in rows]
    assert all(b > a for a, b in zip(times, times[1:])), "时间必须严格递增"
    values = [[float(v) for v in r[1:]] for r in rows]
    assert all(0.0 <= v <= (hi6[i] or 0.0) + 1e-6 for row in values for i, v in enumerate(row)), \
        "压力超出 hi6"
    assert all(abs(v) < 1e-6 for v in values[-1]), "末行必须归零"
    n_steps = len(rows) - 1   # 减去首行静息
    print(f"写入 {args.out}: {n_steps} 步, {times[-1]:.2f}s, 通道={channels}")
    print(f"  单腔{len(channels)}段 + 成对{len(channels)*(len(channels)-1)//2*2}段 + 三腔3段")


if __name__ == "__main__":
    main()
