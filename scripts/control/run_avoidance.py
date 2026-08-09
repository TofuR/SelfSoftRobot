"""避障逆规划实验入口(CLI 薄脚本,包 real_validation.openloop_planner)。

对任意 action_dim(1/3/6 腔道):给定起始骨架(离线 npz 帧)或目标点 + 圆障碍,
用工作台 planner 求 K 步动作序列 → 存 plan JSON + predicted_states.npz(kPa 动作,
可直接喂实机执行)。相比 inverse_plan.py:输出是 safety kPa 动作(非归一化 a_plan)、
有 K_safe/preflight 门、支持 manifest。

Usage:
  python scripts/control/run_avoidance.py \
      --checkpoint train_log/open_loop_transition/<exp>/phase_*/model/best_model.pt \
      --data-dir data/real_seq/<seq>_n15_sam2_clean/val \
      --t-init 500 \
      --target-x 330 --target-y 200 --target-radius 5 \
      --obstacle '300,180,15' \
      --auto-k --out train_log/open_loop_transition/<exp>/eval_avoid
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="避障逆规划(工作台 planner 薄包装)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="val npz 目录(建 anchor 用)")
    parser.add_argument("--t-init", type=int, required=True, help="起始骨架帧索引")
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-radius", type=float, default=5.0)
    parser.add_argument("--target-node", type=int, default=0, help="末端 node(默认 0)")
    parser.add_argument("--obstacle", default="", help="圆障碍 'cx,cy,r',多个用 | 分隔")
    parser.add_argument("--k", type=int, default=None, help="固定 K(与 --auto-k 互斥)")
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--n-restarts", type=int, default=4)
    parser.add_argument("--safety-min", default="0,0,0,0,0,0", help="每通道 min kPa")
    parser.add_argument("--safety-max", default="150,150,150,150,150,150", help="每通道 max kPa")
    parser.add_argument("--rise", default="100,100,100,100,100,100", help="每通道上升 kPa/s")
    parser.add_argument("--fall", default="100,100,100,100,100,100", help="每通道下降 kPa/s")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from real_validation.models import (ActionPlan, Anchor, SafetyPolicy, Scene,
                                        ScenePrimitive)
    from real_validation.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
    from real_validation.offline_anchor import anchor_from_npz

    # 1. 加载模型 + manifest(自动找同目录 deploy_manifest.json)
    from real_validation.model_runtime import ModelRuntime
    runtime = ModelRuntime(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    descriptor = runtime.descriptor
    if descriptor.action_scale_kpa is None:
        parser.error(f"checkpoint 缺 deploy_manifest(部署契约);请先跑 build_deploy_manifest.py")

    # 2. 从 val npz 起始帧建 anchor(离线,需完整 H 历史)
    files = sorted(__import__("glob").glob(os.path.join(args.data_dir, "*.npz")))
    if not files:
        parser.error(f"{args.data_dir} 无 npz")
    anchor = anchor_from_npz(files[0], args.t_init, descriptor, runtime.model, padding="reject")

    # 3. Scene:末端目标点/圆 + 圆障碍(model 坐标 = 像素)
    primitives = [ScenePrimitive(
        "target_circle", "model",
        {"xy": [args.target_x, args.target_y], "radius": args.target_radius,
         "node": args.target_node}, name="target")]
    for obs in [o for o in args.obstacle.split("|") if o]:
        cx, cy, r = (float(v) for v in obs.split(","))
        primitives.append(ScenePrimitive(
            "obstacle_circle", "model", {"center": [cx, cy], "radius": r}, name="obs"))
    scene = Scene("avoidance", tuple(primitives))

    # 4. Safety(kPa 边界;3 腔道填前 3,其余 0)
    def _vec(s): return tuple(float(v) for v in s.split(","))
    safety = SafetyPolicy(
        pressure_min6=_vec(args.safety_min), pressure_max6=_vec(args.safety_max),
        rise_rate6=_vec(args.rise), fall_rate6=_vec(args.fall),
        initial_action6=(0.0,) * 6, required_groups=(1,))

    # 5. 规划
    if (args.k is None) == (not args.auto_k):
        parser.error("--k 与 --auto-k 必须恰有其一")
    config = ShootingConfig(
        horizon=args.k, auto_k=args.auto_k, k_min=args.k_min, k_max=args.k_max,
        n_iter=args.n_iter, n_restarts=args.n_restarts)
    channel_map = descriptor.channel_map or tuple(range(descriptor.action_dim))
    step_interval_s = descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s or 0.2
    plan = OpenLoopShootingPlanner(runtime).plan(
        anchor=anchor, scene=scene, safety=safety, channel_map=tuple(channel_map),
        step_interval_s=step_interval_s, output_dir=args.out, config=config)

    # 6. 落盘
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "plan.json"), "w", encoding="utf-8") as stream:
        json.dump(plan.to_dict(), stream, ensure_ascii=False, indent=2)
    meta = plan.metadata
    print(f"plan 写入 {args.out}/plan.json")
    print(f"  K={meta.get('k_effective')} auto_k={meta.get('auto_k')} gap={meta.get('auto_k_gap_px')}px")
    print(f"  规划耗时 {meta.get('duration_s', 0):.1f}s  clearance={meta.get('predicted_min_obstacle_clearance')}")
    print(f"  动作数 {plan.horizon}, 步长 {step_interval_s:.3f}s(训练 Δt)")
    print(f"  压力范围 kPa: ch0={safety.pressure_min6[0]}..{safety.pressure_max6[0]}")


if __name__ == "__main__":
    main()
