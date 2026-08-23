"""平面形态与避障动作逆规划入口。

对任意 action_dim(1..6):给定离线起始骨架、目标骨架或末端目标与圆障碍，
用工作台 planner 求 K 步动作序列 → 存 plan JSON + predicted_states.npz(kPa 动作,
可经过工作台 Arm 后执行)。输出使用 safety kPa、K_safe、Preflight 和部署 manifest 合同。

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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="避障逆规划(工作台 planner 薄包装)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="val npz 目录(建 anchor 用)")
    parser.add_argument("--t-init", type=int, required=True, help="起始骨架帧索引")
    parser.add_argument("--target-x", type=float)
    parser.add_argument("--target-y", type=float)
    parser.add_argument("--target-radius", type=float, default=5.0)
    parser.add_argument("--target-node", type=int, default=0, help="末端 node(默认 0)")
    parser.add_argument("--target-frame", type=int, default=None,
                        help="从transition NPZ读取该帧的完整N节点目标骨架")
    parser.add_argument("--target-data", default=None,
                        help="目标骨架NPZ；默认与--data-dir中的anchor NPZ相同")
    parser.add_argument("--target-tolerance-px", type=float, default=4.0)
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
    parser.add_argument("--mock-execute", action="store_true",
                        help="Preflight通过后用Mock transport验证逐拍ACK执行并保存execution.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from real_validation.contracts.io import atomic_write_json
    from real_validation.contracts.models import SafetyPolicy, Scene, ScenePrimitive
    from real_validation.contracts.plan_io import write_actions6_csv
    from real_validation.execution.metrics import evaluate_plan_scene
    from real_validation.execution.preflight import validate_plan
    from real_validation.hardware.profile import required_groups_for_channels
    from real_validation.planning.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
    from real_validation.planning.planner_service import expand_model_actions
    from real_validation.planning.units import model_to_kPa
    from src.evaluation.diameter_scale import resolve_diameter_scale
    # 直接走轻量 offline_anchor(纯 contracts.models + numpy),避免经 runtime.anchors
    # 门面 eager 拉入 live_anchor → perception → cv2 依赖图。
    from real_validation.runtime.offline_anchor import anchor_from_npz

    # 1. 加载模型 + manifest(自动找同目录 deploy_manifest.json)
    from real_validation.runtime.model_runtime import ModelRuntime
    runtime = ModelRuntime(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    descriptor = runtime.descriptor
    if descriptor.action_scale_kpa is None:
        parser.error(f"checkpoint 缺 deploy_manifest(部署契约);请先跑 build_deploy_manifest.py")

    # 2. 从 val npz 起始帧建 anchor(离线,需完整 H 历史)
    files = sorted(__import__("glob").glob(os.path.join(args.data_dir, "*.npz")))
    if not files:
        parser.error(f"{args.data_dir} 无 npz")
    anchor = anchor_from_npz(files[0], args.t_init, descriptor, runtime.model, padding="reject")

    # 3. Scene:完整目标骨架或末端目标 + 圆障碍(model坐标=源相机像素)
    target_npz = args.target_data or files[0]
    if args.target_frame is not None:
        with np.load(target_npz) as target_data:
            positions = np.asarray(target_data["positions"], dtype=np.float32)
        if args.target_frame < 0 or args.target_frame >= len(positions):
            parser.error(f"target-frame超出0..{len(positions) - 1}")
        target_nodes = positions[args.target_frame]
        if target_nodes.shape == (3, descriptor.n_nodes):
            target_nodes = target_nodes.T
        if target_nodes.shape[0] != descriptor.n_nodes:
            parser.error("目标骨架节点数与模型不一致")
        primitives = [ScenePrimitive(
            "target_skeleton", "model",
            {"nodes": target_nodes[:, :2].tolist(),
             "tolerance_px": args.target_tolerance_px}, name="shape_target")]
    else:
        if args.target_x is None or args.target_y is None:
            parser.error("请提供--target-frame，或同时提供--target-x/--target-y")
        primitives = [ScenePrimitive(
            "target_circle", "model",
            {"xy": [args.target_x, args.target_y], "radius": args.target_radius,
             "node": args.target_node}, name="tip_target")]
    with np.load(target_npz) as target_data:
        diameter_scale = resolve_diameter_scale(target_data, os.path.dirname(target_npz))
    for obs in [o for o in args.obstacle.split("|") if o]:
        cx, cy, r = (float(v) for v in obs.split(","))
        primitives.append(ScenePrimitive(
            "obstacle_circle", "model", {"center": [cx, cy], "radius": r}, name="obs"))
    scene = Scene("avoidance", tuple(primitives))

    # 4. Safety：初始压力由anchor末个真实动作恢复，保证首拍速率约束接续当前状态。
    def _vec(s): return tuple(float(v) for v in s.split(","))
    channel_map = descriptor.channel_map or tuple(range(descriptor.action_dim))
    initial_model_kpa = model_to_kPa(
        np.asarray(anchor.action_history[-1], dtype=np.float32),
        action_scale_kpa=descriptor.action_scale_kpa,
        action_norm_factor=runtime.info["norm_factor"])
    initial6 = expand_model_actions(
        [initial_model_kpa], channel_map, descriptor.channel_equalities,
        descriptor.channel_source6)[0]
    required_groups = required_groups_for_channels(
        channel_map, descriptor.channel_equalities, descriptor.channel_source6)
    safety = SafetyPolicy(
        pressure_min6=_vec(args.safety_min), pressure_max6=_vec(args.safety_max),
        rise_rate6=_vec(args.rise), fall_rate6=_vec(args.fall),
        initial_action6=initial6, required_groups=required_groups)

    # 5. 规划
    if (args.k is None) == (not args.auto_k):
        parser.error("--k 与 --auto-k 必须恰有其一")
    config = ShootingConfig(
        horizon=args.k, auto_k=args.auto_k, k_min=args.k_min, k_max=args.k_max,
        n_iter=args.n_iter, n_restarts=args.n_restarts)
    step_interval_s = descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s or 0.2
    plan = OpenLoopShootingPlanner(runtime).plan(
        anchor=anchor, scene=scene, safety=safety, channel_map=tuple(channel_map),
        step_interval_s=step_interval_s, output_dir=args.out, config=config)

    # 6. Preflight与落盘；真机执行继续由GUI的Arm门控制。
    preflight = validate_plan(
        plan, descriptor, anchor, scene, safety,
        train_dt_s=descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s)
    preflight.require_ok()
    os.makedirs(args.out, exist_ok=True)
    atomic_write_json(os.path.join(args.out, "plan.json"), plan.to_dict())
    atomic_write_json(os.path.join(args.out, "anchor.json"), anchor.to_dict())
    atomic_write_json(os.path.join(args.out, "scene.json"), scene.to_dict())
    atomic_write_json(os.path.join(args.out, "safety.json"), safety.to_dict())
    write_actions6_csv(plan, os.path.join(args.out, "planned_actions6.csv"))
    if args.mock_execute:
        from real_validation.execution.executor import MockCommandTransport, PlanExecutor
        PlanExecutor(MockCommandTransport(), safety).execute(
            plan, os.path.join(args.out, "execution.csv"))
    meta = plan.metadata
    print(f"plan 写入 {args.out}/plan.json")
    print(f"  K={meta.get('k_effective')} auto_k={meta.get('auto_k')} gap={meta.get('auto_k_gap_px')}px")
    print(f"  规划耗时 {meta.get('duration_s', 0):.1f}s  clearance={meta.get('predicted_min_obstacle_clearance')}")
    print(f"  动作数 {plan.horizon}, 步长 {step_interval_s:.3f}s(训练 Δt)")
    predicted_residual_px = plan.loss_terms.get(
        "predicted_terminal_target_residual_px", float("nan"))
    with np.load(os.path.join(args.out, plan.predicted_states_path)) as predicted_data:
        predicted_states = np.asarray(predicted_data["states_model"], dtype=np.float32)
    predicted_scene = evaluate_plan_scene(
        predicted_states, scene, tip_node=0, mm_per_px=diameter_scale.mm_per_px)
    print("  命令安全预检=PASS")
    print(f"  模型预测目标: kind={plan.metadata.get('target_kind')}, "
          f"residual={predicted_residual_px:.3f}px, "
          f"reached={predicted_scene.get('predicted_target_success')}")
    print(f"  直径尺度={diameter_scale.mm_per_px:.6f}mm/px, "
          f"predicted_terminal_target_residual≈"
          f"{predicted_residual_px * diameter_scale.mm_per_px:.3f}mm")
    print(f"  六通道初始压力={tuple(round(v, 2) for v in initial6)} kPa")
    if args.mock_execute:
        print(f"  Mock执行完成: {args.out}/execution.csv")


if __name__ == "__main__":
    main()
