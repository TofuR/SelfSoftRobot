"""Export a synchronized visualization of one recorded-data inverse plan.

The input directory is the immutable artifact bundle produced by
``scripts/control/run_avoidance.py``: anchor, scene, plan, and predicted states.
The exporter reads those files directly and does not rerun the planner or model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.diameter_scale import resolve_diameter_scale  # noqa: E402
from real_validation.perception.coordinates import SkeletonFrameTransform  # noqa: E402


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ("#2166ac", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _parse_anchor_source(anchor: dict) -> tuple[Path, int]:
    match = re.fullmatch(r"(.+)#frame=(\d+)", str(anchor.get("source", "")))
    if not match:
        raise ValueError("anchor source需要包含transition NPZ路径和frame索引")
    return Path(match.group(1)), int(match.group(2))


def _target_nodes(scene: dict) -> np.ndarray:
    targets = [item for item in scene.get("primitives", [])
               if item.get("kind") == "target_skeleton"]
    if len(targets) != 1:
        raise ValueError("展示入口需要一个target_skeleton")
    nodes = np.asarray(targets[0]["geometry"]["nodes"], dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError("target_skeleton节点格式无效")
    return nodes[:, :2]


def _infer_target_frame(plan_dir: Path) -> int | None:
    match = re.search(r"val\d+_to(\d+)", plan_dir.name)
    return int(match.group(1)) if match else None


def _raw_image_path(npz_path: Path, split_frame: int) -> Path | None:
    match = re.search(r"seq_\d{8}_\d{6}", npz_path.name)
    if not match:
        return None
    sequence = match.group(0)
    offset = 0
    if npz_path.parent.name == "val":
        train_path = npz_path.parent.parent / "train" / f"{sequence}_train.npz"
        if train_path.is_file():
            with np.load(train_path, allow_pickle=False) as train:
                offset = int(train["positions"].shape[0])
    return PROJECT_ROOT / "real_capture" / "data" / "raw" / sequence / \
        "cam0" / f"{offset + split_frame:05d}.png"


def _load(plan_dir: Path, target_frame: int | None) -> dict:
    plan = _read_json(plan_dir / "plan.json")
    anchor = _read_json(plan_dir / "anchor.json")
    scene = _read_json(plan_dir / "scene.json")
    npz_path, init_frame = _parse_anchor_source(anchor)
    if not npz_path.is_absolute():
        npz_path = (PROJECT_ROOT / npz_path).resolve()
    with np.load(npz_path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        if not 0 <= init_frame < len(positions):
            raise IndexError("anchor frame超出transition NPZ")
        initial = positions[init_frame].T
        scale = resolve_diameter_scale(data, str(npz_path.parent))
        transform = (SkeletonFrameTransform.from_dict(json.loads(
            str(data["skeleton_frame_transform"].item())))
            if "skeleton_frame_transform" in data else None)
    states_path = Path(plan["predicted_states_path"])
    if not states_path.is_absolute():
        states_path = plan_dir / states_path
    with np.load(states_path, allow_pickle=False) as states_file:
        predicted = np.asarray(states_file["states_model"], dtype=np.float32)
        state_unit = (str(states_file["state_length_unit"].item())
                      if "state_length_unit" in states_file else
                      str(plan.get("metadata", {}).get("state_length_unit", "px")))
    target = _target_nodes(scene)
    actions = np.asarray(plan["actions6"], dtype=np.float32)
    if predicted.ndim != 3 or predicted.shape[:2] != (len(actions), len(target)):
        raise ValueError("predicted_states、actions6与target_skeleton长度不一致")
    frame = target_frame if target_frame is not None else _infer_target_frame(plan_dir)
    if state_unit not in {"px", "mm"}:
        raise ValueError(f"未知状态长度单位: {state_unit}")
    state_to_mm = 1.0 if state_unit == "mm" else float(scale.mm_per_px)
    if transform is not None and state_unit == "mm":
        initial_camera = transform.model_to_camera(initial[:, :2])
        target_camera = transform.model_to_camera(target)
        predicted_camera = transform.model_to_camera(predicted[:, :, :2])
    elif state_unit == "px":
        initial_camera, target_camera = initial[:, :2], target
        predicted_camera = predicted[:, :, :2]
    else:
        initial_camera = target_camera = predicted_camera = None
    return {
        "plan": plan,
        "anchor": anchor,
        "scene": scene,
        "npz_path": npz_path,
        "init_frame": init_frame,
        "target_frame": frame,
        "initial": initial[:, :2],
        "target": target,
        "predicted": predicted[:, :, :2],
        "initial_camera": initial_camera,
        "target_camera": target_camera,
        "predicted_camera": predicted_camera,
        "actions": actions,
        "state_unit": state_unit,
        "state_to_mm": state_to_mm,
        "mm_per_px": float(scale.mm_per_px),
        "diameter_px": float(scale.diameter_px),
        "diameter_source": scale.source,
    }


def _metrics(data: dict) -> dict:
    initial_node = np.linalg.norm(data["initial"] - data["target"], axis=1)
    residual_nodes = np.linalg.norm(
        data["predicted"] - data["target"][None], axis=2)
    residual = residual_nodes.mean(axis=1)
    terminal_node = residual_nodes[-1]
    state_to_mm = data["state_to_mm"]
    unit = data["state_unit"]
    actions = data["actions"]
    return {
        "schema_version": 2,
        "scope": "offline learned-model inverse planning on recorded 5 Hz skeleton data",
        "state_length_unit": unit,
        "initial_frame": data["init_frame"],
        "target_frame": data["target_frame"],
        "horizon_steps": len(actions),
        "step_interval_s": float(data["plan"]["step_interval_s"]),
        "horizon_duration_s": float(len(actions) * data["plan"]["step_interval_s"]),
        "planner_duration_s": float(data["plan"].get("metadata", {}).get(
            "duration_s", float("nan"))),
        "initial_target_node_mean": float(initial_node.mean()),
        "initial_target_tip": float(initial_node[-1]),
        "terminal_target_node_mean": float(terminal_node.mean()),
        "terminal_target_tip": float(terminal_node[-1]),
        "best_path_node_mean": float(residual.min()),
        "best_path_step": int(residual.argmin() + 1),
        "residual_reduction_percent": float(
            100.0 * (1.0 - terminal_node.mean() / initial_node.mean())),
        "state_to_mm": state_to_mm,
        "initial_target_node_mean_mm": float(initial_node.mean() * state_to_mm),
        "initial_target_tip_mm": float(initial_node[-1] * state_to_mm),
        "terminal_target_node_mean_mm": float(terminal_node.mean() * state_to_mm),
        "terminal_target_tip_mm": float(terminal_node[-1] * state_to_mm),
        "best_path_node_mean_mm": float(residual.min() * state_to_mm),
        "action_min6_kpa": actions.min(axis=0).tolist(),
        "action_max6_kpa": actions.max(axis=0).tolist(),
        "action_max_step_delta6_kpa": np.abs(np.diff(actions, axis=0)).max(axis=0).tolist(),
        "channel_source6": data["plan"].get("channel_source6"),
        "channel_equalities": data["plan"].get("channel_equalities"),
        "metric_semantics": {
            "terminal_target": "model-predicted terminal skeleton versus recorded target skeleton",
            "millimeter_values": ("native robot_planar_mm_v1 coordinates" if unit == "mm"
                                   else "16 mm robot-diameter scale conversion"),
        },
    }


def _action_series(data: dict):
    sources = data["plan"].get("channel_source6") or list(range(6))
    leaders = []
    for source in sources:
        source = int(source)
        if source not in leaders:
            leaders.append(source)
    result = []
    for leader in leaders:
        members = [index for index, source in enumerate(sources)
                   if int(source) == leader]
        label = " = ".join(f"ch{index}" for index in members)
        result.append((leader, label))
    return result


def _limits(data: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate((data["initial"][None], data["target"][None],
                             data["predicted"]), axis=0).reshape(-1, 2)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = np.maximum((hi - lo) * 0.12, 8.0)
    return lo - pad, hi + pad


def _setup_figure(data: dict, metrics: dict):
    figure = plt.figure(figsize=(14.4, 8.1), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.08, 1.0))
    shape_ax = figure.add_subplot(grid[:, 0])
    action_ax = figure.add_subplot(grid[0, 1])
    residual_ax = figure.add_subplot(grid[1, 1])
    figure.suptitle(
        "5 Hz双段软体机器人：记录形态目标的离线逆规划",
        fontsize=16, fontweight="bold")
    lo, hi = _limits(data)
    shape_ax.set_xlim(lo[0], hi[0])
    if data["state_unit"] == "px":
        shape_ax.set_ylim(hi[1], lo[1])
    else:
        shape_ax.set_ylim(lo[1], hi[1])
    shape_ax.set_aspect("equal", adjustable="box")
    shape_ax.grid(alpha=0.2)
    unit = data["state_unit"]
    shape_ax.set_xlabel(f"机器人横向坐标 [{unit}]")
    shape_ax.set_ylabel(f"机器人轴向坐标 [{unit}]")
    action_ax.grid(alpha=0.2)
    action_ax.set_ylabel("计划压力 [kPa]")
    action_ax.set_xlabel("控制步 k")
    residual_ax.grid(alpha=0.2)
    residual_ax.set_ylabel(f"全节点平均目标残差 [{unit}]")
    residual_ax.set_xlabel("控制步 k")
    return figure, shape_ax, action_ax, residual_ax


def _draw_reference(shape_ax, data: dict):
    shape_ax.plot(data["initial"][:, 0], data["initial"][:, 1], "o-",
                  color="#377eb8", lw=2.2, ms=4.5, label="记录起始形态")
    shape_ax.plot(data["target"][:, 0], data["target"][:, 1], "s--",
                  color="#4daf4a", lw=2.4, ms=5, label="记录目标形态")


def export_summary(data: dict, metrics: dict, output: Path) -> None:
    figure, shape_ax, action_ax, residual_ax = _setup_figure(data, metrics)
    _draw_reference(shape_ax, data)
    cmap = plt.cm.plasma
    steps = np.arange(1, len(data["predicted"]) + 1)
    for index in range(4, len(data["predicted"]), 5):
        state = data["predicted"][index]
        shape_ax.plot(state[:, 0], state[:, 1], "-", color=cmap(index / len(steps)),
                      lw=1.1, alpha=0.42)
    terminal = data["predicted"][-1]
    shape_ax.plot(terminal[:, 0], terminal[:, 1], "^-", color="#e41a1c",
                  lw=2.6, ms=5.5, label="模型预测终态")
    tip_path = data["predicted"][:, -1]
    shape_ax.plot(tip_path[:, 0], tip_path[:, 1], color="#984ea3", lw=2,
                  alpha=0.8, label="预测末端路径")
    shape_ax.legend(loc="best", fontsize=9)
    shape_ax.set_title(
        f"初始→目标 {metrics['initial_target_node_mean']:.1f}{data['state_unit']}；"
        f"预测终态残差 {metrics['terminal_target_node_mean']:.1f}{data['state_unit']}")

    for (column, label), color in zip(_action_series(data), COLORS):
        action_ax.plot(steps, data["actions"][:, column], color=color,
                       lw=2.1, label=label)
    action_ax.legend(ncol=2, fontsize=9)
    action_ax.set_title("优化得到的六通道硬件压力（等值通道合并显示）")

    residual = np.linalg.norm(
        data["predicted"] - data["target"][None], axis=2).mean(axis=1)
    residual_ax.plot(steps, residual, color="#e41a1c", lw=2.4,
                     label="模型预测→记录目标")
    residual_ax.axhline(metrics["initial_target_node_mean"], color="#377eb8",
                        ls="--", lw=1.4, label="初始形态→目标")
    residual_ax.scatter([metrics["best_path_step"]],
                        [metrics["best_path_node_mean"]], color="#4daf4a",
                        zorder=3, label=f"路径最小值 k={metrics['best_path_step']}")
    residual_ax.legend(fontsize=9)
    residual_ax.set_title(
        f"终态下降 {metrics['residual_reduction_percent']:.1f}%；"
        f"终态残差 {metrics['terminal_target_node_mean_mm']:.1f}mm")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def export_animation(data: dict, metrics: dict, gif_path: Path,
                     mp4_path: Path, fps: int) -> None:
    figure, shape_ax, action_ax, residual_ax = _setup_figure(data, metrics)
    _draw_reference(shape_ax, data)
    steps = np.arange(1, len(data["predicted"]) + 1)
    for (column, label), color in zip(_action_series(data), COLORS):
        action_ax.plot(steps, data["actions"][:, column], color=color,
                       lw=2.0, label=label)
    action_ax.legend(ncol=2, fontsize=8)
    action_ax.set_xlim(1, len(steps))
    action_ax.set_ylim(0, max(155, float(data["actions"].max()) * 1.08))
    residual = np.linalg.norm(
        data["predicted"] - data["target"][None], axis=2).mean(axis=1)
    residual_ax.plot(steps, residual, color="#e41a1c", lw=2.3)
    residual_ax.axhline(metrics["initial_target_node_mean"], color="#377eb8",
                        ls="--", lw=1.3)
    residual_ax.set_xlim(1, len(steps))
    residual_ax.set_ylim(0, max(metrics["initial_target_node_mean"], residual.max()) * 1.1)

    current_shape, = shape_ax.plot([], [], "o-", color="#e41a1c", lw=2.8,
                                   ms=5.5, label="模型预测当前形态")
    tip_trace, = shape_ax.plot([], [], color="#984ea3", lw=2.2,
                               alpha=0.9, label="预测末端路径")
    action_cursor = action_ax.axvline(1, color="#222222", lw=1.6)
    residual_cursor = residual_ax.axvline(1, color="#222222", lw=1.6)
    residual_point, = residual_ax.plot([], [], "o", color="#e41a1c", ms=7)
    shape_ax.legend(loc="best", fontsize=8)
    status = figure.text(0.5, 0.012, "", ha="center", va="bottom", fontsize=11)
    dt = float(data["plan"]["step_interval_s"])

    def update(index: int):
        state = data["predicted"][index]
        current_shape.set_data(state[:, 0], state[:, 1])
        path = data["predicted"][:index + 1, -1]
        tip_trace.set_data(path[:, 0], path[:, 1])
        step = index + 1
        action_cursor.set_xdata([step, step])
        residual_cursor.set_xdata([step, step])
        residual_point.set_data([step], [residual[index]])
        shape_ax.set_title(f"预测形态路径  k={step}/{len(steps)}")
        action_ax.set_title("六通道计划压力与当前控制步")
        residual_ax.set_title("模型预测形态到记录目标的残差")
        status.set_text(
            f"t={step * dt:.2f}s  全节点残差={residual[index]:.2f}{data['state_unit']} "
            f"({residual[index] * data['state_to_mm']:.2f}mm)")
        return current_shape, tip_trace, action_cursor, residual_cursor, residual_point, status

    animation = FuncAnimation(figure, update, frames=len(steps), interval=1000 / fps,
                              blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(figure)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(gif_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vsync", "0",
        "-movflags", "+faststart",
        str(mp4_path),
    ], check=True)


def export_terminal_overlay(data: dict, metrics: dict, output: Path) -> Path | None:
    target_frame = data["target_frame"]
    if target_frame is None:
        return None
    image_path = _raw_image_path(data["npz_path"], target_frame)
    if image_path is None or not image_path.is_file():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    if data["initial_camera"] is None:
        return None

    def draw(nodes, color, thickness, radius):
        points = np.rint(nodes).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
        for point in points[:, 0]:
            cv2.circle(image, tuple(int(v) for v in point), radius, color, -1,
                       cv2.LINE_AA)

    draw(data["initial_camera"], (255, 120, 40), 2, 3)
    draw(data["target_camera"], (70, 210, 70), 3, 4)
    draw(data["predicted_camera"][-1], (40, 70, 255), 3, 4)
    cv2.rectangle(image, (8, 8), (510, 89), (20, 20, 20), -1)
    cv2.putText(image, "recorded initial", (18, 31), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 120, 40), 2, cv2.LINE_AA)
    cv2.putText(image, "recorded target", (18, 54), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (70, 210, 70), 2, cv2.LINE_AA)
    cv2.putText(image, "model-predicted terminal", (18, 77),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 70, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), image)
    return image_path


def _write_metrics_csv(path: Path, data: dict, metrics: dict) -> None:
    residual_nodes = np.linalg.norm(
        data["predicted"] - data["target"][None], axis=2)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        unit = data["state_unit"]
        writer.writerow(["step", "t_sec", f"node_mean_{unit}", f"tip_{unit}",
                         "node_mean_mm", "tip_mm", *
                         [f"planned_ch{i}_kpa" for i in range(6)]])
        for index, (node_error, action) in enumerate(zip(residual_nodes, data["actions"])):
            writer.writerow([
                index + 1, (index + 1) * metrics["step_interval_s"],
                node_error.mean(), node_error[-1],
                node_error.mean() * data["state_to_mm"],
                node_error[-1] * data["state_to_mm"], *action.tolist(),
            ])


def _write_readme(path: Path, plan_dir: Path, metrics: dict,
                  target_photo: Path | None) -> None:
    unit = metrics["state_length_unit"]
    text = f"""# 5 Hz双段软体机器人离线逆规划展示

数据来源：`{plan_dir}`

本实验从真实采集序列的第 {metrics['initial_frame']} 帧形态出发，以第
{metrics['target_frame']} 帧记录形态作为完整15节点目标。OpenLoop planner优化
{metrics['horizon_steps']}步六通道压力序列，并由学习到的前向模型预测整段形态路径。

## 阶段结果

- 初始形态到目标：{metrics['initial_target_node_mean']:.3f} {unit}（全节点均值），
  {metrics['initial_target_tip']:.3f} {unit}（末端）。
- 预测终态到目标：{metrics['terminal_target_node_mean']:.3f} {unit}（全节点均值），
  {metrics['terminal_target_tip']:.3f} {unit}（末端）。
- 终态全节点残差相对初始下降：{metrics['residual_reduction_percent']:.2f}%。
- 路径最小全节点残差：{metrics['best_path_node_mean']:.3f} {unit}，位于第
  {metrics['best_path_step']}步。
- 控制时域：{metrics['horizon_steps']} × {metrics['step_interval_s']:.6f} s =
  {metrics['horizon_duration_s']:.3f} s。
- 预测终态全节点残差：{metrics['terminal_target_node_mean_mm']:.3f} mm。

这些数值描述学习前向模型中的离线逆规划结果。真实控制结果由真机执行后的相机观测另行评价。

## 文件

- `control_rollout.gif` / `control_rollout.mp4`：预测形态、计划压力和目标残差同步动画。
- `control_summary.png`：完整路径、动作曲线和残差曲线。
- `terminal_overlay.png`：记录目标照片上的起始、目标和模型预测终态叠图。
- `metrics.json` / `per_step.csv`：汇总和逐步数值。
"""
    if target_photo is not None:
        text += f"\n叠图背景对应原始记录图片：`{target_photo}`。\n"
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出真实记录形态的离线逆规划展示")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-frame", type=int)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("fps必须为正数")

    plan_dir = Path(args.plan_dir).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = _load(plan_dir, args.target_frame)
    metrics = _metrics(data)
    export_summary(data, metrics, output / "control_summary.png")
    export_animation(data, metrics, output / "control_rollout.gif",
                     output / "control_rollout.mp4", args.fps)
    target_photo = export_terminal_overlay(
        data, metrics, output / "terminal_overlay.png")
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_metrics_csv(output / "per_step.csv", data, metrics)
    _write_readme(output / "README.md", plan_dir, metrics, target_photo)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
