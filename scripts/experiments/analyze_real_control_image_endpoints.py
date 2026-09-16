#!/usr/bin/env python3
"""Measure visible silicone end faces, then compare against archived pixel targets.

The image-only stage does not open plans, feedback, predictions or target events.
Seeds are approximate end-face locations reviewed on raw images. They locate a
small image region; the actual point comes from its segmented outline. This is
a retrospective, image-derived measurement, not a blinded human annotation study.

Run from the repository root (OpenCV, NumPy, Matplotlib and Pillow required):
  python scripts/experiments/analyze_real_control_image_endpoints.py
  python scripts/experiments/analyze_real_control_image_endpoints.py --stage measure
  python scripts/experiments/analyze_real_control_image_endpoints.py --stage compare
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path("workspace/runs/validation/real_robot_20260912_001")
OUTPUT = Path("workspace/runs/analysis/real_control_paper_20260913_007/visual")
SEEDS = {
    1: (283, 348), 2: (290, 351), 5: (286, 349), 7: (271, 345),
    8: (271, 345), 9: (281, 353), 10: (398, 361), 11: (398, 361),
    12: (397, 362), 13: (398, 361), 17: (259, 345), 18: (259, 347),
    19: (271, 348), 21: (267, 346), 22: (265, 348),
}
RIGHT_FACING = {10, 11, 12, 13}  # Direction of the visible raw end face.
PREVIOUS_CHECKS = {9, 10, 11, 12, 13, 21, 22}
BLUE, GOLD, INK = "#2873A6", "#B17A12", "#30363B"
DEFINITION = (
    "Center of the projected distal end face of the thick white silicone arm: "
    "intersection of its local midline with the robust distal silhouette. "
    "The thin pneumatic tubing continuing beyond that face is excluded."
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"Cannot decode {path} as UTF-8/BOM or GB18030")


def read_json(path: Path):
    text, encoding = read_text(path)
    return json.loads(text), encoding


def read_csv(path: Path):
    text, encoding = read_text(path)
    return list(csv.DictReader(io.StringIO(text))), encoding


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decode_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Undecodable image: {path}")
    if image.shape[:2] != (480, 640):
        raise ValueError(f"Review seeds require 640x480 source pixels: {path}")
    return image


def measure_endpoint(image, seed, side, threshold=150, kernel=5, angle_deg=0,
                     seed_shift=(0, 0), axis_window=(-18, -5)):
    """Local image-only estimator. No archive, plan or target access is possible.

    s points outward along the approximate distal direction; q is transverse.
    Opening separates 1-3 px tubing from the approximately 25 px silicone body.
    Local section midpoints determine the shaft axis. Median leading edges in
    a central 12 px band avoid rounded corners and small tube-attachment nubs.
    Quarter-pixel mask sampling improves numerical consistency, not resolution.
    """
    center = np.asarray(seed, dtype=float) + seed_shift
    normal = np.asarray([side, 0.7], dtype=float)
    normal /= np.linalg.norm(normal)
    angle = np.deg2rad(angle_deg)
    normal = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]) @ normal
    tangent = np.array([-normal[1], normal[0]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray >= threshold).astype("uint8")
    yy, xx = np.indices(mask.shape)
    mask *= (abs(xx - center[0]) < 65) & (abs(yy - center[1]) < 65)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    anchor = np.rint(center - normal * 18).astype(int)
    label = labels[anchor[1], anchor[0]]
    if label == 0:
        raise ValueError("Image-only interior anchor is outside segmented silicone")
    mask = (labels == label).astype("uint8")
    q = np.arange(-28, 28.001, 0.25)
    s = np.arange(-40, 22.001, 0.25)
    map_x = (center[0] + s[:, None] * normal[0] + q[None, :] * tangent[0]).astype("float32")
    map_y = (center[1] + s[:, None] * normal[1] + q[None, :] * tangent[1]).astype("float32")
    aligned = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    mids, positions, widths = [], [], []
    for j, value in enumerate(s):
        inds = np.flatnonzero(aligned[j])
        if axis_window[0] <= value <= axis_window[1] and len(inds) > 20:
            positions.append(value)
            mids.append((q[inds[0]] + q[inds[-1]]) / 2)
            widths.append(q[inds[-1]] - q[inds[0]])
    if len(mids) < 15 or not 12 <= np.median(widths) <= 45:
        raise ValueError("Insufficient visible terminal shaft, or a tube-sized component")
    coefficients = np.polyfit(positions, mids, 1)
    q_center = float(np.polyval(coefficients, 0))
    envelope = []
    for k, value in enumerate(q):
        inds = np.flatnonzero(aligned[:, k])
        if abs(value - q_center) <= 6 and len(inds):
            envelope.append(s[inds[-1]])
    if len(envelope) < 25 or max(envelope) >= s[-1]:
        raise ValueError("Distal face is missing or reaches the local search boundary")
    cap_s = float(np.median(envelope))
    q_center = float(np.polyval(coefficients, cap_s))
    point = center + normal * cap_s + tangent * q_center
    if np.linalg.norm(point - center) > 12:
        raise ValueError("Extraction moved too far from the raw-image face localization")
    diagnostics = dict(cap_s=cap_s, axis_q=q_center, cap_extent=float(np.ptp(envelope)),
                       axis_fit=coefficients.tolist(), median_shaft_width_px=float(np.median(widths)),
                       axis_sections=len(mids), cap_samples=len(envelope))
    return point, mask, diagnostics


def sensitivity_configs():
    return ([dict(threshold=t, kernel=k) for t, k in itertools.product((130, 150, 170), (5, 7, 9))]
            + [dict(seed_shift=shift) for shift in ((3, 0), (-3, 0), (0, 3), (0, -3))]
            + [dict(angle_deg=angle) for angle in (-7.5, 7.5)]
            + [dict(axis_window=window) for window in ((-20, -8), (-16, -6))])


def image_stage(archive: Path, output: Path):
    # Derive image selection solely from archived directory names / frame numbers.
    # In particular, do not read trial_index.csv or initial_plan.npz in this stage.
    directories = sorted(p for p in (archive / "trials").iterdir() if p.is_dir())
    assert {int(p.name[1:3]) for p in directories} == set(SEEDS)
    measurements, sensitivities, previous, inspected = [], [], [], []
    (output / "masks").mkdir(exist_ok=True)
    for directory in directories:
        trial = int(directory.name[1:3])
        # Listing filenames does not decode or hash the intervening images.
        frames = sorted((directory / "raw/cam0").glob("*.png"), key=lambda p: int(p.stem))
        frame = frames[-1]
        image = decode_image(frame)
        inspected.append(relative(frame))
        side = 1 if trial in RIGHT_FACING else -1
        point, mask, diagnostic = measure_endpoint(image, SEEDS[trial], side)
        cv2.imwrite(str(output / "masks" / f"t{trial:02d}_terminal_mask.png"), mask * 255)
        record = dict(trial=trial, name=directory.name, raw_frame=relative(frame),
                      frame_index=int(frame.stem), image_size=[640, 480],
                      seed=list(SEEDS[trial]), side=side, point=point.tolist(), diag=diagnostic,
                      tip_valid=True, validity_reason="Silicone distal face and local shaft both visible",
                      visual_review="Agent inspected raw final crop and segmented terminal outline",
                      reviewer_type="AI visual inspection; independent human repeat annotation not performed")
        measurements.append(record)
        variants = []
        for config in sensitivity_configs():
            alternative, _, _ = measure_endpoint(image, SEEDS[trial], side, **config)
            variants.append(dict(config=config, point=alternative.tolist(),
                                 shift_px=float(np.linalg.norm(alternative - point))))
        sensitivities.append(dict(trial=trial, variants=variants,
                                  max_shift_px=max(v["shift_px"] for v in variants)))
        if trial in PREVIOUS_CHECKS:
            before = frame.with_name(f"{int(frame.stem) - 1:05d}.png")
            previous_image = decode_image(before)
            inspected.append(relative(before))
            before_point, _, _ = measure_endpoint(previous_image, SEEDS[trial], side)
            previous.append(dict(trial=trial, previous_raw_frame=relative(before),
                                 previous_point_px=before_point.tolist(),
                                 last_minus_previous_px=(point - before_point).tolist(),
                                 displacement_px=float(np.linalg.norm(point - before_point)),
                                 reason="G03 small differences / blocker proximity or handheld blocker",
                                 interpretation=("Raw images show visible endpoint motion; terminal sample is not a settled-state estimate"
                                                 if trial == 21 else "Visible final and preceding end faces checked")))
    document = dict(schema="real_control_image_endpoint_v1", status="image_measurements_fixed",
                    definition=DEFINITION,
                    coordinate_system="Original cam0 pixels; origin upper left; x right, y down; 640x480",
                    algorithm="Local gray>=150; ellipse 5 opening; interior-connected silicone; local axis and robust distal cap",
                    independence="Image-stage estimator receives only image, raw-image seed, and raw-image distal direction",
                    review_limit="Existing target-overlay contact sheets were seen for context; this is not a blinded human study",
                    measurements=measurements, source_images_decoded=inspected,
                    unique_source_images=len(set(inspected)), image_selection="15 final frames plus 7 immediately preceding frames",
                    created_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "endpoint_image_only.json", document)
    write_json(output / "extraction_sensitivity.json", sensitivities)
    write_json(output / "preceding_frame_checks.json", previous)
    write_json(output / "frame_selection.json", [dict(trial=r["trial"], raw_frame=r["raw_frame"],
                                                       frame_index=r["frame_index"], image_size=r["image_size"])
                                               for r in measurements])
    return document


def calibration_events(archive: Path):
    source = archive / "session/events.jsonl"
    text, encoding = read_text(source)
    records = []
    for line, value in enumerate(text.splitlines(), 1):
        event = json.loads(value)
        if event.get("event") in ("joint_initial_calibration", "alignment_confirmed"):
            records.append(dict(source=relative(source), line=line, encoding=encoding,
                                **{key: event[key] for key in ("event", "t", "matrix", "diagnostics", "history_epoch") if key in event}))
    return records


def comparison_stage(archive: Path, output: Path):
    image_document, _ = read_json(output / "endpoint_image_only.json")
    sensitivity, _ = read_json(output / "extraction_sensitivity.json")
    previous, _ = read_json(output / "preceding_frame_checks.json")
    sensitivity = {r["trial"]: r for r in sensitivity}
    previous = {r["trial"]: r for r in previous}
    index, index_encoding = read_csv(archive / "trial_index.csv")
    indexed = {int(row["trial"]): row for row in index}
    manifest, manifest_encoding = read_json(archive / "manifest.json")
    events = calibration_events(archive)
    results, calibrations = [], []
    for measured in image_document["measurements"]:
        trial = measured["trial"]
        source = indexed[trial]
        directory = archive / source["archive_relative"]
        metadata, metadata_encoding = read_json(directory / "metadata.json")
        samples, sample_encoding = read_csv(directory / "samples.csv")
        sample = next(r for r in samples if int(r["camera"]) == 0 and int(r["step"]) == measured["frame_index"])
        assert int(source["planned_steps"]) - 1 == measured["frame_index"]
        assert source["status"] == "completed" and sample["fresh_after_command"] == "True"
        plan_path = directory / "initial_plan.npz"
        # These are the only plan arrays used; no prediction is read.
        with np.load(plan_path, allow_pickle=False) as plan:
            goal, matrix, indices = plan["goal_mm"], plan["camera_matrix"], plan["node_indices"]
        assert goal.shape == (15, 2) and matrix.shape == (3, 3) and 14 in indices
        assert np.allclose(matrix[2], [0, 0, 1])
        projected = goal @ matrix[:2, :2].T + matrix[:2, 2]
        assert np.all(np.isfinite(projected))
        point = np.asarray(measured["point"])
        delta = point - projected[-1]
        alternative_errors = [float(np.linalg.norm(np.asarray(v["point"]) - projected[-1]))
                              for v in sensitivity[trial]["variants"]]
        calibration = [event for event in events if event["event"] == "joint_initial_calibration"
                       and event["t"] <= float(source["started"])
                       and np.allclose(event["matrix"], matrix, atol=1e-10, rtol=0)]
        assert calibration, f"Saved camera matrix lacks a matching retained calibration event: T{trial}"
        matched = calibration[-1]
        preceding = previous.get(trial)
        previous_sample = (next(r for r in samples if int(r["camera"]) == 0 and int(r["step"]) == measured["frame_index"] - 1)
                           if preceding else None)
        row = {key: source[key] for key in ("trial", "name", "execution_id", "target_group", "target_kind", "occlusion", "correction", "feedback_interval")}
        row.update(trial=trial, status="valid_final_frame_tip", tip_valid=True,
                   raw_frame=measured["raw_frame"], frame_index=measured["frame_index"],
                   image_width=640, image_height=480,
                   measured_tip_x_px=float(point[0]), measured_tip_y_px=float(point[1]),
                   target_tip_x_px=float(projected[-1, 0]), target_tip_y_px=float(projected[-1, 1]),
                   delta_x_px=float(delta[0]), delta_y_px=float(delta[1]), tip_error_px=float(np.linalg.norm(delta)),
                   extraction_sensitivity_max_shift_px=sensitivity[trial]["max_shift_px"],
                   tip_error_sensitivity_min_px=min(alternative_errors), tip_error_sensitivity_max_px=max(alternative_errors),
                   sensitivity_variants=len(alternative_errors), sensitivity_is_confidence_interval=False,
                   previous_frame_displacement_px=preceding["displacement_px"] if preceding else None,
                   previous_frame_interval_ms=(1000 * (float(sample["frame_timestamp"]) - float(previous_sample["frame_timestamp"]))) if preceding else None,
                   temporal_flag="visible_motion_between_last_two_frames" if trial == 21 else ("adjacent_frame_checked" if preceding else "single_final_frame"),
                   frame_timestamp=float(sample["frame_timestamp"]), after_command_ms=float(sample["after_command_ms"]),
                   fresh_after_command=True, settle_s=metadata["settle_s"],
                   measurement_time_definition="Last archived fresh camera frame associated with final commanded step",
                   full_shape_metric_status="not_measured_clear_body" if source["occlusion"] == "clear" else "unobservable_behind_physical_blocker",
                   target_source=relative(plan_path), target_field="goal_mm[-1] @ camera_matrix[:2,:2].T + camera_matrix[:2,2]",
                   target_node_index=14, physical_scale_validated=False,
                   calibration_event_source=matched["source"], calibration_event_line=matched["line"],
                   calibration_method=matched["diagnostics"]["method"],
                   metadata_encoding=metadata_encoding, samples_encoding=sample_encoding,
                   validity_reason=measured["validity_reason"], visual_review=measured["visual_review"],
                   source_execution=source["source_execution"])
        results.append(row)
        calibrations.append(dict(trial=trial, source=relative(plan_path), camera_matrix=matrix.tolist(),
                                 goal_field="goal_mm", goal_tip_stored_units=goal[-1].tolist(),
                                 node_indices=indices.tolist(), projected_goal_px=projected.tolist(),
                                 calibration_event_source=matched["source"], calibration_event_line=matched["line"],
                                 calibration_event_time=matched["t"], calibration_diagnostics_as_recorded=matched["diagnostics"],
                                 physical_scale_validation="Saved scale was jointly fit to model shape and bounded memory; independent physical scale not established here",
                                 evaluation_use="Forward projection of the archived goal into original pixels only"))
    values = [r["tip_error_px"] for r in results]
    summary = dict(n_original=22, n_complete=15, n_valid_tips=len(results), n_invalid_complete_tips=0,
                   n_clear=9, n_occluded=6, n_measured_full_shapes=0,
                   mean_tip_error_px=float(np.mean(values)), median_tip_error_px=float(np.median(values)),
                   min_tip_error_px=float(min(values)), max_tip_error_px=float(max(values)),
                   max_extraction_shift_px=max(r["extraction_sensitivity_max_shift_px"] for r in results),
                   pooled_summary_scope="Descriptive only; target/occlusion/correction conditions are heterogeneous",
                   temporal_exception="T21 moves about 9.25 px between last two frames, separated by 204 ms; final-frame endpoint remains visible and measurable",
                   group_comparison_limit="Trials have different initial states and histories; comparisons are descriptive, not paired causal estimates")
    exclusions = []
    for trial, source in indexed.items():
        included = trial in SEEDS
        exclusions.append(dict(trial=trial, execution_id=source["execution_id"], source_execution=source["source_execution"],
                               archive_relative=source["archive_relative"], original_status=source["status"],
                               visual_analysis_status="valid_final_frame_tip" if included else "outside_complete_trial_cohort",
                               reason="Complete archived trial with visible distal end face" if included else (source["failure_reason"] or "Execution completed but terminal feedback unconfirmed"),
                               source_images_opened=included))
    audit = dict(existing_analysis="workspace/runs/analysis/real_robot_20260912_001",
                 found_independent_measured_tip_metrics=False, found_independent_measured_shape_metrics=False,
                 evidence=[dict(source=relative(archive / "README.md"), section="初步发现 5 / 下一轮详细分析",
                                finding="Model residuals are explicitly distinguished from physical accuracy; final image extraction remains future work"),
                           dict(source="workspace/runs/analysis/real_robot_20260912_001/visual_review.json",
                                finding="Visibility/condition review, without measured endpoint coordinates"),
                           dict(source="workspace/runs/analysis/real_robot_20260912_001/inventory.json",
                                finding="Completion estimates carry model-estimate semantics and physical_arrival_verified=False")],
                 manifest_source=relative(archive / "manifest.json"), manifest_encoding=manifest_encoding,
                 manifest_counts=manifest["counts"], trial_index_source=relative(archive / "trial_index.csv"),
                 trial_index_encoding=index_encoding, original_data_modified=False, hashes_computed=False)
    write_csv(output / "endpoint_measurements.csv", results)
    write_json(output / "endpoint_measurements.json", dict(schema="real_control_visual_endpoint_results_v1",
               status="validated_final_frame_pixel_measurements", definition=DEFINITION,
               coordinate_system=image_document["coordinate_system"], summary=summary, trials=results,
               calibration_provenance=calibrations))
    write_json(output / "summary.json", summary)
    write_json(output / "calibration_provenance.json", calibrations)
    write_json(output / "calibration_events.json", events)
    write_csv(output / "trial_validity_all22.csv", exclusions)
    write_json(output / "trial_validity_all22.json", exclusions)
    write_json(output / "existing_analysis_audit.json", audit)
    return results, calibrations, image_document, previous, summary


def save_figure(fig, output, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_overlay(ax, row, calibration, crop=None):
    image = Image.open(ROOT / row["raw_frame"])
    ax.imshow(image, interpolation="nearest")
    goal = np.asarray(calibration["projected_goal_px"])
    if row["target_kind"] == "full":
        ax.plot(goal[:, 0], goal[:, 1], color=GOLD, ls="--", lw=1, alpha=0.85)
    x, y = row["measured_tip_x_px"], row["measured_tip_y_px"]
    tx, ty = row["target_tip_x_px"], row["target_tip_y_px"]
    ax.plot([x, tx], [y, ty], color="white", lw=1.2)
    ax.plot(tx, ty, marker="x", color=GOLD, ms=9, mew=1.8, ls="none")
    ax.plot(x, y, marker="o", color=BLUE, ms=7, mfc="none", mew=1.8, ls="none")
    if crop:
        ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_xlabel("x (source px)"); ax.set_ylabel("y (source px)")
    ax.tick_params(labelsize=7)
    ax.set_title(f"T{row['trial']:02d} | {row['occlusion']}, {row['correction']} | {row['tip_error_px']:.1f} px", fontsize=10, loc="left")


def render_artifacts(output, rows, calibrations, image_document, previous):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    indexed = {r["trial"]: r for r in rows}
    calibration = {r["trial"]: r for r in calibrations}
    write_json(output / "chart_contract.json", dict(
        analytical_question="Final visible silicone end-face distance to the saved target, by trial and target group",
        family="comparison", variant="faceted horizontal points with extraction sensitivity ranges",
        renderer="Matplotlib static PNG/PDF/SVG", grain="one final raw cam0 frame per completed trial; n=15",
        units="original image pixels", palette={"closed": BLUE, "open": GOLD},
        noncolor_encoding="closed circles / open squares; explicit condition and trial labels",
        whiskers="Min/max target error over 17 deterministic image-extraction settings; not confidence intervals",
        source_table="endpoint_measurements.csv", final_QA="Inspect rendered PNG and original-pixel overlays"))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), layout="constrained")
    for ax, group in zip(axes.flat, ("G01", "G02", "G03", "G06")):
        subset = [r for r in rows if r["target_group"] == group]
        for i, row in enumerate(subset):
            value = row["tip_error_px"]
            lo, hi = row["tip_error_sensitivity_min_px"], row["tip_error_sensitivity_max_px"]
            closed = row["correction"] == "closed"
            color = BLUE if closed else GOLD
            ax.hlines(i, 0, value, color="#D8DDE1", lw=1)
            ax.errorbar(value, i, xerr=[[max(0, value - lo)], [max(0, hi - value)]],
                        fmt="o" if closed else "s", color=color, mfc=color if closed else "white",
                        ms=6, capsize=3, lw=1.2)
            ax.text(25.7, i, f"{value:.1f}", ha="right", va="center", fontsize=9)
        ax.set_yticks(range(len(subset)), [f"T{r['trial']:02d}{'*' if r['trial']==21 else ''}  {r['occlusion']} / {r['correction']} / i{r['feedback_interval']}" for r in subset])
        ax.set_xlim(0, 26.2); ax.set_ylim(len(subset) - 0.5, -0.7)
        ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.grid(axis="x", color="#E4E7E9", lw=0.6)
        ax.set_axisbelow(True); ax.set_xlabel("Final-frame tip error (px)")
        ax.set_title(f"{group} | {'Full-shape target' if subset[0]['target_kind']=='full' else 'Tip target'} | n={len(subset)}", loc="left")
    fig.suptitle("Real-control final-frame endpoint errors", fontsize=14)
    fig.supxlabel("Whiskers: extraction sensitivity, not confidence intervals.  *T21: visible motion at the final sample.", fontsize=9)
    save_figure(fig, output, "endpoint_errors_by_trial")

    fig, axes = plt.subplots(5, 3, figsize=(12, 13.4), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        x = row["measured_tip_x_px"]
        draw_overlay(ax, row, calibration[row["trial"]], crop=(x - 40, 310, x + 50, 389))
    fig.suptitle("Final raw-image endpoint audit | circle: measured silicone end face; cross: saved target", fontsize=13)
    save_figure(fig, output, "endpoint_overlay_all15")

    raw_directory = output / "raw_samples"
    raw_directory.mkdir(exist_ok=True)
    for trial in (1, 13, 21):
        row = indexed[trial]
        shutil.copyfile(ROOT / row["raw_frame"], raw_directory / f"t{trial:02d}_final.png")
        fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
        draw_overlay(ax, row, calibration[trial])
        ax.legend(handles=[Line2D([], [], marker="o", mfc="none", color=BLUE, ls="none", label="Measured end face"),
                           Line2D([], [], marker="x", color=GOLD, ls="--", label="Saved target")], loc="lower right")
        save_figure(fig, output, f"endpoint_overlay_t{trial:02d}")

    # Original-color, target-free local segmentation audit; only diagnostic marks added.
    tiles = []
    for measured in image_document["measurements"]:
        trial = measured["trial"]
        image = decode_image(ROOT / measured["raw_frame"])
        mask = cv2.imread(str(output / "masks" / f"t{trial:02d}_terminal_mask.png"), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = image.copy()
        cv2.drawContours(canvas, contours, -1, (18, 122, 177), 1)
        cv2.drawMarker(canvas, tuple(np.rint(measured["point"]).astype(int)), (166, 115, 40), cv2.MARKER_CROSS, 9, 1)
        crop = Image.fromarray(cv2.cvtColor(canvas[290:390, 235:425], cv2.COLOR_BGR2RGB)).resize((570, 300), Image.Resampling.NEAREST)
        tiles.append((f"T{trial:02d} | measured {np.round(measured['point'], 1)} source px", crop))
    for start in (0, 8):
        subset = tiles[start:start + 8]
        page = Image.new("RGB", (1140, 330 * ((len(subset) + 1) // 2)), "white")
        draw = ImageDraw.Draw(page)
        for j, (label, crop) in enumerate(subset):
            x, y = j % 2 * 570, j // 2 * 330
            page.paste(crop, (x, y + 25)); draw.text((x + 5, y + 5), label, fill="black")
        page.save(output / f"segmentation_review_{start // 8 + 1}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained")
    last = indexed[21]
    before = previous[21]
    for ax, path, point, title in zip(axes, [before["previous_raw_frame"], last["raw_frame"]],
                                    [before["previous_point_px"], [last["measured_tip_x_px"], last["measured_tip_y_px"]]],
                                    ["T21 | preceding frame 00048", "T21 | final frame 00049"]):
        ax.imshow(Image.open(ROOT / path), interpolation="nearest")
        ax.plot(*point, "o", color=BLUE, mfc="none", mew=1.8, ms=9)
        ax.set_xlim(235, 355); ax.set_ylim(390, 260)
        ax.set_title(title, loc="left"); ax.set_xlabel("x (source px)"); ax.set_ylabel("y (source px)")
    fig.suptitle(f"T21 endpoint movement: {before['displacement_px']:.1f} px over {last['previous_frame_interval_ms']:.0f} ms", fontsize=13)
    save_figure(fig, output, "t21_terminal_frame_check")


def write_readme(output, rows, summary):
    table = ["| Trial | Group / condition | Measured (x,y) px | Target (x,y) px | Error px | Sensitivity max shift px |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| T{row['trial']:02d} | {row['target_group']} / {row['occlusion']} / {row['correction']} / i{row['feedback_interval']} | "
                     f"({row['measured_tip_x_px']:.1f}, {row['measured_tip_y_px']:.1f}) | "
                     f"({row['target_tip_x_px']:.1f}, {row['target_tip_y_px']:.1f}) | {row['tip_error_px']:.1f} | {row['extraction_sensitivity_max_shift_px']:.1f} |")
    text = f"""# Real-control visual endpoint evaluation — 2026-09-13

15/15 complete archived trials have a measurable silicone distal end face in their final raw cam0 frame (9 clear, 6 physically occluded in the body). This retrospective evaluation supplies **final-frame 2D pixel distances to the saved targets**. Full-body centerlines were not measured. All 22 original executions and the seven cohort exclusions are recorded in `trial_validity_all22.csv`.

Primary machine-readable handoff: `endpoint_measurements.csv` (UTF-8 BOM) and `endpoint_measurements.json` (UTF-8). Status is `validated_final_frame_pixel_measurements`; each included trial has `tip_valid=true`. Floating-point values preserve the computation; manuscript values should be rounded to 0.1 px, without implying subpixel accuracy. The heterogeneous pooled median is {summary['median_tip_error_px']:.1f} px; range {summary['min_tip_error_px']:.1f}–{summary['max_tip_error_px']:.1f} px. It is not a common-condition performance estimate.

## Measurement definition and image independence

{DEFINITION}

Coordinates use the original 640×480 image: origin at the upper-left pixel center, x rightward and y downward. The function `measure_endpoint` takes only raw image pixels, an approximate image-selected end-face seed and distal direction. Approximate seeds are saved in `endpoint_image_only.json` and the script. Neither target coordinates, model outputs, nor `prediction_px` enter that function. The image-only measurements were saved before projecting the targets. Existing target-overlay contact sheets were seen during initial context review; this is not a blinded human-rater study. Review was performed by the agent on raw crops and diagnostic overlays; independent human repeat annotation is still available as an additional validation step.

For each frame: threshold OpenCV grayscale at 150/255 within a 129×129 pixel neighborhood; perform an elliptical 5×5 opening; retain the connected component containing the silicone shaft 18 px inward of the seed. The opening removes thin tubing. Sample this component in local outward s and transverse q coordinates. Fit a line to bilateral section midpoints at s=−18…−5 px, avoiding the physical blocker. Estimate the distal boundary using the median leading edge in the central 12 px transverse band, then intersect it with the fitted local midline. Output masks retain original image coordinates. Quarter-pixel resampling of the binary mask is only numerical sampling. The T13 prototype exposed blocker bias with a longer inner interval; the final common interval was selected from image inspection before target comparison and applied to every trial.

## Frame selection, validity and temporal limitation

The script lists raw filenames and decodes only 15 final frames and seven immediately preceding frames (T09–T13, T21–T22). It does not enumerate/decode intermediate image contents or compute file hashes. Final filenames agree with planned step count minus one, and `samples.csv` identifies them as fresh cam0 frames after the final command. There are 22 distinct source images, some reread for plotting. No early frame substitutes for a final frame.

**T21 is a valid image measurement but is temporally sensitive:** its clearly visible end face moves {next(r['previous_frame_displacement_px'] for r in rows if r['trial']==21):.2f} px from frame 00048 to 00049 in 204 ms. Raw images show this displacement. Its final error of {next(r['tip_error_px'] for r in rows if r['trial']==21):.1f} px describes the last saved sample, not a settled endpoint. `t21_terminal_frame_check.png` and `preceding_frame_checks.json` preserve the evidence. Other reviewed adjacent-frame displacements and all available timestamp checks are included in the CSV. Every trial records `settle_s=0.0`; subsequent settled accuracy cannot be inferred from these final command-associated samples.

## Target and calibration provenance

The target is node 14 of `initial_plan.npz:goal_mm`, projected as `goal_mm @ camera_matrix[:2,:2].T + camera_matrix[:2,2]`. Node 14 belongs to `node_indices` in every included plan. For a tip-only task, only its constrained final target is drawn; unconstrained intermediate coordinates are not a target shape. Every per-trial matrix, original stored goal tip and full projected goal array is saved in `calibration_provenance.json`.

The matrices match retained `joint_initial_calibration` events in `session/events.jsonl`, lines 34, 2142 and 3349, with the associated `alignment_confirmed` events. Exact source line, time, matrix and recorded diagnostics are preserved in `calibration_events.json` and the per-trial provenance. The logged method is `joint_similarity_bounded_state_prior`: camera similarity and bounded model memory were fit jointly. Thus the available scale is an alignment parameter; it is not independent evidence for converting measured pixel errors into millimeters. Evaluation uses only forward projection into saved camera coordinates and reports pixels. Recorded calibration diagnostics retain their original field names and units as provenance, not experimental endpoint accuracy.

Relevant implementation provenance: `Hereditary_workbench/real_validation/execution/hereditary_executor.py` saves `goal_mm`, `node_indices`, and `camera_matrix`; `runtime/hereditary_deployment.py:calibrate_full_shape` documents the joint fit. The current source was read for interpretation; the retained execution event/matrix is the trial-specific evidence. Existing runtime endcap logic also defines the tip at the endcap center, rather than a thinning branch corner.

## Sensitivity and interpretation

`extraction_sensitivity.json` contains 17 deterministic settings per final frame: thresholds 130/150/170 crossed with opening kernels 5/7/9, seed shifts ±3 px on each axis, distal-angle shifts ±7.5°, and alternative local-axis windows −20…−8 and −16…−6 px (each remaining perturbation applied individually). Maximum coordinate shifts across trials are 0.4–{summary['max_extraction_shift_px']:.1f} px. Plot whiskers are the minimum/maximum target distance across these settings, **not confidence intervals or a calibrated total uncertainty bound**. The sensitivity exercise does not quantify target-drawing ambiguity, perspective, alignment drift, motion between frames, systematic cap-definition error, or inter-rater variation. Those remain error sources.

The G03 four-trial errors are about 7.4–8.1 px, so their ordering is not resolved by this extraction. T18/T19 give 3.3/9.9 px in the clear G06 condition; T21/T22 give 7.2/3.8 px in the occluded G06 condition, with the T21 temporal qualification above. Initial memory, hold history and commands differ across executions. These are descriptive trial outcomes; a general causal benefit of correction is not established. Body visibility is not a full-shape accuracy result, and hidden centerlines are not reconstructed.

## Files and reproduction

- `endpoint_measurements.csv/json`, `summary.json`: final measurements, targets, signed deltas, errors, sensitivity, frame timing, validity and provenance.
- `endpoint_image_only.json`, `masks/`, `segmentation_review_1.png`, `segmentation_review_2.png`: target-independent extraction inputs, coordinates and visual checks.
- `endpoint_errors_by_trial.png/pdf/svg`: per-trial pixel errors, four target groups, sensitivity whiskers.
- `endpoint_overlay_all15.png/pdf/svg`: all final end faces with measurement circles, target crosses and original-pixel axes.
- `raw_samples/`: three byte-for-byte copies of final raw images (T01, T13, T21); `endpoint_overlay_t01/t13/t21.*`: annotated full-image counterparts. Source images are unchanged.
- `t21_terminal_frame_check.*`, `preceding_frame_checks.json`: final/pre-final temporal check.
- `calibration_provenance.json`, `calibration_events.json`: exact projected target and alignment provenance.
- `trial_validity_all22.csv/json`, `existing_analysis_audit.json`: cohort accounting and prior-metric audit.
- `chart_contract.json`, `validation.json`: plotting semantics and reproducibility checks.

Run from `/Data5/ddf/projects/SelfSoftRobot`:

```bash
python scripts/experiments/analyze_real_control_image_endpoints.py
```

`--stage measure` regenerates the image-only measurements and masks, without opening plans or target events. `--stage compare` reads that saved image-only result and produces targets, metrics and figures. Text readers explicitly support UTF-8 BOM and GB18030; source encodings are recorded. NumPy archives are opened with `allow_pickle=False`.

## Trial measurements

""" + "\n".join(table) + "\n"
    (output / "README.md").write_text(text, encoding="utf-8")


def validate(output, rows, image_document):
    assert len(rows) == 15 and len({r["trial"] for r in rows}) == 15
    assert all(r["tip_valid"] for r in rows)
    assert sum(r["occlusion"] == "clear" for r in rows) == 9
    assert all(not r["physical_scale_validated"] for r in rows)
    for row in rows:
        delta = np.array([row["measured_tip_x_px"] - row["target_tip_x_px"],
                          row["measured_tip_y_px"] - row["target_tip_y_px"]])
        assert abs(np.linalg.norm(delta) - row["tip_error_px"]) < 1e-10
        assert row["sensitivity_variants"] == 17
    # A real failure mode: target changes must affect only the comparison.
    first = image_document["measurements"][0]
    image = decode_image(ROOT / first["raw_frame"])
    recomputed, _, _ = measure_endpoint(image, first["seed"], first["side"])
    assert np.allclose(recomputed, first["point"], atol=1e-10, rtol=0)
    for trial in (1, 13, 21):
        source = next(r for r in rows if r["trial"] == trial)
        assert (ROOT / source["raw_frame"]).read_bytes() == (output / "raw_samples" / f"t{trial:02d}_final.png").read_bytes()
    write_json(output / "validation.json", dict(
        status="passed", n_valid_tips=15, n_cohort_entries=22, source_images=image_document["unique_source_images"],
        recomputed_pixel_errors="passed", saved_image_only_coordinate_reproduction="passed",
        final_frame_matches_final_step="passed", final_frames_fresh_after_command="passed",
        calibration_event_matrix_matches="15/15", sensitivity_variants="17 per final frame",
        raw_sample_byte_comparison="3/3 identical; no hashes computed",
        visual_QA="Raw final faces, segmented boundaries, endpoint plots and T21 preceding-frame comparison reviewed by agent",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__),
        time_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "measure", "compare"), default="all")
    args = parser.parse_args()
    archive, output = ROOT / ARCHIVE, ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "measure"):
        image_stage(archive, output)
    if args.stage in ("all", "compare"):
        rows, calibrations, images, previous, summary = comparison_stage(archive, output)
        render_artifacts(output, rows, calibrations, images, previous)
        write_readme(output, rows, summary)
        validate(output, rows, images)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Results: {output}")


if __name__ == "__main__":
    main()
