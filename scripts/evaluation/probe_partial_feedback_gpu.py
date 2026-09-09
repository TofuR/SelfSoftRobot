#!/usr/bin/env python3
"""Bounded CUDA probe on archived full-suffix problems, no hardware commands.

Includes transfers, synchronized CUDA completion, CPU QP and nonlinear checks.
GPU utilization is archived before/after; shared-load results are not an idle
device latency claim. This probe does not alter the original replay artifacts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.control.hereditary_feedback import ActionBounds, correct_suffix, batched_rollout
from src.utils.model_loader import load_model


def gpu_status():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                                    "--format=csv"], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 40, 70])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    (output/"status.txt").write_text("running\n")
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(args.device)
        source = Path(args.source_run)
        config = json.loads((source/"config.json").read_text())
        with np.load(source/"traces.npz") as data:
            traces = {k: data[k] for k in data.files}
        model = load_model(config["checkpoint"], device=args.device)["model"].eval()
        model.requires_grad_(False)
        scale = traces["action_unit_to_kpa"]
        # The source run validated these exact 0..150 kPa, 50 kPa/s limits.
        raw_meta = json.loads((Path(config["raw"])/"meta.json").read_text())
        if raw_meta["lo6"] != [0.]*6 or raw_meta["hi6"] != [150.]*6 or raw_meta["rise_rates6"] != [50.]*6 or raw_meta["fall_rates6"] != [50.]*6:
            raise ValueError("GPU snapshot probe requires the declared source bounds")
        bounds = ActionBounds(np.zeros(4), 150/scale, 5/scale, 5/scale)
        before = gpu_status()
        rows = []
        with threadpool_limits(1):
            for k in args.steps:
                for repeat in range(args.repeats+1):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    z = torch.as_tensor(traces["sim_state_after"][k], device=args.device)
                    old = torch.as_tensor(traces["suffix_before"][k, k+1:], device=args.device)
                    previous = torch.as_tensor(traces["assumed_actions"][k], device=args.device)
                    reference = torch.as_tensor(traces["reference"][k+1:], device=args.device)
                    result, info = correct_suffix(model, z, old, previous, reference, bounds, rollout_fn=batched_rollout)
                    candidate = result.detach().cpu().numpy()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-started)*1000
                    if not bounds.valid(candidate, traces["assumed_actions"][k]):
                        raise AssertionError("CUDA candidate violates constraints")
                    row = {"step": k, "horizon": len(old), "repeat": repeat,
                           "warmup": repeat == 0, "elapsed_ms": elapsed,
                           "accepted": info["accepted"], "mse_after_mm2": info["mse_after_mm2"]}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        report = {"device": args.device, "name": torch.cuda.get_device_name(),
                  "gpu_before": before, "gpu_after": gpu_status(), "rows": rows,
                  "peak_tensor_memory_mb": torch.cuda.max_memory_allocated()/1e6,
                  "scope": "batched B controller only incl CPU-GPU transfer, QP, validation; synchronized; shared GPU load"}
        (output/"summary.json").write_text(json.dumps(report, indent=2))
        (output/"status.txt").write_text("complete\n")
        (output/"COMPLETE").touch()
    except Exception:
        (output/"status.txt").write_text("failed\n")
        raise


if __name__ == "__main__":
    main()
