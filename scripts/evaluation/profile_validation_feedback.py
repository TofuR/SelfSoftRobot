#!/usr/bin/env python3
"""Profile deployed NumPy feedback on recorded, occluded images (no hardware).

Compare action-block counts on identical post-observation snapshots. Alternative
results never feed the next sample. Recorded images do not respond to proposals;
objective reduction is a model diagnostic, not physical control accuracy.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.benchmark_partial_feedback import prepare
from real_validation.runtime.hereditary_deployment import (
    HereditaryDeployment, load_bundle, edge_state_update, transform,
)
from real_validation.runtime.hereditary_math import fast_suffix_b
from real_validation.perception.partial_edges import extract_edges_vectorized


def stats(values):
    values = np.asarray(values, float)
    return dict(zip(('p50', 'p95', 'max'), np.percentile(values, [50, 95, 100]).tolist()))


def timed(fn):
    tick = time.perf_counter()
    value = fn()
    return value, (time.perf_counter()-tick)*1000


def roi_edges(image, predicted, radius):
    # Include every candidate, its +/-3 pixel appearance probes, and the
    # Gaussian+Sobel support. Coordinates remain in the original image frame.
    margin = radius+12+8
    lo = np.maximum(np.floor(predicted.min(0)-margin).astype(int), 0)
    hi = np.minimum(np.ceil(predicted.max(0)+margin).astype(int)+1, image.shape[1::-1])
    if np.any(hi <= lo):
        return extract_edges_vectorized(image, predicted, radius=radius)
    found = extract_edges_vectorized(image[lo[1]:hi[1], lo[0]:hi[0]], predicted-lo, radius=radius)
    return type(found)(found.pixels+lo, found.segments, found.strengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        source_model, _, data, trace, images, _, audit, _ = prepare(SimpleNamespace(source_run=args.source_run))
        engine, meta = load_bundle(args.bundle)
        runtime = HereditaryDeployment(engine, meta, out/'runtime')
        if not np.isclose(runtime.dt, float(source_model.dt)):
            raise ValueError('Use a source matching the bundle native rate')
        scale = np.asarray(meta['action_unit_to_kpa'])
        actions = data.actions*trace['action_unit_to_kpa']/scale
        runtime.initialize(runtime.mapping.expand(actions[0]))
        matrix = np.asarray(data.model_to_camera)
        runtime.matrix = matrix
        radius = meta['radius_mm']*np.linalg.norm(matrix[:2, 0])
        start, stop = audit['task_local_slice']
        z = runtime.state.copy()
        for u in actions[:start]:
            z = engine.step(z, u)
        source_plan = trace['initial_plan']*trace['action_unit_to_kpa']/scale
        snapshots = []
        for k, raw in enumerate(images[:-1]):
            u = actions[start+k]
            z = engine.step(z, u)
            image = raw.copy()
            x, y, w, h = audit['occluder_xywh']
            image[y:y+h, x:x+w] = audit['occluder_bgr']
            before = z.copy()
            edges = extract_edges_vectorized(image, transform(engine.observe(z, u), matrix), radius=radius)
            z, _ = edge_state_update(engine, z, u, edges, matrix, radius, runtime.bounds)
            old = runtime.bounds.project(source_plan[k+1:], u)
            snapshots.append((image, before, z.copy(), u.copy(), old, trace['reference'][k+1:]))

        rows = []
        # One discarded full pass warms all compared paths. Each subsequent
        # snapshot is fixed across repeats and block counts.
        for repeat in range(-1, args.repeats):
            for k, (image, before, corrected, u, old, reference) in enumerate(snapshots):
                row = dict(repeat=repeat, frame=int(trace['raw_indices'][k]), horizon=len(old))
                predicted, row['prediction_ms'] = timed(lambda: transform(engine.observe(before, u), matrix))
                edges, row['edge_ms'] = timed(lambda: extract_edges_vectorized(image, predicted, radius=radius))
                cropped, row['edge_roi_ms'] = timed(lambda: roi_edges(image, predicted, radius))
                row['roi_equivalent'] = bool(np.array_equal(edges.segments, cropped.segments) and
                    edges.pixels.shape == cropped.pixels.shape and np.allclose(edges.pixels, cropped.pixels, atol=5e-5, rtol=0))
                (_, obs), row['observer_ms'] = timed(lambda: edge_state_update(engine, before, u, edges, matrix, radius, runtime.bounds))
                row.update(edges=len(edges.pixels), observer_accepted=obs['accepted'])
                # Order alternates to reduce systematic order/cache advantage.
                for blocks in ((8, 4, 2) if (k+repeat)%2 else (2, 4, 8)):
                    candidate, info = fast_suffix_b(engine, corrected, old, u, reference, runtime.bounds, blocks=blocks)
                    if not runtime.bounds.valid(candidate, u):
                        raise AssertionError('pressure/rate violation')
                    row[f'b{blocks}'] = info
                # Measure the actual App entry point, including history replay
                # and wall-time hold advancement. No ACK/camera/thread/IO here.
                stamp = time.monotonic()
                runtime.state=before.copy();runtime.action=u.copy();runtime.at=stamp
                runtime.history=deque([(stamp, before.copy(), u.copy())], maxlen=4096)
                runtime.last_frame=-float('inf')
                runtime.target_shape=reference[-1]
                (_, full), row['runtime_wall_ms'] = timed(lambda: runtime.feedback(image, stamp, old, reference))
                row['runtime'] = {key: full[key] for key in ('edge_ms', 'observer_ms', 'compute_ms', 'control')}
                for blocks in (8,4):
                    stamp=time.monotonic();runtime.at=stamp;runtime.state=before.copy();runtime.action=u.copy()
                    runtime.history=deque([(stamp,before.copy(),u.copy())],maxlen=4096);runtime.last_frame=-float('inf')
                    def controller(*a, **kw):
                        return fast_suffix_b(*a, blocks=blocks, **kw)
                    # Experiment-local substitution only; no App source changes.
                    with patch('real_validation.runtime.hereditary_deployment.extract_edges_vectorized',roi_edges), \
                         patch('real_validation.runtime.hereditary_deployment.fast_suffix_b',controller):
                        (_, result), row[f'runtime_roi_b{blocks}_ms']=timed(lambda:runtime.feedback(image,stamp,old,reference))
                if repeat >= 0:
                    rows.append(row)
            print(f'pass {repeat+1}/{args.repeats} complete', flush=True)

        # Fully blind image must be a finite no-evidence return, not a search
        # loop or a synthetic full shape. Same state and nonempty suffix.
        image, before, _, u, old, reference = snapshots[0]
        blind=[]
        for _ in range(10):
            stamp=time.monotonic();runtime.at=stamp;runtime.state=before.copy();runtime.action=u.copy()
            runtime.history=deque([(stamp,before.copy(),u.copy())],maxlen=4096);runtime.last_frame=-float('inf')
            (candidate, info), elapsed=timed(lambda:runtime.feedback(np.full_like(image,35),stamp,old,reference))
            assert info['edges']==0 and not info['control']['accepted'] and np.array_equal(candidate,old)
            blind.append(elapsed)
        # Copy overhead as history fills during a longer preparation period.
        copy_times={}
        for count in (80,4096):
            runtime.history=deque([(float(i),before.copy(),u.copy()) for i in range(count)],maxlen=4096)
            copy_times[str(count)]=stats([timed(runtime.feedback_snapshot)[1] for _ in range(20)])
        summary=dict(schema='validation_feedback_profile_v1', args=vars(args),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            machine=platform.platform(),processor=platform.processor(),threadpools=threadpool_info(),
            opencv_threads=cv2.getNumThreads(), model_sha256=meta['weights_sha256'],dt=runtime.dt,
            model=dict(states=engine.z_dim,nodes=engine.n_nodes), frames=len(snapshots),samples=len(rows),
            raw_frames=trace['raw_indices'][:-1].tolist(),occluder=audit['occluder_xywh'],image_shape=list(images[0].shape),
            scope='CPU recorded-image snapshot benchmark; excludes ACK, camera, worker scheduling, archive and GUI; not physical control accuracy',
            initialization='equilibrium then available dev-prefix recorded actions; old sequence calibration; no deployment-fit claim',
            timing={key:stats([r[key] for r in rows]) for key in ('prediction_ms','edge_ms','edge_roi_ms','observer_ms','runtime_wall_ms','runtime_roi_b8_ms','runtime_roi_b4_ms')},
            runtime_stages={key:stats([r['runtime'][key] for r in rows]) for key in ('edge_ms','observer_ms','compute_ms')},
            blocks={},no_evidence_ms=stats(blind),snapshot_copy_ms=copy_times,
            roi_equivalent=all(r['roi_equivalent'] for r in rows),pressure_constraints_passed=True)
        for b in (8,4,2):
            infos=[r[f'b{b}'] for r in rows]
            summary['blocks'][str(b)]=dict(
                timing={key:stats([i[key] for i in infos]) for key in ('derivative_ms','solver_ms','validation_ms','time_ms')},
                accepted=sum(i['accepted'] for i in infos),solver_failures=sum(not i['solver_success'] for i in infos),
                mean_before_mm2=float(np.mean([i['mse_before_mm2'] for i in infos])),
                mean_after_mm2=float(np.mean([i['mse_after_mm2'] for i in infos])))
        summary['horizon_bins']={}
        for lo,hi in ((1,20),(21,40),(41,60),(61,79)):
            subset=[r for r in rows if lo<=r['horizon']<=hi]
            summary['horizon_bins'][f'{lo}-{hi}']={str(b):stats([r[f'b{b}']['time_ms'] for r in subset]) for b in (8,4,2)}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        runtime.close()
        (out/'COMPLETE').touch()
        print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
