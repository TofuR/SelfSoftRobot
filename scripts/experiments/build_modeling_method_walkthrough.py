#!/usr/bin/env python3
"""Build an offline method tutorial with one fixed validation-window trace."""
from pathlib import Path
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.benchmarks.modeling_models import make_model


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).round(8).tolist()


def export_trace():
    torch.set_num_threads(1)
    run = ROOT / 'workspace/runs/training/modeling_unified20_20260913_004'
    checkpoint_path = run / 'formal/hov/seed_100/best_eval_model.pt'
    assert (checkpoint_path.parent / 'COMPLETE').exists(), 'Tutorial requires a completed fit'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model, _ = make_model('hov', saved['config'],
                         normalization=(np.asarray(saved['center']), saved['scale']),
                         geometry_config=saved['geometry_config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    core = model.core
    manifest = Path(json.loads((run / 'protocol.json').read_text())['dataset_manifest'])
    record = next(r for r in json.loads(manifest.read_text())['files'] if r['role'] == 'val')
    path = Path(record['path'])
    if not path.is_absolute():
        path = manifest.parent / path
    with np.load(path, allow_pickle=False) as source:
        actions = torch.from_numpy(source['actions'][:20].copy()).float()
        target = source['positions'][19].copy()
        frame_ids = source['frame_ids'][:20].copy() if 'frame_ids' in source else np.arange(record['start'], record['start'] + 20)
        timestamps = source['timestamps'][:20].copy() if 'timestamps' in source else np.arange(20) * .2
    assert actions.shape == (20, 4)
    traces = []
    with torch.inference_mode():
        drive = core.drive(actions[:1])
        p = drive.unsqueeze(-1).repeat(1, 1, core.n_play)
        h = drive.unsqueeze(-1).repeat(1, 1, core.n_maxwell)
        state = core._pack_state(p, h)
        for t in range(20):
            a = actions[t:t+1]
            drive = core.drive(a)
            if t:
                output = core.step_state(a, state)
            else:
                output = core._state_output(a, p, h, torch.zeros_like(p), drive)
            state = output['latent_z']
            p, h = core._unpack_state(state)
            q, d = drive.unsqueeze(-1) - p, h - drive.unsqueeze(-1)
            bend, length = core._reference(a)
            ref = torch.cat([bend, length], dim=1)
            def physical_coordinates(value):
                return torch.cat([value[:, :core.n_bend_modes] @ core.bend_basis.T,
                                  value[:, core.n_bend_modes:]], dim=1)
            cp = physical_coordinates(output['pi_generalized'])
            ch = physical_coordinates(output['maxwell_generalized'])
            pred = output['skeleton'] * core.pc_scale + core.pc_center
            ref_skeleton = core.decode_equilibrium(a) * core.pc_scale + core.pc_center
            traces.append(dict(t=t, frame_id=int(frame_ids[t]), timestamp=float(timestamps[t]),
                action=array(a[0]), pressure_kpa=array(a[0] * 150), drive=array(drive[0]),
                p=array(p[0]), q=array(q[0]), h=array(h[0]), d=array(d[0]),
                reference=array(ref[0]), path=array(cp[0]), time=array(ch[0]),
                total=array((ref + cp + ch)[0]), nodes=array(pred[0]), reference_nodes=array(ref_skeleton[0])))
        final = model(actions[None]) * core.pc_scale + core.pc_center
        match = float(torch.max(torch.abs(final - pred)))
        assert match < 1e-5, match
        rawp = core.play.weights[..., None] * core.pi_mode_directions * core.generalized_coordinate_scale
        rawh = core.maxwell_gains[..., None] * core.maxwell_mode_directions * core.generalized_coordinate_scale
        def effective(value):
            value = value.reshape(-1, core.generalized_dim)
            return torch.cat([value[:, :core.n_bend_modes] @ core.bend_basis.T,
                              value[:, core.n_bend_modes:]], dim=1).T
        wp, wh = effective(rawp), effective(rawh)
        readout_error = float(torch.max(torch.abs(cp - q.flatten(1) @ wp.T)))
        assert readout_error < 1e-6, readout_error
    return dict(source=str(checkpoint_path.relative_to(ROOT)), seed=100,
        selected_epoch=saved['selected_epoch'], group=record['group'], role='val',
        selection='first validation record, first 20-point window, first formal seed',
        dt=float(core.maxwell.dt), state_dim=core.operator_state_dim,
        thresholds=array(core.play.thresholds), taus=array(core.maxwell.taus),
        alphas=array(core.maxwell.decays), drive_knots=array(core.drive.knots),
        drive_weights=array(core.drive.weights), wp=array(wp), wh=array(wh),
        reference_lengths=array(core.reference_segment_lengths), base=array(core.base_position),
        section_intervals=list(core.section_intervals), trace=traces, target=array(target),
        config={
            k: saved['geometry_config'][k] for k in ('burnin_mode','drive_normalization','bend_basis_kind','n_bend_modes','residual_mode')},
        verification=dict(forward_trace_max_error_mm=match, readout_max_error=readout_error),
        checkpoint_node_error_mm=float(np.linalg.norm(np.asarray(traces[-1]['nodes'])-target, axis=-1).mean()))


def main():
    data = export_trace()
    out = ROOT / 'docs/icra2027/method_walkthrough.html'
    source = ROOT / 'scripts/experiments/assets/modeling_method_walkthrough.html'
    template = source.read_text()
    assert template.count('__REAL_TRACE_JSON__') == 1
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    out.write_text(template.replace('__REAL_TRACE_JSON__', encoded))
    evidence = out.with_suffix('.data.json')
    evidence.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(dict(html=str(out), bytes=out.stat().st_size,
                         verification=data['verification']), ensure_ascii=False))


if __name__ == '__main__':
    main()
