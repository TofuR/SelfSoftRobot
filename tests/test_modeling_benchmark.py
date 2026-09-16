"""Integration contracts for causal modeling comparisons and retrained ablations."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.benchmarks.modeling_data import CausalWindows, load_sequences, sha256, write_json
from src.benchmarks.modeling_models import MODEL_NAMES, last_direction, make_model, fit_normalization, bezier_sections


def synthetic_sequence(seed=0, frames=40):
    rng = np.random.default_rng(seed)
    actions = rng.uniform(0, 1, (frames, 4)).astype('float32')
    angle = np.pi/2 + actions[:, :1] * np.linspace(0, .3, 14)
    deltas = np.stack([np.cos(angle)*10, np.sin(angle)*10, np.zeros_like(angle)], -1)
    positions = np.concatenate([np.zeros((frames, 1, 3)), deltas.cumsum(1)], 1).astype('float32')
    return dict(actions=actions, positions=positions)


class ModelingModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sequences = [synthetic_sequence()]
        cls.normalization = fit_normalization(cls.sequences)
        cls.config = dict(history=5, hidden=8, dt=.2, prior_steps=2)

    def test_all_models_finite_shape_gradient_and_checkpoint_roundtrip(self):
        actions = torch.from_numpy(self.sequences[0]['actions'][:10].reshape(2, 5, 4))
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                torch.manual_seed(3)
                model, metadata = make_model(name, self.config, self.sequences, self.normalization)
                output = model(actions)
                self.assertEqual(tuple(output.shape), (2, 15, 3))
                self.assertTrue(torch.isfinite(output).all())
                if output.requires_grad:
                    output.square().mean().backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                rebuilt, _ = make_model(name, self.config, normalization=self.normalization, geometry_config=metadata)
                rebuilt.load_state_dict(model.state_dict())
                torch.testing.assert_close(output.detach(), rebuilt(actions).detach())

    def test_prior_fit_uses_the_same_supervised_targets(self):
        from unittest.mock import patch
        from src.benchmarks.modeling_models import fit_ishsm_priors_from_arrays
        config = {**self.config, 'train_stride': 2, 'max_train_windows': 12}
        windows = CausalWindows(self.sequences, 5, 2, 12)
        with patch('src.benchmarks.modeling_models.fit_ishsm_priors_from_arrays',
                   wraps=fit_ishsm_priors_from_arrays) as fit:
            make_model('hov', config, self.sequences, self.normalization)
        np.testing.assert_array_equal(fit.call_args.args[0], np.stack([self.sequences[i]['actions'][t] for i,t in windows.indices]))
        np.testing.assert_array_equal(fit.call_args.args[1], np.stack([self.sequences[i]['positions'][t] for i,t in windows.indices]))

    def test_direction_holds_and_window_initialization(self):
        actions = torch.tensor([[[0., 2.], [1., 1.], [1., 1.], [1., 1.]]])
        torch.testing.assert_close(last_direction(actions), torch.tensor([[1., -1.]]))
        torch.testing.assert_close(last_direction(actions[:, 2:]), torch.zeros(1, 2))

    def test_branch_ablations_zero_during_training(self):
        for name, disabled_index in [('hov_no_play', 0), ('hov_no_maxwell', 1)]:
            model, _ = make_model(name, self.config, self.sequences, self.normalization)
            model.train()
            result = model.core._structured_memory(torch.ones(2, 4, 2), torch.ones(2, 4, 6))
            self.assertTrue(torch.equal(result[disabled_index], torch.zeros_like(result[disabled_index])))
            self.assertGreater(result[1-disabled_index].abs().sum(), 0)

    def test_bezier_straight_arc_and_shared_joint(self):
        control = torch.tensor([[[0., 0., 0.], [0., 3.5, 0.], [0., 7., 0.], [0., 10.5, 0.], [0., 14., 0.]]])
        result = bezier_sections(control)
        torch.testing.assert_close(result[0, :, 1], torch.arange(15, dtype=torch.float32))
        torch.testing.assert_close(result[:, 0], control[:, 0])
        torch.testing.assert_close(result[:, 7], control[:, 2])
        torch.testing.assert_close(result[:, -1], control[:, -1])

    def test_external_future_does_not_enter_window(self):
        sequence = synthetic_sequence()
        dataset = CausalWindows([sequence], history=5)
        before = dataset[3][0].copy()
        sequence['actions'][8:] += 100
        np.testing.assert_array_equal(dataset[3][0], before)

    def test_window_boundaries_and_normalization(self):
        sequences = [synthetic_sequence(0, 10), synthetic_sequence(1, 12)]
        dataset = CausalWindows(sequences, history=5)
        self.assertEqual(len(dataset), 14)
        for index in range(len(dataset)):
            actions, target, group, frame = dataset[index]
            np.testing.assert_array_equal(actions, sequences[group]['actions'][frame-4:frame+1])
            np.testing.assert_array_equal(target, sequences[group]['positions'][frame])
        center, scale = fit_normalization(sequences[:1])
        sequences[1]['positions'] += 1e6
        new_center, new_scale = fit_normalization(sequences[:1])
        np.testing.assert_array_equal(center, new_center)
        self.assertEqual(scale, new_scale)

    def test_group_leakage_and_hash_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            manifest = dict(schema='shape_modeling_grouped_v1', length_unit='mm', node_order='base_to_tip', dt=.2,
                            files=[dict(group='a', role='test'), dict(group='a', role='val')])
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, 'exactly one split'):
                load_sequences(path, roles=())
            seq = synthetic_sequence(frames=10)
            data = Path(tmp)/'a.npz'
            np.savez(data, **seq, frame_ids=np.arange(10), timestamps=np.arange(10)*.2)
            manifest['files'] = [dict(group='a', role='train', path='a.npz', sha256=sha256(data), frames=10)]
            write_json(path, manifest)
            self.assertEqual(len(load_sequences(path)[1]), 1)
            with data.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_sequences(path)


class ModelingRunnerTests(unittest.TestCase):
    def test_train_select_evaluate_and_paired_summary(self):
        import cv2
        from unittest.mock import patch
        from src.benchmarks.modeling_runner import train_run, evaluate_run, aggregate
        from src.evaluation.modeling_benchmark_metrics import render_tube
        from scripts.experiments.modeling_benchmark import DEFAULTS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            matrix = np.array([[1., 0., 50.], [0., 1., 10.], [0., 0., 1.]])
            for seed, role in enumerate(('train', 'val', 'test')):
                seq = synthetic_sequence(seed, 24)
                path = root / f'{role}.npz'
                np.savez_compressed(path, **seq, frame_ids=np.arange(24), timestamps=np.arange(24)*.2,
                                    model_to_mask=matrix)
                mask_dir = root / f'{role}_masks'
                mask_dir.mkdir()
                inventory = []
                for frame, target in enumerate(seq['positions']):
                    mask_path = mask_dir / f'{frame:05d}.png'
                    cv2.imwrite(str(mask_path), render_tube(target, matrix, (180, 100), 8.).astype('uint8')*255)
                    inventory.append(dict(frame=frame, sha256=sha256(mask_path)))
                inventory_path = root / f'{role}_inventory.json'
                write_json(inventory_path, inventory)
                rows.append(dict(group=role, role=role, path=path.name, sha256=sha256(path), frames=24,
                                 masks=str(mask_dir), mask_shape=[180, 100], mask_inventory=inventory_path.name,
                                 mask_inventory_sha256=sha256(inventory_path)))
            manifest_path = root / 'manifest.json'
            write_json(manifest_path, dict(schema='shape_modeling_grouped_v1', length_unit='mm',
                       node_order='base_to_tip', dt=.2, files=rows, fold=0,
                       label_source='synthetic integration fixture', evidence_level='smoke'))
            config = {**DEFAULTS, 'run_kind': 'smoke', 'history': 5, 'epochs': 1, 'hidden': 8,
                      'max_train_windows': 4, 'max_eval_windows': 3, 'batch_size': 4, 'threads': 1}
            original_load = np.load
            def guarded_load(path, *args, **kwargs):
                if Path(path).name == 'test.npz':
                    raise AssertionError('Training opened test data')
                return original_load(path, *args, **kwargs)
            evaluations = []
            for name in ('mlp', 'mean'):
                run = root / name
                with patch('numpy.load', side_effect=guarded_load):
                    train_run(manifest_path, run, {**config, 'model': name})
                self.assertTrue((run / 'COMPLETE').exists())
                with self.assertRaises(FileExistsError):
                    train_run(manifest_path, run, {**config, 'model': name})
                evaluation = root / f'{name}_eval'
                records = evaluate_run(run, evaluation, role='val')
                self.assertEqual(records[0]['frames'], 3)
                self.assertEqual(records[0]['mask_frames'], 3)
                self.assertAlmostEqual(records[0]['label_tube_consistency']['iou'], 1.)
                arrays = np.load(evaluation / 'val_predictions.npz')
                expected_rmse = np.sqrt(np.mean(np.sum((arrays['prediction_mm']-arrays['target_mm'])**2, axis=-1)))
                self.assertAlmostEqual(records[0]['metrics']['node_rmse_mm'], expected_rmse, places=4)
                evaluations.append(evaluation)
            with self.assertRaisesRegex(ValueError, 'Smoke'):
                aggregate(evaluations, root / 'blocked_report', 'mean', ['mean_node_mm'])
            report = aggregate(evaluations, root / 'report', 'mean', ['mean_node_mm', 'mask_iou'], True)
            self.assertTrue((root / 'report/summary.csv').exists())
            self.assertFalse(report['inference_eligible'])
            self.assertEqual({c['status'] for c in report['comparisons']}, {'inconclusive'})
            self.assertEqual(next(c['direction'] for c in report['comparisons'] if c['metric']=='mask_iou'), 'higher')
            # Evaluation verifies the selected checkpoint before deserializing it.
            with (root / 'mlp/best_eval_model.pt').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                evaluate_run(root / 'mlp', root / 'tampered_eval')


if __name__ == '__main__':
    unittest.main()
