"""Chronological pooled splits, source frame IDs, and new modeling baselines."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.benchmarks.modeling_data import prepare_temporal_pool,load_sequences,CausalWindows,sha256,write_json
from src.benchmarks.modeling_foundations import PCCShape,KoopmanShape
from src.benchmarks.modeling_runner import _predict
from src.benchmarks.modeling_models import Polynomial
from tests.test_modeling_benchmark import synthetic_sequence


class SingleHoldoutTests(unittest.TestCase):
    def test_exact_ratios_disjoint_history_and_original_frame_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rows=[]
            inventory=root/'masks.json';write_json(inventory,[])
            for i,n in enumerate((101,103,107,109,113,127,140)):
                seq=synthetic_sequence(i,n);path=root/f's{i}.npz'
                np.savez(path,**seq,frame_ids=np.arange(n),timestamps=np.arange(n)*.2,model_to_mask=np.eye(3))
                rows.append(dict(group=f's{i}',role='test' if i==6 else 'val' if i==5 else 'train',
                                 frames=n,path=path.name,sha256=sha256(path),masks=str(root),mask_shape=[200,100],
                                 mask_inventory=inventory.name,mask_inventory_sha256=sha256(inventory)))
            source=root/'source.json';write_json(source,dict(schema='shape_modeling_grouped_v1',files=rows,
                dt=.2,length_unit='mm',node_order='base_to_tip',label_source='test fixture',fold=0))
            target=prepare_temporal_pool(source,root/'pooled',history=5)
            meta,sequences=load_sequences(target)
            total=sum(r['frames'] for r in rows)
            self.assertEqual(meta['counts']['train'],round(.6*total))
            self.assertEqual(meta['counts']['val'],round(.2*total))
            self.assertEqual(sum(meta['counts'].values()),total)
            for group in [r['group'] for r in rows]:
                slices=[s for s in sequences if s['record']['group']==group]
                flat=np.concatenate([s['frame_ids'] for s in slices])
                np.testing.assert_array_equal(flat,np.arange(len(flat)))
                self.assertEqual(len({int(f) for f in flat}),len(flat))
            test=[s for s in sequences if s['record']['role']=='test']
            cfg=dict(history=5,eval_stride=1,batch_size=16)
            predictions=_predict(Polynomial('mean'),test,cfg,torch.zeros(3),1.,'cpu')
            for seq,frames,_,_ in predictions:
                self.assertEqual(frames[0],seq['record']['start']+4)
                self.assertEqual(frames[-1],seq['record']['stop']-1)
            # Reject overlap even when only the training role is requested.
            meta['files'][0]['stop']+=1;write_json(target,meta)
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                load_sequences(target,roles=('train',))

    def test_actual_slice_length_must_match_declared_interval(self):
        for role in ('train', 'val', 'test'):
            for delta in (-1, 1):
                with self.subTest(role=role, delta=delta), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    rows = []
                    for i, split in enumerate(('train', 'val', 'test')):
                        start, stop = i * 10, (i + 1) * 10
                        frames = 10 + (delta if split == role else 0)
                        path = root / f'{split}.npz'
                        np.savez(path, **synthetic_sequence(frames=frames),
                                 frame_ids=np.arange(start, start + frames),
                                 timestamps=np.arange(start, start + frames) * .2)
                        rows.append(dict(group='s0', role=split, start=start, stop=stop,
                                         original_frames=30, parent_sha256='same-parent',
                                         frames=frames, path=path.name, sha256=sha256(path)))
                    manifest = root / 'manifest.json'
                    write_json(manifest, dict(schema='shape_modeling_temporal_pool_v1',
                               length_unit='mm', node_order='base_to_tip', dt=.2, files=rows))
                    # Arrays, frame IDs and hashes agree with frames, but not start/stop.
                    # Validate even excluded roles before opening any slice arrays.
                    for roles in (('train', 'val', 'test'), ('train',), ()):
                        with self.subTest(roles=roles), self.assertRaisesRegex(ValueError, 'frames.*stop-start'):
                            load_sequences(manifest, roles=roles)

    def test_pcc_straight_limit_and_koopman_causality(self):
        model=PCCShape(16,dict(rest_lengths=[70.,70.],base=[0.,0.,0.]),([0.,0.,0.],1.))
        action=torch.rand(2,20,4)
        shape=model(action)
        torch.testing.assert_close(shape[0,:,1],torch.arange(15,dtype=torch.float32)*10)
        torch.testing.assert_close(shape[0,:,0],torch.zeros(15),atol=1e-5,rtol=0)
        shape.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        koopman=KoopmanShape(16,8)
        action.requires_grad_();koopman(action).sum().backward()
        self.assertGreater(float(action.grad[:,:-1].abs().sum()),0.)

if __name__=='__main__':unittest.main()
