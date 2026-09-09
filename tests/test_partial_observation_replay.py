"""Focused invariants for offline partial-image feedback; no hardware access."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from tests.test_hereditary_geometry_model import _model
from src.control.hereditary_feedback import (
    ActionBounds, block_basis, correct_state, correct_suffix, physical_shape, rollout)
from src.control.partial_image import EdgeEvidence, extract_edges
from src.evaluation.partial_replay_data import load_replay_data
from scripts.evaluation.replay_partial_observation import fixed_occlusion, hidden_nodes, select_motion_window


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model=_model(residual_mode="none").eval()
        self.model.requires_grad_(False)
        self.actions=torch.full((6,4),0.4)
        self.state=self.model.init_z_from_action(self.actions[None])[0]
        self.bounds=ActionBounds(np.zeros(4),np.ones(4),np.ones(4)*.05,np.ones(4)*.05)

    def test_observe_does_not_consume_action_and_step_matches_forward(self):
        m=self.model;state=self.state.clone();action=torch.full((1,4),.7)
        expected=m.forward(action[:,None],prev_z=state[None])
        actual=m.step_state(action,state[None])
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key])
        for _ in range(3):
            observed=m.observe_state(action,actual['latent_z'])
            torch.testing.assert_close(observed,expected['skeleton'])
        torch.testing.assert_close(state,self.state,rtol=0,atol=0)
        second=m.step_state(action,actual['latent_z'])
        self.assertFalse(torch.equal(second['latent_z'],actual['latent_z']))

    def test_observation_gradient_matches_finite_difference(self):
        fn=lambda z:physical_shape(self.model,z,self.actions[0]).reshape(-1)
        jac=torch.autograd.functional.jacobian(fn,self.state)
        direction=torch.linspace(-.1,.1,len(self.state))
        epsilon=.002
        finite=(fn(self.state+epsilon*direction)-fn(self.state-epsilon*direction))/(2*epsilon)
        torch.testing.assert_close(jac@direction,finite,atol=1e-3,rtol=.03)

    def test_observer_empty_evidence_preserves_state(self):
        corrected,info=correct_state(self.model,self.state,self.actions[0],lambda z:z[:0],
                                     torch.zeros(4),torch.ones(4))
        torch.testing.assert_close(corrected,self.state,rtol=0,atol=0)
        self.assertFalse(info['accepted'])

    def test_partial_update_uses_prior_and_improves_compatible_evidence(self):
        m=self.model
        target=physical_shape(m,self.state+.03,self.actions[0]).detach()[[4,8,12]]
        residual=lambda z:(physical_shape(m,z,self.actions[0])[[4,8,12]]-target).reshape(-1)
        before=float(residual(self.state).square().sum())
        corrected,info=correct_state(m,self.state,self.actions[0],residual,torch.zeros(4),torch.ones(4),prior_std=.3)
        self.assertTrue(info['accepted'])
        self.assertLess(float(residual(corrected).square().sum()),before)
        self.assertLessEqual(float((corrected-self.state).abs().max()),.120001)
        p,_=m._unpack_state(corrected[None]);e=m.drive(self.actions[:1]).unsqueeze(-1)
        self.assertTrue(torch.all(abs(p-e)<=m.play.thresholds+1e-6))

    def test_full_suffix_qp_is_feasible_and_does_not_mutate_state(self):
        m=self.model;state=self.state.clone();old=self.actions.clone()
        other=self.actions.clone();other[:,0]+=.04
        reference=rollout(m,state,other).detach()
        before=rollout(m,state,old).detach()
        new,info=correct_suffix(m,state,old,self.actions[0],reference,self.bounds,blocks=3)
        self.assertTrue(info['solver_success'])
        self.assertTrue(info['accepted'])
        self.assertTrue(self.bounds.valid(new.numpy(),self.actions[0].numpy()))
        self.assertLess(float((rollout(m,state,new)-reference).square().mean()),float((before-reference).square().mean()))
        torch.testing.assert_close(state,self.state,rtol=0,atol=0)
        torch.testing.assert_close(old,self.actions,rtol=0,atol=0)
        self.assertTrue(np.any(new[-1].numpy()!=old[-1].numpy()))

    def test_block_basis_covers_final_row_and_rate_boundary(self):
        d=block_basis(7,4,3)
        np.testing.assert_allclose(d.sum(axis=1),np.ones(28))
        self.assertEqual(d.shape,(28,12))
        projected=self.bounds.project(np.ones((3,4))*2,np.ones(4)*.4)
        np.testing.assert_allclose(projected[:,0],[.45,.5,.55])
        self.assertTrue(self.bounds.valid(projected,np.ones(4)*.4))
        with self.assertRaises(ValueError):
            self.bounds.project(np.full((1,4),np.nan),np.ones(4)*.4)

    def test_saturated_float32_rates_do_not_make_constant_block_qp_infeasible(self):
        previous=np.ones(4)*.4
        old=torch.tensor(self.bounds.project(np.ones((8,4))*.9,previous),dtype=torch.float32)
        reference=rollout(self.model,self.state,old-.015).detach()
        _,info=correct_suffix(self.model,self.state,old,torch.tensor(previous,dtype=torch.float32),
                              reference,self.bounds,blocks=2)
        self.assertTrue(info['solver_success'],info)


class ImageAndWindowTests(unittest.TestCase):
    def test_fixed_square_replaces_pixels_and_does_not_follow_motion(self):
        camera=np.stack([np.ones(15)*90,np.linspace(20,190,15)],axis=1)
        box=fixed_occlusion(camera,40,(220,200,3))
        raw=np.full((220,200,3),180,np.uint8)
        images=[]
        for _ in range(3):
            image=raw.copy();x,y,w,h=box;image[y:y+h,x:x+w]=35;images.append(image)
        for image in images:
            self.assertTrue(np.all(image[y:y+h,x:x+w]==35))
        self.assertTrue(np.all(raw==180))
        self.assertNotEqual(hidden_nodes(camera,box).sum(),hidden_nodes(camera+[100,0],box).sum())

    def test_auto_edges_reject_horizontal_occlusion_cut_and_all_blind(self):
        image=np.full((220,200,3),25,np.uint8)
        cv2.rectangle(image,(80,15),(100,205),(220,220,220),-1)
        image[90:130,60:120]=35
        center=np.stack([np.ones(15)*90,np.linspace(20,200,15)],axis=1)
        evidence=extract_edges(image,center,radius=10,search=8)
        self.assertGreater(len(evidence.pixels),8)
        self.assertFalse(np.any((evidence.pixels[:,1]>94)&(evidence.pixels[:,1]<126)))
        blank=extract_edges(np.full_like(image,35),center,radius=10)
        self.assertEqual(len(blank.pixels),0)

    def test_edge_distance_sign_and_derivative(self):
        line=torch.tensor([[0.,0.],[0.,10.]],requires_grad=True)
        edge=EdgeEvidence(np.array([[2.,5.]],np.float32),np.array([0]),np.array([1.]))
        residual=edge.residual(line,2,sigma=1)
        self.assertAlmostEqual(float(residual.detach()[0]),0)
        gradient=torch.autograd.grad(residual.sum(),line)[0]
        self.assertAlmostEqual(float(gradient[:,0].sum()),-1)

    def test_motion_selection_ignores_static_prefix_and_preserves_dev_bounds(self):
        positions=np.zeros((40,15,2),np.float32)
        positions[22:,1:,0]=np.arange(18)[:,None]*2
        start,candidates=select_motion_window(positions,5,10)
        self.assertGreaterEqual(start,22)
        self.assertEqual(len(candidates),26)
        self.assertLessEqual(start+10,len(positions))


class DataPairingTests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';raw.mkdir();(raw/'cam0').mkdir()
        dev=root/'split/dev';dev.mkdir(parents=True)
        channels=[0,1,3,5];expansion=[0,1,1,2,2,3]
        meta={'action_interval_s':.1,'lo6':[0]*6,'hi6':[150]*6}
        (raw/'meta.json').write_text(json.dumps(meta))
        def csv_file(name,rows):
            with (raw/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        action_rows=[];samples=[];commands=[]
        for i in range(4):
            action_rows.append({'t_sec':i*.1+.09,**{f'c{c}':i+1 for c in range(6)}})
            samples.append({'frame_idx':i,'command_id':10+i,'t_grab':i*.1+.09,'frame_age0':.01})
            commands.append({'command_id':10+i,'t_command':i*.1,'communication_status':'ack',
                             **{f'action_command{c}':i+1 for c in range(6)}})
            cv2.imwrite(str(raw/'cam0'/f'{i:05d}.png'),np.zeros((8,8,3),np.uint8))
        csv_file('actions6.csv',action_rows);csv_file('samples.csv',samples);csv_file('commands.csv',commands)
        npz=dev/'seq_train.npz'
        np.savez(npz,actions=np.tile(np.arange(1,5)[:,None],(1,6))/150,
                 raw_action_scale6_kpa=np.ones(6)*150,action_scale_kpa=np.ones(4)*150,
                 positions=np.zeros((4,3,15)),positions_camera_px=np.zeros((4,3,15)),
                 model_action_channels=channels,action_expansion6=expansion,node_order='base_to_tip',
                 state_coordinate_frame='robot_planar_mm_v1',state_length_unit='mm',
                 skeleton_frame_transform=json.dumps({'model_to_camera_matrix':np.eye(3).tolist()}),
                 evaluation_mask=[False,True,True,True],robot_diameter_px=20.)
        (dev.parent/'split_manifest.json').write_text(json.dumps({'roles':{'dev':{'files':[
            {'source':'old/seq_train.npz','source_slice':[0,4],'evaluation_start':1}]}}}))
        cfg={'action_view':{'model_action_channels':channels},'node_order':'base_to_tip',
             'state_view':{'state_coordinate_frame':'robot_planar_mm_v1'}}
        return npz,raw,cfg

    def test_pairing_checks_units_coordinates_and_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,r,c=self.fixture(Path(tmp))
            result=load_replay_data(p,r,c,model_dt=.1,norm_factor=.5)
            np.testing.assert_allclose(result.actions[:,0],np.arange(1,5)/75)
            self.assertEqual(result.audit['invalid_time_pairs'],0)
            self.assertEqual(result.evaluation_start,1)
            c['node_order']='tip_to_base'
            with self.assertRaisesRegex(ValueError,'node order'):
                load_replay_data(p,r,c,model_dt=.1,norm_factor=.5)

    def test_mismatched_recorded_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,r,c=self.fixture(Path(tmp))
            text=(r/'commands.csv').read_text().replace('0.0,ack,1','0.0,ack,9',1)
            (r/'commands.csv').write_text(text)
            with self.assertRaisesRegex(ValueError,'receipt mismatch'):
                load_replay_data(p,r,c,model_dt=.1,norm_factor=.5)


if __name__=='__main__':
    unittest.main()
