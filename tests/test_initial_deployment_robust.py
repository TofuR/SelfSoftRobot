"""Initialization evidence, scale changes, attached fixtures and manual endpoints."""
import tempfile
import threading
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tests.test_hereditary_geometry_model import _model
from src.control.hereditary_fast import FrozenHereditary
from real_validation.perception.initial_shape import extract_initial_shape,refine_initial_shape
from real_validation.runtime.hereditary_deployment import HereditaryDeployment,transform


class RobustInitialSegmentationTests(unittest.TestCase):
    def attached(self,scale=1.):
        image=np.full((320,320,3),35,np.uint8)
        cv2.rectangle(image,(144,45),(176,275),(215,215,215),-1)
        cv2.rectangle(image,(0,25),(220,50),(215,215,215),-1)
        image=cv2.resize(image,None,fx=scale,fy=scale)
        guide=np.linspace([160,50],[160,275],15)*scale
        return image,guide

    def test_manual_ends_survive_border_connected_fixture_at_multiple_scales(self):
        for scale in (.6,1.,2.):
            image,guide=self.attached(scale);draft=guide.copy();draft[1:-1,0]+=4*scale
            result,mask,info=refine_initial_shape(image,draft,search_px=0,preserve_endpoints=True)
            np.testing.assert_allclose(result[[0,-1]],draft[[0,-1]],atol=1e-6)
            self.assertLess(abs(result[2:-2,0]-160*scale).mean(),2*scale)
            self.assertTrue(info['endpoints_preserved']);self.assertTrue(info['automatic_search'])

    def test_roi_excludes_fixture_and_reports_artificial_crop_end(self):
        image,guide=self.attached()
        curve,mask,info=extract_initial_shape(image,15,roi=[130,50,190,285])
        self.assertLess(abs(curve[:,0]-160).mean(),2.)
        self.assertTrue(any('框选' in value for value in info['warnings']))
        self.assertFalse(mask[:,0].any())

    def test_sam_mask_does_not_require_white_pixels(self):
        image=np.full((300,300,3),(90,120,130),np.uint8)
        mask=np.zeros((300,300),np.uint8);cv2.rectangle(mask,(125,35),(175,265),255,-1)
        curve,_,info=extract_initial_shape(image,15,supplied_mask=mask)
        self.assertEqual(info['method'],'supplied_mask')
        np.testing.assert_allclose(curve[[0,-1]],[[150,35],[150,265]],atol=2.)


class JointCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.threads=threadpool_limits(1);self.tmp=tempfile.TemporaryDirectory()
        model=_model(residual_mode='none').eval();model.requires_grad_(False)
        self.engine=FrozenHereditary(model)
        meta=dict(dt=float(model.dt),expansion6=[0,1,1,2,2,3],action_unit_to_kpa=[150.]*4,
                  lower_kpa=[0.]*4,upper_kpa=[150.]*4,rate_kpa_s=[50.]*4,max_horizon=80,radius_mm=8.,checkpoint_sha256='test')
        self.r=HereditaryDeployment(self.engine,meta,Path(self.tmp.name)/'run',clock=lambda:100.)
        self.r.initialize([30.]*6,100.)
    def tearDown(self):self.r.close();self.tmp.cleanup();self.threads.restore_original_limits()

    def test_camera_scale_and_rotation_fit_are_scale_invariant(self):
        for pixel_length in (60.,240.,600.):
            self.r.initialize([30.]*6,100.)
            shape=self.engine.observe(self.r.state,self.r.action)
            scale=pixel_length/np.linalg.norm(np.diff(shape,axis=0),axis=1).sum()
            angle=.4;matrix=np.eye(3);matrix[:2,:2]=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);matrix[:2,2]=[230,80]
            curve=transform(shape,matrix)
            info=self.r.calibrate_full_shape(curve,100.)
            self.assertTrue(info['accepted']);self.assertLess(info['rms_fraction'],.001)
            self.assertAlmostEqual(info['scale_px_per_mm'],scale,delta=scale*.01)
            self.assertFalse(self.r.ready);self.assertFalse(self.r.alignment_confirmed)

    def test_cancel_keeps_existing_camera_and_memory(self):
        curve=self.engine.observe(self.r.state,self.r.action)*2+200
        before=self.r.state.copy();version=self.r.version;cancel=threading.Event();cancel.set()
        with self.assertRaisesRegex(ValueError,'取消'):self.r.calibrate_full_shape(curve,100.,cancel)
        np.testing.assert_array_equal(self.r.state,before);self.assertEqual(self.r.version,version)
        self.assertIsNone(self.r.matrix)

    def test_bad_shape_does_not_commit_candidate(self):
        phase=np.linspace(0,np.pi*2,self.engine.n_nodes)
        target=np.column_stack([200+100*np.cos(phase),200+100*np.sin(phase)])
        version=self.r.version
        with self.assertRaisesRegex(ValueError,'RMS'):self.r.calibrate_full_shape(target,100.)
        self.assertEqual(self.r.version,version);self.assertIsNone(self.r.matrix)
