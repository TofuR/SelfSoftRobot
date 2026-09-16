import unittest
import numpy as np
from src.benchmarks.modeling_fast_training import validation_mean
from src.benchmarks.modeling_models import make_model
from tests.test_modeling_geometry_calibration import geometry_config
import torch


class PooledTrainingTests(unittest.TestCase):
    def test_frame_weighted_validation_uses_sample_counts(self):
        errors=np.array([1.,1.,1.,9.]);groups=np.array([0,0,0,1])
        self.assertEqual(validation_mean(errors,groups,'pooled_frames'),3.)
        self.assertEqual(validation_mean(errors,groups,'sequence_macro'),5.)
        with self.assertRaises(ValueError):validation_mean(errors,groups,'unknown')

    def test_calibrated_factory_static_reconstruction_keeps_trainable_reference(self):
        cfg=dict(calibrate_reference=True,history=5,hidden=8,dt=.2)
        model,metadata=make_model('hov_no_memory',cfg,normalization=([0.,0.,0.],100.),geometry_config=geometry_config())
        self.assertTrue(model.core.reference_bend_bias.requires_grad)
        other,_=make_model('hov_no_memory',cfg,normalization=([0.,0.,0.],100.),geometry_config=metadata)
        other.load_state_dict(model.state_dict(),strict=True)
        x=torch.rand(2,5,4)
        torch.testing.assert_close(model(x),other(x))


if __name__=='__main__':unittest.main()
