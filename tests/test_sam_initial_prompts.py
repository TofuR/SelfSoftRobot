"""Image-only initialization hints and explicit prompt/cache contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
from real_validation.perception.sam_initial import automatic_foreground_prompt,SamInitialSegmenter


def arm_image(scale=1):
    image=np.full((240,320,3),25,np.uint8)
    cv2.rectangle(image,(150,45),(174,205),(220,220,220),-1)
    # Large fixture and thin central cable must not become automatic arm hints.
    cv2.rectangle(image,(15,5),(65,225),(220,220,220),-1)
    cv2.line(image,(140,30),(140,220),(220,220,220),1)
    return cv2.resize(image,None,fx=scale,fy=scale)


class TestAutomaticPrompt(unittest.TestCase):
    def test_close_and_far_center_arm(self):
        for scale in (.5,1,2):
            image=arm_image(scale);point,info=automatic_foreground_prompt(image)
            self.assertTrue(150*scale<=point[0]<=174*scale)
            self.assertFalse(info['semantic_detection'])
            self.assertTrue(info['review_required'])

    def test_dark_polarity(self):
        point,_=automatic_foreground_prompt(255-arm_image(),'dark')
        self.assertTrue(150<=point[0]<=174)

    def test_uniform_requires_click(self):
        with self.assertRaisesRegex(ValueError,'点一下'):
            automatic_foreground_prompt(np.full((240,320,3),130,np.uint8))

    def test_equally_close_objects_require_click(self):
        image=np.zeros((240,320,3),np.uint8)
        cv2.rectangle(image,(105,35),(130,205),(220,220,220),-1)
        cv2.rectangle(image,(190,35),(215,205),(220,220,220),-1)
        with self.assertRaisesRegex(ValueError,'多个相近'):
            automatic_foreground_prompt(image)


class TestSamPromptContract(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'fake.pt';self.path.write_bytes(b'fixture')
        self.segmenter=SamInitialSegmenter()
        self.segmenter.key=(str(self.path.resolve()),self.path.stat().st_mtime_ns,'cpu')
        self.segmenter.weights_hash='test'
        self.mask=np.zeros((240,320),bool);self.mask[45:205,150:175]=True
        self.predictor=Mock()
        self.predictor.predict.return_value=(np.stack([self.mask]),np.array([.95]),None)
        self.segmenter.predictor=self.predictor

    def test_no_box_automatic_hint_and_cached_single_click(self):
        image=arm_image()
        mask,info=self.segmenter.segment(image,self.path,'cpu')
        self.assertEqual(info['prompt_source'],'automatic_image_hint')
        self.assertIsNone(self.predictor.predict.call_args_list[0].kwargs['box'])
        self.assertIsNotNone(info['automatic_box'])
        self.assertFalse(info['cache_hit'])
        _,refined=self.segmenter.segment(image,self.path,'cpu',points=[[162,120]],labels=[1])
        self.assertEqual(refined['prompt_source'],'operator')
        self.assertTrue(refined['cache_hit'])
        self.assertEqual(self.predictor.set_image.call_count,1)
        for key in ('load_ms','encode_ms','decode_ms','elapsed_ms'):
            self.assertGreaterEqual(refined[key],0)
        self.assertIsNone(self.predictor.predict.call_args_list[2].kwargs['box'])
        self.assertIsNotNone(refined['automatic_box'])

    def test_similar_score_containing_mask_preserves_complete_arm(self):
        distal=self.mask.copy();distal[:120]=False
        self.predictor.predict.return_value=(np.stack([distal,self.mask]),np.array([.66,.64]),None)
        mask,info=self.segmenter.segment(arm_image(),self.path,'cpu')
        self.assertEqual(info['selection'],'automatic_box_refinement')
        np.testing.assert_array_equal(mask>0,self.mask)

    def test_box_cleanup_does_not_grow_base_into_fixture(self):
        grown=self.mask.copy();grown[25:45,150:175]=True
        self.predictor.predict.side_effect=[
            (np.stack([self.mask]),np.array([.75]),None),
            (np.stack([grown]),np.array([.95]),None)]
        mask,info=self.segmenter.segment(arm_image(),self.path,'cpu')
        self.assertEqual(info['selection'],'automatic_box_refinement')
        self.assertFalse(mask[:43].any())
        self.assertTrue(mask[100,160])

    def test_unrelated_box_refinement_keeps_initial_candidate(self):
        other=np.zeros_like(self.mask);other[45:205,30:55]=True
        self.predictor.predict.side_effect=[
            (np.stack([self.mask]),np.array([.75]),None),
            (np.stack([other]),np.array([.99]),None)]
        mask,info=self.segmenter.segment(arm_image(),self.path,'cpu')
        self.assertEqual(info['selection'],'highest_prompt_score')
        np.testing.assert_array_equal(mask>0,self.mask)

    def test_new_frame_reencodes(self):
        image=arm_image()
        self.segmenter.segment(image,self.path,'cpu')
        image[0,0]=100
        _,info=self.segmenter.segment(image,self.path,'cpu')
        self.assertFalse(info['cache_hit'])
        self.assertEqual(self.predictor.set_image.call_count,2)

    def test_border_background_candidate_is_not_automatic_arm(self):
        self.predictor.predict.return_value=(np.ones((1,240,320),bool),np.array([.99]),None)
        with self.assertRaisesRegex(ValueError,'点一下'):
            self.segmenter.segment(arm_image(),self.path,'cpu')

    def test_operator_box_and_guide_preserved(self):
        _,info=self.segmenter.segment(arm_image(),self.path,'cpu',roi=[145,40,180,210],
                                     guide=np.array([[162,45],[162,100],[162,150],[162,205]]))
        self.assertEqual(info['box'],[145.,40.,180.,210.])
        self.assertIsNone(info['automatic_prompt'])
        self.assertEqual(info['labels'],[1,1])


if __name__=='__main__':unittest.main()
