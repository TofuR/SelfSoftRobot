import unittest
import cv2
import numpy as np
from scripts.experiments.train_robot_segmentation import mask_polygon

class MaskCoordinateTest(unittest.TestCase):
    def test_crop_mask_restores_source_coordinates(self):
        mask=np.zeros((80,100),np.uint8);mask[10:71,30:51]=255
        polygon,iou=mask_polygon(mask,[220,68,100,80],[640,480])
        reconstructed=np.zeros((480,640),np.uint8)
        cv2.fillPoly(reconstructed,[np.rint(polygon*[640,480]).astype('int32')],1)
        self.assertTrue(reconstructed[78:139,250:271].all())
        self.assertEqual(int(reconstructed.sum()),61*21);self.assertGreater(iou,.99)
    def test_fragmented_or_wrong_geometry_rejected(self):
        mask=np.zeros((80,100),np.uint8);mask[10:40,20:30]=255;mask[45:75,20:30]=255
        with self.assertRaisesRegex(ValueError,'disconnected'):mask_polygon(mask,[0,0,100,80],[640,480])
        with self.assertRaisesRegex(ValueError,'geometry'):mask_polygon(mask,[600,0,100,80],[640,480])
