"""Fixed-base edits must remain image-supported and local to the attachment."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
from scripts.real import finalize_modeling_labels as finalizer
from scripts.real.finalize_modeling_labels import normalize_base,remove_remote_thin_components


class BaseBoundaryTest(unittest.TestCase):
    def test_first_contact_sheet_creates_qc_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); name = 'seq_demo'; base = root/'sequences'/name
            base.mkdir(parents=True)
            (base/'image_qa.csv').write_text('frame,raw_foreground_support\n' +
                                           ''.join(f'{f},1.0\n' for f in range(4)))
            (base/'image_qa.json').write_text(json.dumps({'flagged_frames': []}))
            (base/'geometry_qa.json').write_text(json.dumps(
                {'flagged_frames': [], 'interpolated_frames': []}))
            for role in ('train', 'val'):
                folder = base/'processed_verified'/role; folder.mkdir(parents=True)
                camera = np.tile(np.array([370., 110., 0.])[None, :, None], (2, 1, 15))
                np.savez(folder/'data.npz', positions_camera_px=camera)
            raw = root/'workspace/data/raw/real'/name/'cam0'; raw.mkdir(parents=True)
            masks = base/'intermediate_verified/sam2_masks'; masks.mkdir(parents=True)
            mask = np.zeros((300, 300), np.uint8); mask[42:240, 140:161] = 255
            for f in range(4):
                cv2.imwrite(str(raw/f'{f:05d}.png'), np.full((400, 600, 3), 180, np.uint8))
                cv2.imwrite(str(masks/f'{f:05d}.png'), mask)
            self.assertFalse((root/'qc').exists())
            with patch.object(finalizer, 'ROOT', root):
                finalizer.contact_sheets(root, [name])
            self.assertIsNotNone(cv2.imread(str(root/'qc'/f'{name}_page0.png')))
            self.assertEqual(json.loads((root/'qc/selected_frames.json').read_text()),
                             {name: [0, 1, 2, 3]})

    def test_cable_noise_removed_but_occluded_arm_retained(self):
        mask=np.zeros((300,300),np.uint8)
        mask[42:140,140:161]=1;mask[145:240,140:161]=1
        mask[145:220,165:167]=1  # narrow body-adjacent visible strip
        mask[150:280,80:82]=1  # remote cable
        result,removed=remove_remote_thin_components(mask)
        self.assertEqual(removed,260)
        np.testing.assert_array_equal(result[:,130:],mask[:,130:])

    def test_visible_gap_and_bracket_are_handled_locally(self):
        image=np.zeros((300,300,3),np.uint8)
        image[42:240,140:161]=180
        mask=np.zeros((300,300),np.uint8)
        mask[55:240,140:161]=1
        mask[30:40,110:180]=1
        result,detail=normalize_base(mask,image)
        self.assertFalse(result[:42].any())
        self.assertTrue(result[42:55,140:161].all())
        np.testing.assert_array_equal(result[70:],mask[70:])
        self.assertGreater(detail['added_pixels'],0)
        self.assertEqual(detail['removed_pixels'],700)

    def test_dark_pixels_are_never_filled(self):
        image=np.zeros((300,300,3),np.uint8)
        mask=np.zeros((300,300),np.uint8);mask[55:240,140:161]=1
        result,detail=normalize_base(mask,image)
        np.testing.assert_array_equal(result,mask)
        self.assertEqual(detail['added_pixels'],0)

    def test_small_base_gap_also_includes_fixed_anchor(self):
        image=np.zeros((300,300,3),np.uint8);image[42:240,140:161]=180
        mask=np.zeros((300,300),np.uint8);mask[45:240,140:161]=1
        result,detail=normalize_base(mask,image)
        self.assertEqual(result[42,150],1)
        self.assertEqual(detail['base_gap_px'],3)

    def test_large_gap_requires_review(self):
        image=np.full((300,300,3),180,np.uint8)
        mask=np.zeros((300,300),np.uint8);mask[90:240,140:161]=1
        with self.assertRaisesRegex(ValueError,'exceeds inspected repair scope'):
            normalize_base(mask,image)

    def test_bracket_rim_below_base_is_removed(self):
        image=np.full((300,300,3),180,np.uint8)
        mask=np.zeros((300,300),np.uint8);mask[42:240,140:161]=1
        mask[42:45,120:181]=1
        result,detail=normalize_base(mask,image)
        self.assertFalse(result[42:45,120:135].any())
        self.assertTrue(result[42:240,140:161].all())
        np.testing.assert_array_equal(result[52:],mask[52:])


if __name__=='__main__':unittest.main()
