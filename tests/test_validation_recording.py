"""Camera selection and experimental records preserve source evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from real_validation.hardware.profile import HardwareProfile
from real_validation.hardware.camera import camera_specs,CameraHardwareError,OpenCVCam
from real_validation.perception.software_occlusion import OcclusionConfig
from real_validation.execution.experiment_archive import ExperimentArchive
from real_validation.execution.executor import CommandReceipt

class RecordingTests(unittest.TestCase):
    def test_multiple_masks_preserve_raw_and_scale_with_resolution(self):
        raw=np.full((100,200,3),225,np.uint8)
        config=OcclusionConfig(True,((10,20,30,40,0),(70,70,20,20,100)))
        used,meta=config.apply(raw)
        self.assertTrue(np.all(raw==225));self.assertTrue(np.all(used[20:60,20:80]==0))
        self.assertTrue(np.all(used[70:90,140:180]==100));self.assertEqual(used[0,0,0],225)
        self.assertEqual(len(meta['rectangles']),2)
        with self.assertRaises(ValueError):OcclusionConfig(True,((90,0,20,10,0),))
        total,_=OcclusionConfig(True,((0,0,100,100,35),)).apply(raw)
        self.assertTrue(np.all(total==35))

    def test_camera_selection_is_explicit_and_does_not_reuse_serials(self):
        with patch('real_validation.hardware.camera.RealSenseCam.list_devices',return_value=['D435','D455']):
            driver,ids=camera_specs(HardwareProfile.real(camera_count=2))
            self.assertEqual((driver,ids),('realsense',['D435','D455']))
            self.assertEqual(camera_specs(HardwareProfile.real(camera_driver='opencv',camera_sources=(3,))),('opencv',[3]))
        with patch('real_validation.hardware.camera.RealSenseCam.list_devices',return_value=[]):
            self.assertEqual(camera_specs(HardwareProfile.real()),('opencv',[0]))
            with self.assertRaises(CameraHardwareError):camera_specs(HardwareProfile.real(camera_driver='realsense'))
        with self.assertRaises(ValueError):HardwareProfile(camera_count=2,camera_sources=(0,0))

    def test_archive_stores_raw_and_feedback_and_optional_ndi_without_filling_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'frames').mkdir()
            raw=np.full((60,80,3),220,np.uint8);modified,mask=OcclusionConfig(True,((0,0,50,50,0),)).apply(raw)
            receipt=CommandReceipt('id',(1.,)*6,(.8,)*6,10.,10.01,'ack')
            provider=lambda start,end:dict(backend='mock',state='ready',connected=True,probe_count=2,
                samples=[(10.05,[1.,2.,3.,0.,0.,0.,1.,0.,0.,0.,.9]+[float('nan')]*11),(10.12,[2.]*22)])
            archive=ExperimentArchive(folder,10.,{'kind':'test'},provider)
            archive.command(0,receipt);archive.image(0,0,(raw,10.1),receipt,modified,mask);archive.close('failed')
            np.testing.assert_array_equal(cv2.imread(str(folder/'raw/cam0/00000.png')),raw)
            np.testing.assert_array_equal(cv2.imread(str(folder/'frames/00000.png')),modified)
            with (folder/'ndi.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4);self.assertEqual(rows[1]['valid'],'False')
            with (folder/'frame_ndi.csv').open() as f:links=list(csv.DictReader(f))
            self.assertEqual(float(links[0]['ndi_timestamp']),10.05)
            self.assertAlmostEqual(float(links[0]['ndi_age_ms']),50.)
            self.assertEqual(json.loads((folder/'metadata.json').read_text())['status'],'failed')
        with tempfile.TemporaryDirectory() as directory:
            archive=ExperimentArchive(directory,10.,{},None);archive.close('completed')
            meta=json.loads((Path(directory)/'metadata.json').read_text())
            self.assertEqual(meta['ndi_samples'],0);self.assertFalse(meta['ndi']['connected'])

    def test_background_images_own_pixels_and_bound_backlog(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'frames').mkdir()
            archive=ExperimentArchive(folder,10.,{},None)
            release=threading.Event();entered=threading.Event();write=cv2.imwrite
            def blocked(path,pixels):
                entered.set();release.wait(2.);return write(path,pixels)
            raw=np.full((10,10,3),220,np.uint8)
            receipt=CommandReceipt('id',(1.,)*6,(1.,)*6,10.,10.01,'ack')
            try:
                with patch('real_validation.execution.experiment_archive.cv2.imwrite',blocked):
                    archive.enqueue_image(0,0,(raw,10.1),receipt)
                    self.assertTrue(entered.wait(1.));raw[:]=0
                    for i in range(1,16):archive.enqueue_image(i,0,(raw,10.1),receipt)
                    with self.assertRaisesRegex(RuntimeError,'积压'):archive.enqueue_image(16,0,(raw,10.1),receipt)
                    release.set();archive.close('failed')
                self.assertTrue(np.all(cv2.imread(str(folder/'raw/cam0/00000.png'))==220))
                with (folder/'samples.csv').open() as handle:rows=list(csv.DictReader(handle))
                self.assertEqual(len(rows),16);self.assertGreater(float(rows[0]['image_write_ms']),0)
            finally:release.set()

    def test_uvc_driver_reads_frame_and_releases_capture(self):
        from PyQt5.QtCore import Qt
        class Capture:
            released=False
            def isOpened(self):return True
            def set(self,*args):pass
            def read(self):return True,np.full((20,30,3),100,np.uint8)
            def release(self):self.released=True
        capture=Capture();camera=OpenCVCam(2);frames=[]
        def got(frame,stamp):frames.append(frame);camera._running=False
        camera.frame_ready.connect(got,Qt.DirectConnection)
        with patch('cv2.VideoCapture',return_value=capture) as open_camera:camera.run()
        open_camera.assert_called_once_with(2);self.assertEqual(len(frames),1);self.assertTrue(capture.released)

if __name__=='__main__':unittest.main()
