from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from demo import parse_args
from instanthmr.camera_orbbec import OrbbecGemini2Source, _frame_to_rgb
from instanthmr.inference import HMRPrediction, InstantHMR


class _Formats:
    RGB = 1
    BGR = 2
    MJPG = 3
    YUYV = 4
    UYVY = 5
    I420 = 6
    NV12 = 7
    NV21 = 8


class _Frame:
    def __init__(self, data, fmt, width=2, height=2, timestamp=1_000_000):
        self.data, self.fmt = np.asarray(data, np.uint8), fmt
        self.width, self.height, self.timestamp = width, height, timestamp

    def get_data(self):
        return self.data

    def get_format(self):
        return self.fmt

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_index(self):
        return 7

    def get_timestamp_us(self):
        return self.timestamp


class CameraConversionTests(unittest.TestCase):
    def setUp(self):
        self.ob = types.SimpleNamespace(OBFormat=_Formats)

    def test_rgb_is_copied(self):
        data = np.arange(12, dtype=np.uint8)
        image = _frame_to_rgb(_Frame(data, _Formats.RGB), self.ob)
        data[:] = 0
        self.assertEqual(image.shape, (2, 2, 3))
        self.assertNotEqual(int(image.sum()), 0)

    def test_rgb_accepts_buffer_data(self):
        frame = _Frame(np.arange(12, dtype=np.uint8), _Formats.RGB)
        frame.get_data = lambda: bytes(frame.data)
        image = _frame_to_rgb(frame, self.ob)
        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(image.reshape(-1).tolist(), list(range(12)))

    def test_bgr_is_converted_to_rgb(self):
        image = _frame_to_rgb(_Frame([1, 2, 3] * 4, _Formats.BGR), self.ob)
        self.assertEqual(image[0, 0].tolist(), [3, 2, 1])

    def test_mjpeg_is_decoded_to_rgb(self):
        bgr = np.zeros((2, 2, 3), np.uint8)
        bgr[..., 2] = 255
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        image = _frame_to_rgb(_Frame(encoded, _Formats.MJPG), self.ob)
        self.assertGreater(int(image[0, 0, 0]), 240)


class OrbbecSourceTests(unittest.TestCase):
    def _sdk(self):
        profile = types.SimpleNamespace(
            get_width=lambda: 2,
            get_height=lambda: 2,
            get_fps=lambda: 30,
            get_intrinsic=lambda: types.SimpleNamespace(fx=100.0, fy=102.0, cx=1.0, cy=1.0),
            get_distortion=lambda: types.SimpleNamespace(k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0),
        )
        color = _Frame(np.arange(12), _Formats.RGB)

        class Pipeline:
            def __init__(self, device):
                self.started = False

            def get_stream_profile_list(self, sensor):
                return types.SimpleNamespace(get_video_stream_profile=lambda *args: profile,
                                             get_default_video_stream_profile=lambda: profile)

            def start(self, config):
                self.started = True

            def stop(self):
                self.started = False

            def wait_for_frames(self, timeout):
                return types.SimpleNamespace(get_color_frame=lambda: color)

        info = types.SimpleNamespace(get_name=lambda: "Gemini 2", get_serial_number=lambda: "ABC123",
                                     get_firmware_version=lambda: "1.4.98")
        device = types.SimpleNamespace(get_device_info=lambda: info)
        devices = types.SimpleNamespace(get_count=lambda: 1, get_device_by_index=lambda index: device,
                                        get_device_by_serial_number=lambda serial: device,
                                        get_device_serial_number_by_index=lambda index: "ABC123")

        class Config:
            def enable_stream(self, selected):
                self.selected = selected

        return types.SimpleNamespace(
            Context=lambda: types.SimpleNamespace(query_devices=lambda: devices), Pipeline=Pipeline, Config=Config,
            OBSensorType=types.SimpleNamespace(COLOR_SENSOR=1), OBFormat=_Formats,
        )

    def test_source_reads_calibration_timestamp_and_frame(self):
        with patch.dict(sys.modules, {"pyorbbecsdk": self._sdk()}):
            source = OrbbecGemini2Source(width=2, height=2, fps=30)
            self.assertEqual(source.calibration.model_focal, 101.0)
            self.assertEqual(source.calibration.principal_point, (1.0, 1.0))
            with source:
                frame = next(iter(source))
                self.assertEqual(frame.image_rgb.shape, (2, 2, 3))
                self.assertEqual(frame.timestamp, 0.0)
            self.assertIn("ABC123", source.description)


class PredictionTests(unittest.TestCase):
    def test_crop_uses_supplied_principal_point(self):
        hmr = InstantHMR.__new__(InstantHMR)
        hmr.cliff_focal = True
        hmr.input_size = 16
        image = np.zeros((80, 100, 3), np.uint8)
        _, cliff, _ = hmr._crop(image, np.array([40, 30, 60, 50]), 1.0, 100.0, (50.0, 40.0))
        np.testing.assert_allclose(cliff[:2], 0.0)

    def test_skeleton_only_prediction_has_no_vertices(self):
        prediction = HMRPrediction(
            bbox=np.zeros(4), confidence=1.0, joints_2d=np.zeros((70, 2)),
            joints_3d_local=np.zeros((70, 3)), cam_trans=np.zeros(3), mhr_params=np.zeros(204),
            shape_params=np.zeros(45), focal_length=100.0, principal_point=(10.0, 20.0),
            image_shape=(40, 30), vertices_local=None,
        )
        self.assertIsNone(prediction.vertices_cam)


class CliTests(unittest.TestCase):
    def test_jetson_preset_and_explicit_override(self):
        with patch.object(sys, "argv", ["demo.py", "--orbbec", "--jetson", "--fit", "0"]):
            args = parse_args()
        self.assertEqual(args.model, "small")
        self.assertEqual(args.fit, 0)
        self.assertEqual(args.detector, "nano")
        self.assertEqual(args.max_persons, 2)
        self.assertEqual(args.detector_stride, 2)
        self.assertEqual(args.trt_workspace_gb, 2.0)


if __name__ == "__main__":
    unittest.main()
