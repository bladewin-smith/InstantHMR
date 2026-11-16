from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from demo import parse_args
from instanthmr.camera_orbbec import OrbbecGemini2Source, _frame_to_rgb
from instanthmr.detector import PersonDetector
from instanthmr.inference import HMRPrediction, InstantHMR, _fit_providers, providers_for
from instanthmr.models import REPO_ID, _download
from instanthmr.onnx_compat import rewrite_asymmetric_conv_padding
from tools.check_jetson import classify_stack, dependency_pin_issues


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


class ModelCacheTests(unittest.TestCase):
    def test_cached_model_does_not_contact_hugging_face(self):
        download = Mock(return_value="/cache/model.onnx")
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(_download("model.onnx"), Path("/cache/model.onnx"))
        download.assert_called_once_with(REPO_ID, "model.onnx", local_files_only=True)

    def test_cache_miss_downloads_from_hugging_face(self):
        download = Mock(side_effect=[FileNotFoundError, "/cache/model.onnx"])
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(_download("model.onnx"), Path("/cache/model.onnx"))
        self.assertEqual(download.call_args_list, [
            call(REPO_ID, "model.onnx", local_files_only=True),
            call(REPO_ID, "model.onnx"),
        ])


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.ort = types.SimpleNamespace(
            get_available_providers=lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                                             "CPUExecutionProvider"]
        )

    def test_cuda_algorithm_is_forwarded_to_ort(self):
        with patch.dict(sys.modules, {"onnxruntime": self.ort}):
            providers = providers_for("cuda", cuda_conv_algo="HEURISTIC")
        self.assertEqual(providers[0], ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))

    def test_fit_graph_combines_cuda_graph_and_algorithm_options(self):
        with patch.dict(sys.modules, {"onnxruntime": self.ort}):
            providers = _fit_providers("cuda", "model", cuda_conv_algo="HEURISTIC")
        self.assertEqual(providers[0], ("CUDAExecutionProvider", {
            "enable_cuda_graph": "1", "cudnn_conv_algo_search": "HEURISTIC",
        }))

@unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is only required by the Jetson deployment")
class OnnxCompatibilityTests(unittest.TestCase):
    def test_asymmetric_conv_is_rewritten_and_numerically_equivalent(self):
        import onnx
        from onnx import TensorProto, helper
        from onnx.reference import ReferenceEvaluator

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source = root / "asymmetric.onnx"
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
            y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
            weight = helper.make_tensor("weight", TensorProto.FLOAT, [1, 1, 2, 2], [1, 2, 3, 4])
            conv = helper.make_node("Conv", ["x", "weight"], ["y"], name="conv", pads=[0, 0, 1, 1])
            model = helper.make_model(helper.make_graph([conv], "test", [x], [y], [weight]),
                                      opset_imports=[helper.make_opsetid("", 13)])
            onnx.save_model(model, source)
            sample = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
            expected = ReferenceEvaluator(model).run(None, {"x": sample})[0]

            rewritten_path = rewrite_asymmetric_conv_padding(source, root / "cache")
            rewritten = onnx.load(rewritten_path)
            actual = ReferenceEvaluator(rewritten).run(None, {"x": sample})[0]

            np.testing.assert_allclose(actual, expected)
            self.assertEqual([node.op_type for node in rewritten.graph.node], ["Conv", "Slice"])
            pads = next(attr.ints for attr in rewritten.graph.node[0].attribute if attr.name == "pads")
            self.assertEqual(list(pads), [1, 1, 1, 1])
            self.assertEqual(rewrite_asymmetric_conv_padding(source, root / "cache"), rewritten_path)

            if importlib.util.find_spec("onnxruntime"):
                import onnxruntime as ort

                optimized_path = root / "optimized.onnx"
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.optimized_model_filepath = str(optimized_path)
                ort.InferenceSession(str(rewritten_path), sess_options=options,
                                     providers=["CPUExecutionProvider"])
                optimized = onnx.load(optimized_path)
                asymmetric = [
                    list(attr.ints) for node in optimized.graph.node if node.op_type == "Conv"
                    for attr in node.attribute if attr.name == "pads"
                    and list(attr.ints[:len(attr.ints) // 2]) != list(attr.ints[len(attr.ints) // 2:])
                ]
                self.assertEqual(asymmetric, [])


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
        self.assertEqual(args.cuda_conv_algo, "HEURISTIC")
        self.assertTrue(args.onnx_pad_fix)
        self.assertTrue(args.no_viewer)

    def test_jetson_compatibility_options_can_be_overridden(self):
        with patch.object(sys, "argv", ["demo.py", "--orbbec", "--jetson",
                                        "--cuda-conv-algo", "DEFAULT", "--no-onnx-pad-fix",
                                        "--local-viewer", "--duration", "10", "--max-frames", "20",
                                        "--rerun-stride", "2"]):
            args = parse_args()
        self.assertEqual(args.cuda_conv_algo, "DEFAULT")
        self.assertFalse(args.onnx_pad_fix)
        self.assertFalse(args.no_viewer)
        self.assertEqual(args.duration, 10.0)
        self.assertEqual(args.max_frames, 20)
        self.assertEqual(args.rerun_stride, 2)


class JetsonEnvironmentTests(unittest.TestCase):
    def test_selects_verified_jetpack_6_requirements(self):
        stack, requirements = classify_stack((3, 10), "12.6")
        self.assertEqual(stack, "JetPack 6 / CUDA 12")
        self.assertEqual(requirements, "requirements-jetson.txt")

    def test_selects_jetpack_7_requirements(self):
        stack, requirements = classify_stack((3, 12), "13.2")
        self.assertEqual(stack, "JetPack 7 / CUDA 13")
        self.assertEqual(requirements, "requirements-jetson-jp7.txt")

    @patch("tools.check_jetson.package_version")
    def test_rejects_transformers_5_on_verified_jetpack_6(self, package_version):
        versions = {
            "onnx": "1.17.0", "rfdetr": "1.12.1", "transformers": "5.20.2", "huggingface-hub": "2.4.0",
            "tokenizers": "0.23.3", "supervision": "0.29.1", "av": "12.3.0",
        }
        package_version.side_effect = versions.get
        issues = dependency_pin_issues("requirements-jetson.txt")
        self.assertTrue(any("rfdetr 1.12.1" in issue for issue in issues))
        self.assertTrue(any("transformers 5.20.2" in issue for issue in issues))

    @patch("tools.check_jetson.package_version")
    def test_accepts_complete_jetpack_6_detector_pin_set(self, package_version):
        versions = {
            "onnx": "1.17.0", "rfdetr": "1.5.1", "transformers": "4.57.3", "huggingface-hub": "0.36.0",
            "tokenizers": "0.22.1", "supervision": "0.29.1", "av": "12.3.0",
        }
        package_version.side_effect = versions.get
        self.assertEqual(dependency_pin_issues("requirements-jetson.txt"), [])


class DetectorCompatibilityTests(unittest.TestCase):
    def test_new_rfdetr_uses_uncompiled_inference_object(self):
        predictor = types.SimpleNamespace(predict=lambda *args, **kwargs: None)

        class Model:
            compile = None

            def inference(self, compile=True):
                Model.compile = compile
                return predictor

        fake = types.SimpleNamespace(RFDETRNano=Model)
        with patch.dict(sys.modules, {"rfdetr": fake}):
            detector = PersonDetector("nano", compile_model=False)
        self.assertIs(detector.model, predictor)
        self.assertFalse(Model.compile)

    def test_legacy_rfdetr_keeps_in_place_model(self):
        class Model:
            optimized = False
            compile = None

            def optimize_for_inference(self, compile=True):
                self.optimized = True
                self.compile = compile

        fake = types.SimpleNamespace(RFDETRNano=Model)
        with patch.dict(sys.modules, {"rfdetr": fake}):
            detector = PersonDetector("nano", compile_model=False)
        self.assertTrue(detector.model.optimized)
        self.assertFalse(detector.model.compile)


if __name__ == "__main__":
    unittest.main()
