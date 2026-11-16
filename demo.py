#!/usr/bin/env python3
"""InstantHMR demo: people in an image, a video or a live camera, shown in the Rerun viewer.

    python demo.py --image photo.jpg
    python demo.py --video clip.mp4 --pose-filter one-euro
    python demo.py --camera 0 --model small --fit 1 --device tensorrt
    python demo.py --orbbec --jetson --pose-filter one-euro
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from instanthmr import DEFAULT_FIT, DEFAULT_MODEL, MODELS, PosePipeline
from instanthmr.smoothing import TemporalSmoother
from instanthmr.visualizer import RerunVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file")
    src.add_argument("--video", help="video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="live camera, e.g. 0")
    src.add_argument("--orbbec", action="store_true", help="Orbbec Gemini 2 through Orbbec SDK v2")
    p.add_argument("--jetson", action="store_true",
                   help="8 GB Jetson preset: small/fit1, uncompiled RF-DETR nano, 2 persons, detector stride 2")
    p.add_argument("--model", default=None,
                   help=f"{', '.join(MODELS)}, or a path to an .onnx file")
    p.add_argument("--fit", type=int, default=None, choices=(0, 1, 2),
                   help="kinematic fit on 1 or 2 crops per person, 0 for the network alone")
    p.add_argument("--device", default="cuda", choices=("cuda", "tensorrt", "cpu", "coreml"))
    p.add_argument("--cuda-conv-algo", choices=("EXHAUSTIVE", "HEURISTIC", "DEFAULT"),
                   help="ONNX Runtime cuDNN convolution search (Jetson preset: HEURISTIC)")
    p.add_argument("--onnx-pad-fix", action=argparse.BooleanOptionalAction, default=None,
                   help="rewrite asymmetric Conv padding for affected cuDNN releases (Jetson preset: on)")
    p.add_argument("--max-persons", type=int, default=None)
    p.add_argument("--detector", default=None, choices=("nano", "small", "medium", "base", "large"),
                   help="RF-DETR size")
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--detector-stride", type=int, default=None, metavar="N",
                   help="run the detector every N frames and reuse its boxes in between")
    p.add_argument("--focal", type=float, help="camera focal length in pixels, if known")
    p.add_argument("--principal-point", type=float, nargs=2, metavar=("CX", "CY"),
                   help="camera principal point in pixels; Gemini 2 reads it automatically")
    p.add_argument("--trt-workspace-gb", type=float,
                   help="TensorRT workspace limit in GiB (Jetson preset: 2)")
    p.add_argument("--orbbec-serial", help="select an Orbbec camera by serial number")
    p.add_argument("--orbbec-width", type=int, default=1280)
    p.add_argument("--orbbec-height", type=int, default=720)
    p.add_argument("--orbbec-fps", type=int, default=30)
    p.add_argument("--orbbec-timeout-ms", type=int, default=1000)
    p.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"),
                   help="video: filter each person's pose across frames (more accurate and steadier)")
    p.add_argument("--smooth", action="store_true",
                   help="video: smooth the keypoints and mesh instead (steadiest, but lags on fast motion)")
    p.add_argument("--no-mesh", action="store_true", help="show the skeleton only")
    p.add_argument("--mesh-alpha", type=float, default=0.35, help="mesh opacity (default: 0.35)")
    p.add_argument("--save-rrd", metavar="PATH", help="also save the Rerun recording")
    p.add_argument("--no-viewer", action="store_true", help="do not open the Rerun viewer")
    args = p.parse_args()
    if args.jetson:
        args.model = args.model or "small"
        args.fit = 1 if args.fit is None else args.fit
        args.max_persons = 2 if args.max_persons is None else args.max_persons
        args.detector = args.detector or "nano"
        args.detector_stride = 2 if args.detector_stride is None else args.detector_stride
        args.trt_workspace_gb = 2.0 if args.trt_workspace_gb is None else args.trt_workspace_gb
        args.cuda_conv_algo = args.cuda_conv_algo or "HEURISTIC"
        args.onnx_pad_fix = True if args.onnx_pad_fix is None else args.onnx_pad_fix
    else:
        args.model = args.model or DEFAULT_MODEL
        args.fit = DEFAULT_FIT if args.fit is None else args.fit
        args.max_persons = 4 if args.max_persons is None else args.max_persons
        args.detector = args.detector or "medium"
        args.detector_stride = 1 if args.detector_stride is None else args.detector_stride
        args.onnx_pad_fix = False if args.onnx_pad_fix is None else args.onnx_pad_fix
    if args.max_persons < 1 or args.detector_stride < 1:
        p.error("--max-persons and --detector-stride must be at least 1")
    if args.focal is not None and args.focal <= 0:
        p.error("--focal must be positive")
    if args.trt_workspace_gb is not None and args.trt_workspace_gb <= 0:
        p.error("--trt-workspace-gb must be positive")
    if min(args.orbbec_width, args.orbbec_height, args.orbbec_fps, args.orbbec_timeout_ms) <= 0:
        p.error("Orbbec width, height, fps and timeout must be positive")
    if not args.orbbec and args.orbbec_serial:
        p.error("--orbbec-serial requires --orbbec")
    return args


def frames(args):
    """(index, timestamp in s, RGB frame, focal, principal point) from the chosen source."""
    principal = tuple(args.principal_point) if args.principal_point is not None else None
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"cannot read image {args.image}")
        yield 0, 0.0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.focal, principal
        return
    if args.orbbec:
        from instanthmr import OrbbecGemini2Source

        source = OrbbecGemini2Source(width=args.orbbec_width, height=args.orbbec_height,
                                     fps=args.orbbec_fps, serial=args.orbbec_serial,
                                     timeout_ms=args.orbbec_timeout_ms)
        calibration = source.calibration
        focal = args.focal if args.focal is not None else calibration.model_focal
        principal = principal if principal is not None else calibration.principal_point
        print(f"Orbbec: {source.description}")
        print(f"Orbbec RGB intrinsics: fx={calibration.fx:.2f}, fy={calibration.fy:.2f}, "
              f"cx={calibration.cx:.2f}, cy={calibration.cy:.2f}; model focal={focal:.2f}")
        with source:
            for i, frame in enumerate(source):
                yield i, frame.timestamp, frame.image_rgb, focal, principal
        return
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video or f'camera {args.camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.perf_counter(), 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t = i / fps if args.video else time.perf_counter() - start
            yield i, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.focal, principal
            i += 1
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    fit = f", kinematic fit on {args.fit} crop{'s' if args.fit > 1 else ''}" if args.fit else ""
    print(f"loading InstantHMR ({args.model}{fit}) on {args.device} and RF-DETR {args.detector}...")
    if args.onnx_pad_fix:
        print(f"Jetson compatibility: asymmetric-padding rewrite enabled; "
              f"cuDNN search={args.cuda_conv_algo or 'ORT default'}")
    principal = tuple(args.principal_point) if args.principal_point is not None else None
    workspace = (int(args.trt_workspace_gb * 1024 ** 3) if args.trt_workspace_gb is not None else None)
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector=args.detector,
                            det_confidence=args.det_confidence, max_persons=args.max_persons,
                            detector_stride=args.detector_stride, focal=args.focal, principal_point=principal,
                            pose_filter=args.pose_filter, mesh=not args.no_mesh, trt_workspace_size=workspace,
                            detector_compile=not args.jetson, cuda_conv_algo=args.cuda_conv_algo,
                            rewrite_asymmetric_padding=args.onnx_pad_fix)
    print(f"InstantHMR runs on {pipeline.hmr.provider}")
    viewer = (RerunVisualizer(pipeline.hmr.faces, args.mesh_alpha, spawn=not args.no_viewer,
                              save_path=args.save_rrd)
              if not args.no_viewer or args.save_rrd else None)
    smoother = TemporalSmoother() if args.smooth else None

    # Cameras may run for days on an embedded target. Keep only the last minute of timings there;
    # image/video inputs remain unbounded so their final median still covers the complete file.
    live = args.orbbec or args.camera is not None
    history = deque(maxlen=1800 if live else None)
    try:
        for i, t, rgb, focal, principal in frames(args):
            result = pipeline(rgb, t, focal=focal, principal_point=principal)
            persons = smoother(result.persons, t) if smoother else result.persons
            timings = {"detector": result.detector_ms, "InstantHMR": result.hmr_ms}
            if viewer is not None:
                viewer.log(i, rgb, persons, timings)
            history.append((result.detector_ms, result.hmr_ms))
            if i % 30 == 0:
                print(f"frame {i:5d}  {len(persons)} person(s)  detector {result.detector_ms:5.1f} ms  "
                      f"InstantHMR {result.hmr_ms:5.1f} ms")
    except KeyboardInterrupt:
        pass
    if len(history) > 1:
        det, hmr = np.median(np.array(history[1:]), axis=0)
        scope = "recent " if live and len(history) == history.maxlen else ""
        print(f"median over {len(history) - 1} {scope}frames: detector {det:.1f} ms, InstantHMR {hmr:.1f} ms")


if __name__ == "__main__":
    main()
