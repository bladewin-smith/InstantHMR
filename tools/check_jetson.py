#!/usr/bin/env python3
"""Check the Jetson/Orbbec runtime before downloading or building model engines.

    python tools/check_jetson.py
    python tools/check_jetson.py --camera
"""
from __future__ import annotations

import argparse
import platform
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", action="store_true", help="open Gemini 2 and acquire one RGB frame")
    p.add_argument("--serial", help="Orbbec serial number when more than one camera is connected")
    args = p.parse_args()
    failed = False

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {sys.version.split()[0]}")

    try:
        import torch
        print(f"torch:    {torch.__version__}; CUDA available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:      {torch.cuda.get_device_name(0)}")
        else:
            print("ERROR: RF-DETR needs a CUDA-enabled Jetson PyTorch build")
            failed = True
    except Exception as e:
        print(f"ERROR: cannot import torch: {e}")
        failed = True

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ORT:      {ort.__version__}; providers={providers}")
        if "CUDAExecutionProvider" not in providers:
            print("ERROR: ONNX Runtime has no CUDAExecutionProvider")
            failed = True
        if "TensorrtExecutionProvider" not in providers:
            print("NOTE: TensorRT EP is unavailable; --device cuda remains supported")
    except Exception as e:
        print(f"ERROR: cannot import onnxruntime: {e}")
        failed = True

    try:
        import tensorrt
        print(f"TensorRT: {tensorrt.__version__}")
    except Exception as e:
        print(f"NOTE: TensorRT Python module is unavailable: {e}")

    try:
        import pyorbbecsdk as ob
        print(f"Orbbec:   SDK {ob.get_version()}")
    except Exception as e:
        print(f"ERROR: cannot import pyorbbecsdk (package pyorbbecsdk2): {e}")
        failed = True

    if args.camera:
        try:
            from instanthmr import OrbbecGemini2Source

            with OrbbecGemini2Source(serial=args.serial) as source:
                frame = next(iter(source))
                c = source.calibration
                print(f"camera:   {source.description}")
                print(f"frame:    {frame.image_rgb.shape}, t={frame.timestamp:.6f}s")
                print(f"intrinsic fx={c.fx:.2f} fy={c.fy:.2f} cx={c.cx:.2f} cy={c.cy:.2f}")
        except Exception as e:
            print(f"ERROR: Gemini 2 capture failed: {e}")
            failed = True

    print("result:   " + ("FAILED" if failed else "OK"))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
