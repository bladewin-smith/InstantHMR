#!/usr/bin/env python3
"""Check the Jetson/Orbbec runtime before downloading or building model engines.

    python tools/check_jetson.py
    python tools/check_jetson.py --camera
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def classify_stack(python_version: tuple[int, int], cuda_version: str | None) -> tuple[str, str | None]:
    """Return a human-readable JetPack family and this repository's matching requirements file."""
    cuda_major = cuda_version.split(".", 1)[0] if cuda_version else None
    if python_version == (3, 10) and cuda_major == "12":
        return "JetPack 6 / CUDA 12", "requirements-jetson.txt"
    if python_version >= (3, 12) and cuda_major == "13":
        return "JetPack 7 / CUDA 13", "requirements-jetson-jp7.txt"
    return "unrecognised or mixed Jetson stack", None


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


JP6_DETECTOR_PINS = {
    "rfdetr": "1.5.1",
    "transformers": "4.57.3",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1",
    "supervision": "0.29.1",
    "av": "12.3.0",
}


def dependency_pin_issues(requirements: str | None) -> list[str]:
    """Return actionable incompatibilities before importing the detector dependency graph."""
    if requirements != "requirements-jetson.txt":
        return []
    issues = []
    for distribution, expected in JP6_DETECTOR_PINS.items():
        actual = package_version(distribution)
        if actual is None:
            issues.append(f"{distribution} is missing (expected {expected})")
        elif actual != expected:
            issues.append(f"{distribution} {actual} is incompatible (expected {expected})")
    return issues


def l4t_release() -> str | None:
    release = Path("/etc/nv_tegra_release")
    try:
        return release.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", action="store_true", help="open Gemini 2 and acquire one RGB frame")
    p.add_argument("--serial", help="Orbbec serial number when more than one camera is connected")
    args = p.parse_args()
    failed = False

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {sys.version.split()[0]}")
    release = l4t_release()
    if release:
        print(f"L4T:      {release}")
    print(f"env:      {'virtualenv' if sys.prefix != sys.base_prefix else 'system Python'}")
    if sys.prefix == sys.base_prefix:
        print("NOTE: use a --system-site-packages venv; do not mutate the ROS/JetPack system Python")

    cuda_version = None
    try:
        import torch
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        print(f"torch:    {torch.__version__}; CUDA={cuda_version}; cuDNN={cudnn_version}; "
              f"available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:      {torch.cuda.get_device_name(0)}")
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        else:
            print("ERROR: RF-DETR needs a CUDA-enabled Jetson PyTorch build")
            failed = True
    except Exception as e:
        print(f"ERROR: cannot import torch: {e}")
        failed = True

    stack, requirements = classify_stack(sys.version_info[:2], cuda_version)
    print(f"stack:    {stack}" + (f"; use {requirements}" if requirements else ""))

    try:
        import torchvision
        print(f"vision:   {torchvision.__version__}")
    except Exception as e:
        print(f"ERROR: cannot import the JetPack-compatible torchvision build: {e}")
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
        if requirements == "requirements-jetson.txt" and not ort.__version__.startswith("1.20."):
            print("NOTE: the verified JetPack 6 deployment uses ORT 1.20.x; test model loading before use")
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
        wrapper = package_version("pyorbbecsdk2") or "unknown"
        print(f"Orbbec:   wrapper={wrapper}; SDK runtime={ob.get_version()}")
    except Exception as e:
        print(f"ERROR: cannot import pyorbbecsdk (package pyorbbecsdk2): {e}")
        failed = True

    try:
        import numpy as np
        import cv2
        print(f"vision IO: NumPy {np.__version__}; OpenCV {cv2.__version__}")
        if requirements == "requirements-jetson.txt" and int(np.__version__.split('.', 1)[0]) >= 2:
            print("ERROR: the verified ROS/JetPack 6 environment requires NumPy 1.26.x")
            failed = True
    except Exception as e:
        print(f"ERROR: cannot import NumPy/OpenCV: {e}")
        failed = True

    pin_issues = dependency_pin_issues(requirements)
    if pin_issues:
        for issue in pin_issues:
            print(f"ERROR: detector dependency: {issue}")
        print("ERROR: recreate .venv-jp6 and install requirements-jetson.txt; "
              "RF-DETR 1.6+ / Transformers 5 cannot use JetPack 6 torch 2.5")
        failed = True

    rfdetr_version = package_version("rfdetr")
    if rfdetr_version is None:
        print("ERROR: RF-DETR is not installed; install the matching Jetson requirements file")
        failed = True
    else:
        try:
            import rfdetr
            required_variants = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")
            missing = [name for name in required_variants if not hasattr(rfdetr, name)]
            if missing:
                print(f"ERROR: RF-DETR {rfdetr_version} lacks required classes: {', '.join(missing)}")
                failed = True
            else:
                print(f"RF-DETR:  {rfdetr_version}; detector API OK")
        except Exception as e:
            print(f"ERROR: RF-DETR {rfdetr_version} cannot be imported: {e}")
            failed = True
    rerun_version = package_version("rerun-sdk")
    print(f"Rerun:    {rerun_version or 'not installed (only --no-viewer without --save-rrd works)'}")
    if requirements == "requirements-jetson.txt" and rerun_version and rerun_version != "0.22.1":
        print(f"ERROR: Rerun {rerun_version} is incompatible with the JP6 NumPy 1.26 pin; use 0.22.1")
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
