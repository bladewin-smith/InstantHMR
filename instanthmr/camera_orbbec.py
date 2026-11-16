"""Orbbec Gemini 2 colour capture with calibration and hardware timestamps.

The dependency is imported lazily so normal image/video/OpenCV-camera use does not require
``pyorbbecsdk2``. InstantHMR is an RGB model: this source deliberately enables only the colour
stream. Depth can be captured by a downstream application, but it must not be substituted for the
model's camera translation without a proper RGB-D registration and mesh-to-depth fit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class OrbbecCalibration:
    """Colour-camera calibration for the selected stream profile."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, float, float, float, float, float, float, float]

    @property
    def model_focal(self) -> float:
        """Scalar focal accepted by the released InstantHMR ONNX graphs."""
        return 0.5 * (self.fx + self.fy)

    @property
    def principal_point(self) -> tuple[float, float]:
        return self.cx, self.cy


@dataclass(frozen=True)
class OrbbecFrame:
    index: int
    timestamp: float
    image_rgb: np.ndarray


class OrbbecGemini2Source:
    """Synchronous RGB source for Gemini 2 through Orbbec SDK v2.

    A requested RGB profile is preferred because it avoids JPEG decoding on the Jetson CPU. If the
    camera firmware does not expose it, the SDK's default colour profile is used and converted to
    RGB here. ``serial`` selects one device when more than one Orbbec camera is attached.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 serial: str | None = None, timeout_ms: int = 1000):
        try:
            import pyorbbecsdk as ob
        except ImportError as e:
            raise RuntimeError("Orbbec SDK is not installed; run: pip install pyorbbecsdk2") from e

        self._ob = ob
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.timeout_ms = int(timeout_ms)
        self._context = ob.Context()
        devices = self._context.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("no Orbbec camera found; check the USB 3 cable and Linux udev rules")
        try:
            device = (devices.get_device_by_serial_number(serial) if serial
                      else devices.get_device_by_index(0))
        except Exception as e:
            found = [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]
            raise RuntimeError(f"Orbbec camera {serial!r} was not found; connected serials: {found}") from e

        info = device.get_device_info()
        self.device_name = info.get_name()
        self.serial = info.get_serial_number()
        self.firmware = info.get_firmware_version()
        self._pipeline = ob.Pipeline(device)
        profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        self.used_fallback_profile = False
        try:
            self._profile = profiles.get_video_stream_profile(
                self.width, self.height, ob.OBFormat.RGB, self.fps)
        except Exception:
            self._profile = profiles.get_default_video_stream_profile()
            self.used_fallback_profile = True

        self.width = int(self._profile.get_width())
        self.height = int(self._profile.get_height())
        self.fps = int(self._profile.get_fps())
        intrinsic = self._profile.get_intrinsic()
        distortion = self._profile.get_distortion()
        self.calibration = OrbbecCalibration(
            fx=float(intrinsic.fx), fy=float(intrinsic.fy), cx=float(intrinsic.cx), cy=float(intrinsic.cy),
            width=self.width, height=self.height,
            distortion=tuple(float(getattr(distortion, k)) for k in
                             ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")),
        )
        self._config = ob.Config()
        self._config.enable_stream(self._profile)
        self._started = False
        self._clock_start = time.perf_counter()
        self._device_start_us: int | None = None
        self._last_timestamp = -1.0

    @property
    def description(self) -> str:
        fallback = " (SDK default profile)" if self.used_fallback_profile else ""
        return (f"{self.device_name} serial={self.serial} firmware={self.firmware}, "
                f"RGB {self.width}x{self.height}@{self.fps}{fallback}")

    def start(self) -> None:
        if not self._started:
            self._pipeline.start(self._config)
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._pipeline.stop()
            self._started = False

    def __enter__(self) -> OrbbecGemini2Source:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __iter__(self) -> Iterator[OrbbecFrame]:
        self.start()
        while True:
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            if frames is None:
                continue
            color = frames.get_color_frame()
            if color is None:
                continue
            yield OrbbecFrame(index=int(color.get_index()), timestamp=self._timestamp(color),
                              image_rgb=_frame_to_rgb(color, self._ob))

    def _timestamp(self, color_frame) -> float:
        raw_us = int(color_frame.get_timestamp_us())
        if raw_us > 0:
            if self._device_start_us is None or raw_us < self._device_start_us:
                self._device_start_us = raw_us
            timestamp = (raw_us - self._device_start_us) / 1_000_000.0
        else:
            timestamp = time.perf_counter() - self._clock_start
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1.0 / max(self.fps, 1)
        self._last_timestamp = timestamp
        return timestamp


def _frame_to_rgb(frame, ob) -> np.ndarray:
    """Copy and convert an Orbbec colour frame to contiguous RGB uint8."""
    width, height, fmt = int(frame.get_width()), int(frame.get_height()), frame.get_format()
    raw = frame.get_data()
    # SDK wheels currently expose an ndarray, while older/source builds may expose a buffer.
    # frombuffer avoids treating bytes as a scalar string and asarray keeps ndarray testability.
    data = (np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview))
            else np.asarray(raw, dtype=np.uint8).reshape(-1))
    if fmt == ob.OBFormat.RGB:
        return data.reshape(height, width, 3).copy()
    if fmt == ob.OBFormat.BGR:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
    if fmt == ob.OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Orbbec returned an invalid MJPEG colour frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == ob.OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
    if fmt == ob.OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_UYVY)
    if fmt == ob.OBFormat.I420:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_I420)
    if fmt == ob.OBFormat.NV12:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)
    if fmt == ob.OBFormat.NV21:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV21)
    raise RuntimeError(f"unsupported Orbbec colour format: {fmt}")
