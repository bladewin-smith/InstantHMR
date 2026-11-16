# InstantHMR: Real-Time Human Mesh Recovery on Your Phone

<p align="center">
  <a href="https://huggingface.co/momolesang/InstantHMR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107" alt="Hugging Face models"></a>
  <a href="https://hal.science/hal-05854387"><img src="https://img.shields.io/badge/HAL-hal--05854387-b31b1b" alt="Paper on HAL"></a>
  <br>
  <a href="https://paperswithcode.co/api/v1/papers/2611.09195/leaderboard-badge-link?eval=47651"><img src="https://paperswithcode.co/api/v1/papers/2611.09195/leaderboard-badge.svg?eval=47651&live=1" alt="Papers with Code: EMDB 1 (Camera-Relative 3D Human Pose) leaderboard"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2611.09195/leaderboard-badge-link?eval=47652"><img src="https://paperswithcode.co/api/v1/papers/2611.09195/leaderboard-badge.svg?eval=47652&live=1" alt="Papers with Code: 3DPW test (SMPL Mesh Recovery) leaderboard"></a>
</p>

InstantHMR recovers the 3D body of every person in an image: pose, body shape, 70 keypoints in 2D
and 3D, and a full body mesh in the [Momentum Human Rig](https://github.com/facebookresearch/MHR)
(MHR). It runs in about 5 ms per person on a phone NPU and in about 1 ms on a desktop GPU.

## News

- **[2026-11-16] Jetson/Orbbec:** native Gemini 2 capture with colour intrinsics and hardware
  timestamps, an Orin Nano 8 GB preset, ARM64 dependencies, headless execution and a deployment
  guide.
- **[2026-11-10]** PyTorch checkpoints of the five models, with their `run_config.json`, on
  [Hugging Face](https://huggingface.co/momolesang/InstantHMR/tree/main/pytorch), to fine-tune them
  on your own data. See [Fine-tuning a released model](training/README.md#fine-tuning-a-released-model).
- **[2026-11-08] v1.1:** a kinematic fit inside the model graph refines each pose against the
  network's own 2D keypoints (`--fit`, on by default), and two pose filters for video
  (`--pose-filter`). See [Kinematic fit](#kinematic-fit).
- **[2026-11-06] v1.0:** first release: four model sizes and a commercially usable medium model as
  ONNX graphs, the demo, training code and the dataset pipeline. Paper on
  [HAL](https://hal.science/hal-05854387).

<p align="center">
  <img src="assets/demo_desktop.gif" height="300" alt="InstantHMR on a desktop GPU, shown in the Rerun viewer">
  <img src="assets/demo_phone.gif" height="300" alt="InstantHMR on a Samsung Galaxy S23">
</p>

## Models

Four sizes, trained on the released annotations of the
[SAM 3D Body dataset](https://huggingface.co/datasets/facebook/sam-3d-body-dataset).
Accuracy is PA-MPJPE (lower is better) under the protocol of the paper; latency is per person,
for the model alone on the GPU as measured in the paper (the desktop column with TensorRT directly,
up to a 6,890-vertex mesh).

| Model | Parameters | EMDB-1 (mm) | 3DPW (mm) | RTX 4070, TensorRT (ms) | Galaxy S23 NPU (ms) | Licence |
|---|---|---|---|---|---|---|
| `nano` | 20.0 M | 61.7 | 46.6 | 0.70 | 4.60 | SAM License, non-commercial |
| `tiny` | 20.3 M | 58.0 | 44.0 | 0.73 | 4.84 | SAM License, non-commercial |
| `small` | 24.4 M | 53.3 | 41.7 | 0.84 | 5.38 | SAM License, non-commercial |
| `medium` (default) | 32.2 M | 48.8 | 40.0 | 0.98 | 5.49 | SAM License, non-commercial |
| `medium-commercial` | 32.2 M | 72.2 | 50.7 | 0.98 | 5.49 | SAM License |

`medium-commercial` is the medium model trained only on images that allow commercial use
(Harmony4D and part of COCO). The weights download automatically from
[Hugging Face](https://huggingface.co/momolesang/InstantHMR) on first use.

### Kinematic fit

Every model also comes with a kinematic fit in the same graph. The network predicts the MHR pose
and, separately, the person's 2D keypoints; the 2D keypoints are about twice as accurate as the
projection of the 3D pose. The fit moves the pose (one Levenberg-Marquardt step through the MHR
skeleton, each keypoint weighted by its confidence) so that the projected body lands on those 2D
keypoints. It has no learned parameters and runs inside the graph in fp16, on the GPU or the phone
NPU. `--fit 1` runs it on one crop of each person (0.9 of the box); `--fit 2` (default) adds a
zoomed-in crop (0.7 of the box) in the same batch and fuses the 2D of both.

| Model | EMDB-1 (mm) | 3DPW (mm) | RTX 4070, TensorRT (ms) | Galaxy S23 NPU (ms) |
|---|---|---|---|---|
| `nano`, `--fit 0 / 1 / 2` | 61.7 / 57.2 / 55.9 | 46.6 / 44.5 / 44.1 | 0.70 / 0.85 / 1.10 | 4.60 / 5.49 / 8.73 |
| `tiny`, `--fit 0 / 1 / 2` | 58.0 / 53.8 / 52.6 | 44.0 / 42.4 / 42.0 | 0.73 / 0.87 / 1.13 | 4.84 / 5.84 / 9.20 |
| `small`, `--fit 0 / 1 / 2` | 53.3 / 49.2 / 48.1 | 41.7 / 40.3 / 40.0 | 0.84 / 1.00 / 1.30 | 5.38 / 6.37 / 10.36 |
| `medium`, `--fit 0 / 1 / 2` | 48.8 / 45.2 / 44.4 | 40.0 / 39.4 / 39.3 | 0.98 / 1.12 / 1.53 | 5.49 / 6.40 / 10.22 |
| `medium-commercial`, `--fit 0 / 1 / 2` | 72.2 / 68.8 / 68.3 | 50.7 / 47.7 / 47.5 | 0.98 / 1.12 / 1.53 | 5.49 / 6.40 / 10.22 |

The fit graphs take one person per call, and their crop scales assume the person box of the
benchmarks; the demo enlarges RF-DETR's tighter boxes to match (by 1/0.866, measured on 3DPW).

## Install

Python 3.10 to 3.12, and on Linux or Windows an NVIDIA driver 525 or newer.

```bash
git clone https://github.com/Naturalpad-studio/InstantHMR.git
cd InstantHMR
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The first `pip` line installs PyTorch for CUDA 12, which the person detector uses and onnxruntime
shares, so a single set of CUDA libraries is installed. On a Mac, or without an NVIDIA GPU, skip
it.

### NVIDIA Jetson Orin Nano 8 GB + Orbbec Gemini 2

The Jetson path is an inference deployment: the live demo, 2D/3D keypoints and full MHR mesh are
supported. Training the released models is not practical in the Orin Nano's 8 GB of shared memory;
train on a workstation/server and deploy the exported ONNX files here.

The supported reference environment for this guide is:

- Jetson Orin Nano 8 GB with adequate active cooling and USB 3;
- [JetPack 7.2.1](https://developer.nvidia.com/embedded/jetpack/downloads) / Ubuntu 24.04 /
  Python 3.12 / CUDA 13.2 / TensorRT 10.16;
- Orbbec Gemini 2 with Orbbec SDK v2 and firmware 1.4.92 or newer (Orbbec recommends 1.4.98;
  do not downgrade a newer working firmware).

JetPack 6 uses CUDA 12 and needs a different Jetson-specific ONNX Runtime build. Do not install the
JetPack 7 requirements below on JetPack 6. Check the installed release first:

```bash
uname -m                         # must print aarch64
cat /etc/nv_tegra_release
python3 --version                # 3.12 on the reference image
```

#### 1. Install the JetPack components and system prerequisites

Install JetPack 7.2.1 using NVIDIA's current Orin Nano ISO/SDK Manager instructions, boot it, then:

```bash
sudo apt update
sudo apt install -y nvidia-jetpack python3-venv python3-pip libopenblas-dev libgl1
sudo reboot
```

Use an NVMe SSD when possible: model downloads and TensorRT engine caches are much more pleasant
there than on a small microSD card. Select `MAXN SUPER` in the Jetson power-mode UI and use
`sudo jetson_clocks` while benchmarking. Keep a fan attached; throttling makes timings misleading.

#### 2. Create the Python environment

JetPack 7 provides the CUDA/TensorRT system stack. Install a CUDA-enabled ARM64 PyTorch before
RF-DETR, then the Jetson-specific requirements:

```bash
git clone https://github.com/Naturalpad-studio/InstantHMR.git
cd InstantHMR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements-jetson.txt
```

This deliberately uses `onnxruntime-gpu 1.32` for its CUDA 13 / Linux ARM64 wheel. The desktop
`onnxruntime-gpu[cuda,cudnn]==1.26.*` dependency has no aarch64 wheel and must not be used on this
Jetson. Likewise, do not install `tensorrt-cu12`: JetPack already owns the TensorRT 10.16 libraries.

Verify the accelerator stack before connecting the camera:

```bash
python tools/check_jetson.py
```

The required result is `CUDA available=True` and a `CUDAExecutionProvider`. A
`TensorrtExecutionProvider` is optional for the first run: `--device cuda` works without it. If
PyTorch reports CUDA false, replace it with a JetPack-compatible ARM64 build before continuing; a
CPU-only PyTorch makes RF-DETR unusably slow.

#### 3. Configure and verify Gemini 2

Connect Gemini 2 directly to a USB 3 port using a USB 3 data cable. Confirm that the link is not
falling back to 480 Mbit/s:

```bash
lsusb
lsusb -t                       # the camera path should show 5000M or faster
```

Install Orbbec's Linux metadata and udev rules from the installed `pyorbbecsdk2` package:

```bash
python "$(python -c 'import os, pyorbbecsdk; print(os.path.join(os.path.dirname(pyorbbecsdk.__file__), "shared", "setup_env.py"))')"
```

Reconnect the camera (or reboot), then acquire one calibrated RGB frame:

```bash
python tools/check_jetson.py --camera
```

The native camera source requests RGB 1280x720 at 30 fps to avoid MJPEG decode cost. If that exact
profile is unavailable on the installed firmware, it prints `SDK default profile` and safely
converts RGB, BGR, MJPEG, YUYV, UYVY, I420, NV12 or NV21 to RGB. With multiple Orbbec cameras, pass
`--serial SERIAL` to the check and `--orbbec-serial SERIAL` to the demo.

#### 4. First live run (safe 8 GB preset)

`--jetson` expands to `small`, `--fit 1`, RF-DETR `nano`, two people, detector stride 2 and a 2 GiB
TensorRT workspace limit. Start with CUDA and no viewer so camera/inference problems are separated
from GUI problems:

```bash
python demo.py --orbbec --jetson --device cuda --pose-filter one-euro --no-viewer
```

The console should print the Gemini 2 serial, firmware, selected stream, `fx/fy/cx/cy`, the
effective focal used by InstantHMR, active ONNX Runtime provider, person count and stage timings.
Unlike an ordinary `--camera 0`, `--orbbec` uses Orbbec SDK hardware timestamps and the selected
colour profile's real calibration. The released ONNX graph accepts one focal value, so the adapter
uses `(fx + fy) / 2` and passes the real `cx/cy` to CLIFF and the in-graph kinematic fit.

For the local Rerun UI, omit `--no-viewer`:

```bash
python demo.py --orbbec --jetson --device cuda --pose-filter one-euro
```

For a headless Jetson, save an RRD without spawning the viewer, then copy it to a workstation:

```bash
python demo.py --orbbec --jetson --device cuda --no-viewer --save-rrd jetson-run.rrd
rerun jetson-run.rrd             # run this on the workstation after copying the file
```

#### 5. Enable TensorRT

Only after CUDA works, check that `TensorrtExecutionProvider` appears in
`python tools/check_jetson.py`, then run:

```bash
python demo.py --orbbec --jetson --device tensorrt --pose-filter one-euro
```

The first run builds engines locally and can take several minutes. Never copy an RTX desktop engine
cache to Jetson. Rebuild `models/trt_cache/` after changing JetPack, TensorRT, ONNX Runtime, model,
fit mode or maximum-person count. If the TensorRT provider is missing, stay on `--device cuda` until
a matching ONNX Runtime TensorRT build is installed; installing desktop `tensorrt-cu12` is not a
valid fix on JetPack 7.

The PyPI ARM64 wheel is sufficient for CUDA, but the providers compiled into a wheel can change.
If the check reports CUDA but no TensorRT and you specifically need TensorRT, build the same ORT
release on the Jetson against JetPack's libraries (allow roughly 15 GB of free disk space):

```bash
sudo apt install -y --no-install-recommends build-essential cmake git \
  libopenblas-dev python3-dev python3-setuptools python3-wheel
cd ..
git clone --recursive --branch v1.32.0 --depth 1 https://github.com/microsoft/onnxruntime.git
cd onnxruntime
export CUDACXX=/usr/local/cuda/bin/nvcc
./build.sh --config Release --update --build --parallel 1 --build_wheel --skip_tests \
  --use_tensorrt --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native \
    onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FLASH_ATTENTION=OFF \
    onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF
python -m pip install --no-deps --force-reinstall build/Linux/Release/dist/onnxruntime_gpu-1.32.0-*.whl
cd ../InstantHMR
python tools/check_jetson.py
```

`--parallel 1` is intentional on the 8 GB board. Do not remove the existing wheel until this build
succeeds; CUDA mode is the recovery path if the TensorRT build fails.

#### Resource/quality presets

| Goal | Recommended options |
|---|---|
| Balanced 8 GB live use | `--jetson` (small, fit1, detector nano, 2 people, stride 2) |
| Lowest memory/latency | `--jetson --model nano --fit 0 --max-persons 1 --no-mesh --no-viewer` |
| Better pose quality | `--jetson --model medium --fit 2 --max-persons 1` |
| Faster detection | raise `--detector-stride` to `3` or `4` |
| Full original desktop defaults | explicitly use `--model medium --fit 2 --detector medium --max-persons 4 --detector-stride 1` |

`--no-viewer` now skips Rerun initialization and per-frame serialization unless `--save-rrd` is
also present. `--no-mesh` now also skips the 18,439-vertex MHR body output instead of merely hiding
it. Live sources retain only the latest 1,800 timing samples rather than growing an unbounded list.
These are useful recovery options if the process is killed under memory pressure.

Gemini 2 is an RGB-D camera, but InstantHMR is trained as a monocular RGB model. This adapter does
not replace `cam_trans` with a depth-box median: that measures the visible clothing surface rather
than the MHR body origin and produces a systematic translation error. Depth fusion requires an
aligned depth stream plus a rendered-mesh-to-depth fit and is intentionally kept separate from a
faithful InstantHMR reproduction.

#### Jetson/Gemini troubleshooting

- **`no Orbbec camera found` or permission denied:** rerun `shared/setup_env.py`, reconnect the
  camera, check `lsusb -t`, avoid charge-only cables and unpowered hubs.
- **Requested RGB profile falls back:** update Gemini 2 firmware/SDK, or specify a supported
  `--orbbec-width`, `--orbbec-height` and `--orbbec-fps`. The fallback is valid but MJPEG may use
  more CPU.
- **`CUDAExecutionProvider` missing:** the desktop ORT wheel or a CPU build was installed. Recreate
  the environment with `requirements-jetson.txt` and verify JetPack 7/Python 3.12/aarch64.
- **`TensorrtExecutionProvider` missing:** use `--device cuda`; TensorRT acceleration is optional,
  and the installed ORT wheel must match JetPack's CUDA/TensorRT ABI.
- **Out of memory / process killed:** use `--no-viewer`, `--no-mesh`, `--max-persons 1`, detector
  `nano`, `--fit 0` or `1`, close the Ubuntu desktop/browser, and reduce camera resolution. Swap can
  help CPU allocations but does not create GPU bandwidth.
- **TensorRT engine fails after an upgrade:** remove only `models/trt_cache/` and let this device
  rebuild it. Engine files are hardware and software-stack specific.
- **First run cannot download weights:** download the ONNX graphs and `mhr-body.onnx` on a connected
  machine, place them together, and pass the local model with `--model /path/model.onnx`.

## Demo

```bash
python demo.py --image photo.jpg
python demo.py --video clip.mp4 --pose-filter one-euro
python demo.py --camera 0 --model small --fit 1
python demo.py --orbbec --jetson --pose-filter one-euro
```

The [Rerun](https://rerun.io) viewer opens with the frame and its 2D skeletons, the 3D scene, and
the latency of each stage. People are found by the [RF-DETR](https://github.com/roboflow/rf-detr)
detector, then InstantHMR runs on each of them.

| Option | |
|---|---|
| `--model` | `nano`, `tiny`, `small`, `medium` (default), `medium-commercial`, or a path to an `.onnx` file |
| `--fit` | kinematic fit on `2` crops per person (default), `1` crop, or `0` for the network alone |
| `--device` | `cuda` (default), `tensorrt`, `cpu` or `coreml` |
| `--pose-filter` | videos and cameras: `one-euro` or `alpha-beta` filters each person's pose before the mesh is built; both were tuned on 3DPW and EgoHumans, lower the error slightly and the jitter by a third, and lag less than half a frame |
| `--smooth` | videos and cameras: smooths the keypoints and mesh instead; steadier still, but it lags on fast motion |
| `--focal` | the camera's focal length in pixels, if you know it; it places people at the right depth |
| `--principal-point` | camera `cx cy` in pixels; Gemini 2 supplies this automatically |
| `--orbbec` | capture Gemini 2 with Orbbec SDK v2, including calibration and hardware timestamps |
| `--orbbec-serial` | choose one Orbbec device when several are connected |
| `--orbbec-width`, `--orbbec-height`, `--orbbec-fps` | requested colour profile (1280x720 at 30 fps by default) |
| `--jetson` | safe Orin Nano 8 GB model/detector/batch/workspace preset |
| `--trt-workspace-gb` | TensorRT workspace cap; the Jetson preset uses 2 GiB |
| `--detector` | RF-DETR size: `nano`, `small`, `medium` (default), `base`, `large` |
| `--max-persons` | most people per frame (default 4) |
| `--detector-stride` | run the detector every N frames; in between, each person's box follows their 2D keypoints |
| `--no-mesh` | skeleton only; also skips full mesh generation |
| `--save-rrd` | also save the recording, to replay with `rerun file.rrd` |
| `--no-viewer` | headless mode; skips Rerun work unless `--save-rrd` is present |

The latencies in the tables above are the model alone, as measured in the paper. The demo's timings
are what a frame costs end to end in Python. On an RTX 4070 with TensorRT, the InstantHMR stage takes
2.8 ms per person with `--fit 1` (0.25 ms to cut the crop on the CPU, 1.3 ms for the graph, 1.2 ms
for the keypoints and the full 18,439-vertex mesh) and 3.3 ms with `--fit 2`; with CUDA, 4.7 and
5.7 ms. The fit graphs are faster end to end than the network alone (3.3 ms with TensorRT), because
they take the crop as 8-bit pixels and run in fp16. The detector is the slow stage (about 27 ms for
RF-DETR medium); for a faster live demo, use `--detector nano` or `--detector-stride 3`.

## Use it from Python

```python
import cv2
from instanthmr import PosePipeline

pipeline = PosePipeline("medium", fit=2, device="cuda")       # pose_filter="one-euro" for video
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
for person in pipeline(image).persons:
    person.joints_2d        # (70, 2) keypoints, pixels
    person.joints_3d_cam    # (70, 3) keypoints, metres, camera frame (x right, y down, z forward)
    person.vertices_cam     # (18439, 3) mesh, metres, camera frame; faces: pipeline.hmr.faces
    person.mhr_params       # (204,) MHR pose and skeleton parameters
    person.shape_params     # (45,) MHR body shape
```

With your own person boxes, use `InstantHMR` directly:
`InstantHMR(resolve_model("medium", fit=2)).predict(image, boxes)`. The 70 keypoints are listed in
`instanthmr.JOINT_NAMES`.

## TensorRT

TensorRT runs InstantHMR in fp16 on NVIDIA GPUs. Install it next to onnxruntime and pick the
device:

```bash
pip install tensorrt-cu12==10.15.*
python demo.py --camera 0 --device tensorrt
```

That pip command is for CUDA 12 desktop installations. On Jetson, use the TensorRT supplied by
JetPack and follow the Jetson section above; do not install `tensorrt-cu12` over it.

The first run builds an engine for your GPU and caches it in `models/trt_cache/`: about two
minutes for a network graph (it covers batches of up to `--max-persons` people), and four to ten
minutes for a fit graph (one person per call, one engine per graph). Later runs start in under a
second. With CUDA, the fit graphs are replayed as a single CUDA graph, so they need no extra
installation to be fast.

## Training and evaluation

- [`training/`](training/README.md): train the released models, or your own, from the SAM 3D Body
  annotations, fine-tune a released model, and export them to ONNX.
- [`datasets_pipeline/`](datasets_pipeline/README.md): download the annotations and images and
  build the training data.
- [`benchmark/`](benchmark/README.md): reproduce the paper's EMDB-1 and 3DPW numbers.

## Licence

The code is under the [Apache-2.0](LICENSE) licence, except
[`training/mhr_cont.py`](training/mhr_cont.py) and [`training/assets/`](training/assets/), which
are derived from SAM 3D Body and are under the [SAM License](LICENSE-SAM).

The weights, all five models (with and without the fit, and their PyTorch checkpoints) and the shared
`mhr-body.onnx`, are under the
[SAM License](LICENSE-SAM): they are trained on the SAM 3D Body annotations, which Meta releases
under that licence, and they contain tables from the SAM 3D Body checkpoint. The SAM License
allows commercial use, under the conditions it lists (trade controls, no military use, among
others). In addition, the `nano`, `tiny`, `small` and `medium` weights are for non-commercial
use only, because some of their training images (SA-1B, AI Challenger, and COCO photos under
non-commercial Flickr licences) are licensed for research or non-commercial use only.
`medium-commercial` was trained only on images that allow commercial use and has no such
restriction. The body model inside `mhr-body.onnx` is Meta's
[MHR](https://github.com/facebookresearch/MHR) (Apache-2.0).

## Acknowledgements

InstantHMR is trained on the annotations Meta released with
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and uses Meta's
[MHR](https://github.com/facebookresearch/MHR) body model. People are detected with
[RF-DETR](https://github.com/roboflow/rf-detr) and shown with [Rerun](https://rerun.io). This
project was provided with computer and storage resources by GENCI at IDRIS thanks to the grant
2025-AD011017116 on the V100 partition of the supercomputer Jean Zay.

## Citation

<!-- ticket 20: add the arXiv identifier. -->
```bibtex
@misc{adjel2026instanthmr,
  title        = {InstantHMR: Real-Time Human Mesh Recovery on Your Phone},
  author       = {Adjel, Mohamed and Chalabi, Kahina and Bonnet, Vincent},
  year         = {2026},
  howpublished = {HAL preprint hal-05854387},
  url          = {https://hal.science/hal-05854387},
  note         = {LAAS report no. 26486}
}
```
