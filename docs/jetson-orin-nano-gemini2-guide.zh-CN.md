# 在 Jetson Orin Nano 8GB + Orbbec Gemini 2 上部署 InstantHMR

> 从硬件接线、JetPack 6 环境搭建、Gemini 2 取流，到无界面录制 RRD、电脑端回放与技术原理解析的完整教程

本文面向第一次接触 Jetson、Orbbec 或人体网格恢复的读者。只要按顺序执行、每一步都看到预期结果后再继续，就可以在 NVIDIA Jetson Orin Nano 8GB 上使用 Orbbec Gemini 2 采集彩色画面，运行 RF-DETR 与 InstantHMR，并把二维骨架、三维骨架、人体网格和耗时信息录制为 Rerun 的 `.rrd` 文件。

本文的首要目标是“稳定复现”。对于已经出现过载板过流、系统冻结并且必须断电重启的设备，本文不把 Jetson 本地可视化窗口作为默认方案，而是采用：

> **Jetson 只负责相机采集、AI 推理和 RRD 录制，个人电脑负责 RRD 可视化回放。**

---

> **标题图生图提示词｜建议作为文章封面**
>
> 正向提示词：一张专业、克制、具有工程纪实感的横版技术封面，深灰色工作台上放置 NVIDIA Jetson Orin Nano 8GB 开发板与 Orbbec Gemini 2 双目 RGB-D 相机，二者通过短 USB 3 数据线连接；相机前方站着一名自然姿态的人，画面中以半透明青绿色线框叠加完整三维人体骨架与人体网格，右侧显示简洁的数据流光线，从相机流向 Jetson，再流向一台正在回放三维结果的笔记本电脑；真实硬件比例，散热风扇、供电线和 USB 线清晰但不杂乱，边缘计算、离线部署、低功耗视觉 AI 氛围，电影级柔和侧光，蓝绿色科技配色，写实产品摄影与精密信息可视化融合，画面中心留出标题区域，不生成任何品牌宣传文案，16:9，2560×1440。
>
> 负向提示词：错误接口，悬空线缆，多余摄像头，夸张机器人，赛博朋克霓虹污染，人体肢体畸形，错误骨骼连接，虚假性能数字，不可读乱码，水印，二维码，品牌口号，过度发光，杂乱背景。
>
> 后期排版建议：不要让生成模型直接写中文；在预留区域后期加入标题“Jetson Orin Nano + Gemini 2：InstantHMR 边缘部署实战”。

## 阅读路线

- 第 1～2 章介绍项目来源、适配内容、能力边界和应用场景；
- 第 3～5 章完成硬件、供电、JetPack 6 环境和 Gemini 2 配置；
- 第 6～8 章完成安全录制、电脑端回放和最终验收；
- 第 9 章集中排查实机已经遇到过的依赖、模型、cuDNN、TensorRT 和过流问题；
- 第 10～21 章解释相机模型、人体检测、InstantHMR、运动学拟合、时序滤波、ONNX Runtime、图改写、RRD 与深度融合原理；
- 附录提供常用命令、缓存位置和配图视觉规范。

## 1. 项目来源、适配定位与能力边界

### 1.1 项目来源

本仓库改动自开源项目 [Naturalpad-studio/InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)。InstantHMR 从单张 RGB 图像中恢复画面内人物的三维姿态、体型、70 个二维/三维关键点，以及 Momentum Human Rig（MHR）完整人体网格。

当前适配仓库为 [bladewin-smith/InstantHMR](https://github.com/bladewin-smith/InstantHMR)，重点面向以下固定设备和软件组合：

- NVIDIA Jetson Orin Nano 8GB；
- JetPack 6 代软件栈、Ubuntu 22.04、Python 3.10；
- Orbbec Gemini 2 相机与 Orbbec SDK v2；
- CUDA 12.6、cuDNN 9.6、TensorRT 10.7；
- NVIDIA Jetson 版 PyTorch 2.5.0a0；
- ONNX Runtime GPU 1.20.0。

代码仍保留原项目的图像、视频、普通摄像头和桌面 GPU 用法；本文只讲已经针对上述 Jetson + Gemini 2 组合验证过的路径。

### 1.2 本适配版本做了什么

| 适配点 | 实际作用 |
|---|---|
| Gemini 2 原生取流 | 通过 `pyorbbecsdk2` 读取 RGB 帧，而不是把相机当作普通 `/dev/video*` 设备 |
| 相机标定接入 | 自动读取彩色相机的 `fx`、`fy`、`cx`、`cy`，并传给三维人体恢复模型 |
| 硬件时间戳 | 使用相机时间戳驱动视频姿态滤波，降低主机调度抖动的影响 |
| Jetson 8GB 预设 | 使用 small + fit1、RF-DETR nano、最多 2 人、检测间隔 2 帧、2 GiB TensorRT 工作空间上限 |
| 默认无界面 | `--jetson` 默认不在 Jetson 上启动 Rerun 原生窗口，避免额外 GPU、CPU 和显示负载 |
| 低开销 RRD 录制 | 支持限定录制时间、帧数和 Rerun 写入步长 |
| cuDNN 9.6 兼容修复 | 把问题 ONNX 图中的非对称卷积填充等价改写为“对称卷积 + Slice”并缓存 |
| 离线缓存优先 | 模型已经缓存时先直接使用本地文件，不再先向 Hugging Face 发送网络请求 |
| JetPack 6 依赖锁定 | 解决 RF-DETR、Transformers、PyAV、Supervision、Rerun 与 NumPy 的版本冲突 |
| 诊断工具 | `tools/check_jetson.py` 同时检查 CUDA、cuDNN、ORT、TensorRT、依赖版本、相机和真实 HMR 推理 |

### 1.3 能输出什么

对每个检测到的人，程序可输出：

- 人体检测框及检测置信度；
- 70 个二维关键点，单位为图像像素；
- 70 个三维关键点，单位为米；
- 18,439 个顶点组成的 MHR 人体网格；
- 204 维 MHR 姿态与骨架参数；
- 45 维人体形状参数；
- 相机坐标系中的人体平移量；
- RF-DETR 检测阶段和 InstantHMR 阶段的逐帧耗时。

### 1.4 必须提前理解的边界

1. **当前主链路只使用 Gemini 2 的 RGB 彩色流。** Gemini 2 虽然是 RGB-D 相机，但本适配没有把深度图直接塞进 InstantHMR。
2. **这是视觉算法与工程原型，不是安全认证产品。** 不应把单模型结果直接作为停机、医疗诊断、跌倒救援或人身安全决策的唯一依据。
3. **模型输出不是裸眼“真值”。** 遮挡、宽松衣物、强背光、快速运动、极端视角和人物过小都会降低质量。
4. **训练不适合在 8GB Orin Nano 上完成。** 该设备用于推理部署；训练、微调和大规模评估应放在独立工作站或服务器上。
5. **使用前要核对许可证。** 代码主要采用 Apache-2.0；模型权重和部分训练资产还受仓库内 `LICENSE-SAM` 约束。`nano`、`tiny`、`small`、`medium` 权重包含非商业使用限制；准备商用时必须重新审查模型、数据和业务合规性。

## 2. 适合应用在哪些场景

以下场景描述的是可行的研发方向，而不是开箱即用的行业承诺。真正落地时，还需要针对现场视角、光照、人群密度和误报成本进行数据测试，并增加业务规则、日志、隐私保护和人工复核机制。

### 2.1 工业作业姿态分析与风险提示

可在固定工位观察弯腰、抬手、下蹲、搬运动作和异常倒地姿态，生成可回放的三维动作记录。适合用于工效学研究、作业流程复盘和风险提示原型。若要联动设备停机，必须增加冗余传感器、明确的状态机和经过验证的安全控制器。

> **应用场景生图提示词｜工业作业姿态分析**
>
> 正向提示词：现代化但真实的工业装配工位，一名穿普通工装和安全帽的工人在搬取中等大小的箱体，固定在工位上方的 Orbbec Gemini 2 相机朝向作业区，侧边安装 Jetson Orin Nano 边缘计算盒；人物身上叠加准确的黄色二维骨架和半透明青色三维人体网格，背景屏幕只显示简洁的姿态曲线与风险等级色块，强调本地离线处理和隐私保护，真实工业摄影，安全、克制、无事故画面，16:9。
>
> 负向提示词：血腥，真实事故，危险机械正在夹人，夸张红色警报，错误 PPE，人体畸形，骨骼错位，多余肢体，不可读文字，虚假认证标志。

### 2.2 机器人与人的空间交互研究

三维关键点和人体朝向可为移动机器人、机械臂或服务机器人提供人的大致空间位置与动作上下文，用于交互策略研究、协作区域可视化和离线数据分析。实际协作机器人安全仍应依赖安全激光雷达、安全 PLC 或经过认证的防护系统。

> **应用场景生图提示词｜人机协作研究**
>
> 正向提示词：明亮整洁的机器人实验室，一名研究人员站在小型协作机械臂旁做自然指向动作，Gemini 2 固定在机器人上方，Jetson Orin Nano 安装在机器人底座内部；画面用精确简洁的三维坐标轴、人体骨架、机器人感知锥和安全距离圆环表达空间关系，白色与蓝灰色科研视觉风格，真实设备比例，技术纪录片摄影，16:9。
>
> 负向提示词：科幻战斗机器人，武器，危险碰撞，机械臂穿过人体，错误骨架，浮空硬件，密集乱码，霓虹赛博朋克。

### 2.3 运动训练与康复动作辅助评估

系统可以记录深蹲、弓步、抬臂等动作的三维骨架轨迹，为教练或研究人员提供可回放材料。它适合做动作对比和趋势分析原型，但不能替代医学影像、临床量表或专业诊断。

> **应用场景生图提示词｜运动与康复动作评估**
>
> 正向提示词：安静明亮的运动训练室，一名成年人在瑜伽垫上进行标准深蹲动作，正前方三脚架上安装小型 Gemini 2 相机，旁边是封装后的 Jetson Orin Nano；画面叠加左右对称的三维骨架、膝关节与髋关节角度弧线、前后两次动作的淡色轨迹对比，医疗级克制配色但明确表现为非诊断性的训练分析，写实摄影，16:9。
>
> 负向提示词：医院急救，诊断结论，手术场景，夸张肌肉，人体解剖暴露，关节反向弯曲，不可读数值，医疗认证标志。

### 2.4 智能空间、数字孪生与人流行为研究

在获得合法授权并做好匿名化的前提下，可以将人体位置和骨架映射到三维空间，用于展厅交互、空间布局评估、数字孪生原型或小规模人流行为研究。部署时应尽量在端侧处理，并制定原始图像的保存期限和访问权限。

> **应用场景生图提示词｜智能空间与数字孪生**
>
> 正向提示词：小型科技展厅的真实室内场景，三名参观者分散站立，天花板边缘安装 Gemini 2，相机数据由 Jetson Orin Nano 本地处理；右半画面以半透明数字孪生方式呈现简化空间、匿名人体骨架和移动轨迹，不显示人脸身份，不使用云端图标，强调端侧隐私计算，建筑可视化与真实摄影自然融合，16:9。
>
> 负向提示词：人脸识别框，姓名标签，监控压迫感，密集人群，错误透视，人物穿墙，乱码，云端上传符号。

### 2.5 离线人体动作数据采集与算法验证

RRD 文件可以把图像、二维骨架、三维网格和耗时一起保存，适合在无稳定网络的现场采集数据，然后带回电脑复盘。该方式也适合比较不同模型、阈值和滤波设置，但 RRD 内可能含有可识别的原始 RGB 图像，仍需按个人信息处理。

> **应用场景生图提示词｜离线现场数据采集**
>
> 正向提示词：临时搭建的室内测试现场，Jetson Orin Nano、Gemini 2 和 NVMe 边缘采集盒安装在便携支架上，一名测试人员在标定区域行走，现场没有互联网设备；画面中用一条清晰流程线表示“相机采集—Jetson 推理—NVMe 保存 RRD—实验室电脑回放”，工程师手持笔记本检查三维人体网格，真实工程测试氛围，信息图与写实摄影融合，16:9。
>
> 负向提示词：云端上传，5G 基站，夸张数据洪流，丢失电源线，不可读流程文字，错误骨架，虚假速度数字。

## 3. 部署前的硬件准备

### 3.1 硬件清单

| 类别 | 建议准备 | 原因 |
|---|---|---|
| 计算平台 | Jetson Orin Nano 8GB + 与之匹配的载板 | 本文的推理和功耗策略按 8GB 共享内存设计 |
| 供电 | **严格符合载板厂商规格**的稳定电源和线缆 | 相机、GPU、CPU、NVMe 与显示负载叠加时会产生瞬态电流 |
| 散热 | 主动散热器、风扇，保证风道通畅 | 模型首次加载、推理和 TensorRT 构建会产生持续负载 |
| 存储 | 推荐 128GB 或更大的 NVMe SSD | 模型缓存、兼容 ONNX 图、TensorRT 引擎和 RRD 文件都需要空间 |
| 相机 | Orbbec Gemini 2，建议使用已验证固件 | 本设备已验证固件为 1.4.98 |
| 数据线 | 短而可靠的 USB 3 数据线 | 充电线或劣质线可能使链路退化为 480M 或频繁掉线 |
| 可选供电 USB Hub | 有独立电源的高质量 USB 3 Hub | 对供电余量较小的自定义载板，可把相机负载移出载板 USB 电源轨 |
| 网络 | 首次下载模型时可用的网络；正式运行可离线 | 已缓存的模型不会强制访问 Hugging Face |
| 回放电脑 | Windows、Linux 或 macOS 电脑 | 用于安全地打开 `.rrd`，避免 Jetson 本地渲染负载 |

Gemini 2 数据表给出的峰值功耗高于平均功耗。载板已经出现过流事件时，不要把“平均运行正常”当作供电余量充足的证明；启动瞬间、推理峰值、相机状态切换和本地三维渲染可能在同一时间叠加。

### 3.2 推荐接线

1. Jetson 断电。
2. 安装 NVMe、主动散热器和风扇。
3. Gemini 2 使用 USB 3 数据线接入载板的 USB 3 主机口；若载板 USB 供电余量不足，则接入独立供电的 USB 3 Hub。
4. 接好网线或 Wi-Fi 天线，仅供安装与模型下载使用。
5. 接入载板厂商指定的电源，最后上电。
6. 首次安装阶段可以接显示器；正式推理建议通过 SSH 操作并关闭不需要的桌面程序。

> **硬件接线图生图提示词｜建议放在本节末尾**
>
> 正向提示词：白色背景的精密工程接线示意图，中心是 Jetson Orin Nano 8GB 与载板，左侧是符合载板规格的直流电源，右侧是 Orbbec Gemini 2，相机通过带独立电源的 USB 3 Hub 可选连接到 Jetson，底部是 NVMe SSD 和主动散热风扇，另一侧通过局域网连接一台用于回放 RRD 的个人电脑；使用清晰的编号引线、统一线宽和蓝灰橙配色，明确区分“电源线”“USB 3 数据线”“局域网”，接口位置符合真实硬件逻辑，扁平矢量信息图，16:9，预留中文标注空白。
>
> 负向提示词：错误 USB 方向，电源接到数据口，一根线同时连接多个端口，无散热器，杂乱透视，虚假端口，乱码，品牌广告，3D 卡通风格。

### 3.3 上电后的功耗安全检查

先查询当前功耗模式：

```bash
sudo nvpmodel -q
```

首次验证阶段遵循三条原则：

- 使用载板明确支持的受限功耗模式；
- 不启用 `MAXN SUPER`；
- 不运行 `sudo jetson_clocks`。

记录过流计数器基线：

```bash
grep "" /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null
```

完成 30 秒测试后再执行一次。如果计数增加，或者系统冻结、黑屏、重启，就应先解决供电路径和负载峰值问题。**不要通过提高 INA3221 电流上限或修改 BPMP 节流阈值来掩盖过流事件。** NVIDIA 对功耗模式与过流保护的说明可参考 [Jetson Orin 系列功耗与性能文档](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)，相机峰值可参考 [Gemini 2 数据表](https://www.orbbec.com/staging/wp-content/uploads/2023/04/ORBBEC_Datasheet_Gemini-2.pdf)。

## 4. 已验证的软件基线

本文对应的实机基线如下。版本不是越新越好；Jetson 的 CUDA、cuDNN、TensorRT、PyTorch 与 ONNX Runtime 存在 ABI 对应关系，随意升级其中一个组件可能破坏整套环境。

| 项目 | 已验证版本或状态 |
|---|---|
| 操作系统 | Ubuntu 22.04，Linux 5.15.148-tegra，aarch64 |
| Python | 3.10.12 |
| CUDA | 12.6 |
| cuDNN | 9.6，`torch.backends.cudnn.version()` 为 90600 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 |
| torchvision | 0.20.0a0，Jetson 对应构建 |
| ONNX Runtime GPU | 1.20.0 |
| ORT Provider | TensorRT、CUDA、CPU 均可见 |
| TensorRT | 10.7.0 |
| Orbbec Python 包 | `pyorbbecsdk2` 2.1.1 |
| Orbbec SDK Runtime | 2.8.6 |
| Gemini 2 | 固件 1.4.98，RGB 1280×720@30 |
| RF-DETR | 1.5.1 |
| Transformers | 4.57.3 |
| Supervision | 0.29.1 |
| PyAV | 12.3.0 |
| NumPy | 1.26.4 |
| Rerun SDK（Jetson） | 0.22.1 |

> 本文只适用于上述 JetPack 6 / Python 3.10 / CUDA 12 代环境。Python 3.12 + CUDA 13 的 JetPack 7 设备应使用仓库中的 `requirements-jetson-jp7.txt`，不能把 JetPack 7 的 ORT 包装到本设备上。

## 5. 手把手环境搭建

以下命令均在 Jetson 的 Bash 终端执行。示例仓库目录为 `~/ws/InstantHMR`；如果你的路径不同，请在对应目录操作。

### 第 1 步：确认 JetPack 基础环境

```bash
uname -m
cat /etc/nv_tegra_release
python3 --version
```

本文预期：

- `uname -m` 输出 `aarch64`；
- Python 为 3.10.x；
- CUDA 属于 12.x。

再检查 NVIDIA PyTorch、CUDA 和 cuDNN：

```bash
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA usable:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

如果这里的 `CUDA usable` 不是 `True`，先修复 JetPack 与 NVIDIA PyTorch，暂时不要安装本项目依赖。不要用桌面 x86 教程中的 PyTorch wheel 覆盖 Jetson 版本。

### 第 2 步：安装少量系统工具

已经存在并且工作正常的 CUDA、cuDNN、TensorRT、PyTorch 和 ONNX Runtime 不要重装。只补充缺少的通用工具：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libopenblas-dev libgl1 git
```

### 第 3 步：获取适配仓库

首次下载：

```bash
mkdir -p ~/ws
cd ~/ws
git clone https://github.com/bladewin-smith/InstantHMR.git
cd InstantHMR
```

如果仓库已经存在：

```bash
cd ~/ws/InstantHMR
git status
git pull --ff-only origin main
```

若 `git status` 显示你有自己的未提交修改，不要直接覆盖。先提交、另存或使用新目录重新克隆。

### 第 4 步：创建隔离的 JetPack 6 虚拟环境

Jetson 的 CUDA 版 PyTorch、torchvision 和 ORT 通常在系统 Python 中，因此这里必须使用 `--system-site-packages` 继承它们，同时把项目特有版本限制在虚拟环境内。

如果终端前面还有 Conda 的 `(base)`，先退出 Conda：

```bash
conda deactivate 2>/dev/null || true
```

创建全新的环境：

```bash
cd ~/ws/InstantHMR
python3 -m venv --system-site-packages .venv-jp6-v2
source .venv-jp6-v2/bin/activate
python -m pip install --upgrade pip
```

确认解释器路径：

```bash
which python
python --version
```

`which python` 必须指向：

```text
.../InstantHMR/.venv-jp6-v2/bin/python
```

如果后续报错堆栈指向 `/home/jetson/.local/lib/python3.10/site-packages`，通常说明当前虚拟环境没有正确激活，或者用户级包污染了导入路径。不要继续在共享用户环境中叠加升级。

### 第 5 步：安装仓库锁定的 Jetson 依赖

```bash
python -m pip install -r requirements-jetson.txt
```

安装过程中不要额外执行下面这些命令：

```text
pip install --upgrade torch torchvision
pip install --upgrade rfdetr transformers supervision av
pip install --upgrade numpy rerun-sdk onnxruntime-gpu tensorrt
```

这些包不是独立可随意升级的。仓库锁定的关键组合是：

```text
NumPy 1.26.4
RF-DETR 1.5.1 + Transformers 4.57.3
Supervision 0.29.1 + PyAV 12.3.0
Rerun 0.22.1
ONNX 1.17.0
httpcore 1.0.9 + h11 0.16.0
ORT GPU 1.20.0（通常来自系统可见的 Jetson 安装）
```

建议抑制 Albumentations 的在线版本提示：

```bash
export NO_ALBUMENTATIONS_UPDATE=1
```

如需每次登录自动生效，可把这一行加入你自己的 shell 配置；这不是程序运行的必要条件。

### 第 6 步：运行统一环境检查

```bash
python tools/check_jetson.py
```

在已验证设备上，关键输出应类似：

```text
python:   3.10.12
torch:    2.5.0a0+...nv24.08; CUDA=12.6; cuDNN=90600; available=True
GPU:      Orin
stack:    JetPack 6 / CUDA 12; use requirements-jetson.txt
ORT:      1.20.0; providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
TensorRT: 10.7.0
Orbbec:   wrapper=2.1.1; SDK runtime=2.8.6
RF-DETR:  1.5.1; detector API OK
Rerun:    0.22.1
result:   OK
```

这里的 `result: OK` 比“全局 `pip check` 没有任何告警”更重要。因为 `--system-site-packages` 会看到系统里与本项目无关的 ROS、TensorFlow 或 airo 包，全局依赖检查可能报告历史冲突。

### 第 7 步：配置 Gemini 2 的 Linux 权限

连接 Gemini 2 后先确认 USB 枚举：

```bash
lsusb
lsusb -t
```

在 `lsusb -t` 中找到相机所在分支。应显示 `5000M` 或更高，而不是 `480M`。若只有 `480M`：

- 更换明确支持 USB 3 的数据线；
- 更换 USB 3 主机口；
- 移除无源 Hub；
- 避免使用只支持充电的 Type-C 线。

安装 `pyorbbecsdk2` 自带的 udev 规则和运行环境配置：

```bash
python "$(python -c 'import os, pyorbbecsdk; print(os.path.join(os.path.dirname(pyorbbecsdk.__file__), "shared", "setup_env.py"))')"
```

命令完成后，拔下相机再重新插入。若脚本提示需要重新登录或重启，就按提示执行。

### 第 8 步：只测试相机，不加载完整模型

```bash
python tools/check_jetson.py --camera
```

已经验证过的实机输出为：

```text
camera:   Orbbec Gemini 2 serial=AY3794302V2 firmware=1.4.98, RGB 1280x720@30
frame:    (720, 1280, 3), t=0.000000s
intrinsic fx=691.28 fy=691.35 cx=637.12 cy=363.80
result:   OK
```

每台相机的内参会略有差异，不要求数值完全相同；只要尺寸正确、数值合理且结果为 `OK` 即可。多相机环境使用：

```bash
python tools/check_jetson.py --camera --serial 你的相机序列号
```

### 第 9 步：完成一次真实 InstantHMR CUDA 冒烟测试

```bash
python tools/check_jetson.py --hmr-smoke
```

该测试会：

1. 解析 `small-fit1` 模型；
2. 本地缓存存在时直接使用缓存；
3. 首次使用且缓存不存在时从 Hugging Face 下载；
4. 应用 Jetson 非对称卷积填充修复；
5. 在 CUDA 上完成一次真实预测。

成功标志是：

```text
HMR smoke: small-fit1 prediction OK on CUDAExecutionProvider
result:   OK
```

首次运行可能生成：

```text
models/onnx_compat/*-pad-v2.onnx
```

它是从原始模型生成的兼容缓存；原始下载模型不会被覆盖。

#### 网络受限时怎样准备模型

如果 Jetson 无法访问 Hugging Face，可以在一台能联网的 Linux/Windows 电脑上下载以下两个文件，并把它们放在 Jetson 的同一目录中：

```text
instanthmr-small-fit1.onnx
mhr-body.onnx
```

模型仓库地址为 [momolesang/InstantHMR](https://huggingface.co/momolesang/InstantHMR)。复制后可用本地模型路径启动：

```bash
python demo.py \
  --orbbec \
  --jetson \
  --device cuda \
  --model ~/models/InstantHMR/instanthmr-small-fit1.onnx \
  --fit 1 \
  --no-viewer \
  --save-rrd jetson-run.rrd \
  --duration 30
```

RF-DETR nano 也会在首次使用时准备自己的权重。如果部署网络完全隔离，应在一套版本相同、能够正常启动 RF-DETR 1.5.1 的环境中先完成下载，再复制对应的模型缓存。不要从版本不同的 RF-DETR 安装中随意拿同名文件覆盖。

当 Hugging Face 文件已经完整缓存时，可以临时强制离线：

```bash
export HF_HUB_OFFLINE=1
```

缓存不完整时设置该变量只会让程序更快报“找不到文件”，不会凭空生成模型。

## 6. 首次安全运行：录制 30 秒 RRD

### 6.1 使用推荐命令

在当前设备上，以下命令是首选和已经验证的运行方式：

```bash
cd ~/ws/InstantHMR
source .venv-jp6-v2/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1

python demo.py \
  --orbbec \
  --jetson \
  --device cuda \
  --pose-filter one-euro \
  --no-viewer \
  --save-rrd jetson-run.rrd \
  --duration 30 \
  --rerun-stride 2
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--orbbec` | 使用 Orbbec SDK v2 读取 Gemini 2 RGB 流、内参和硬件时间戳 |
| `--jetson` | 启用 Orin Nano 8GB 安全预设，并默认关闭本地查看器 |
| `--device cuda` | 使用稳定、已经通过测试的 ORT CUDA Provider |
| `--pose-filter one-euro` | 对连续帧的 MHR 姿态做因果滤波，降低抖动 |
| `--no-viewer` | 明确不在 Jetson 上创建 Rerun 原生窗口 |
| `--save-rrd` | 把图像、骨架、网格和耗时写入指定 RRD 文件 |
| `--duration 30` | 按相机时间录制 30 秒后自动结束 |
| `--rerun-stride 2` | 每两个处理帧记录一个 Rerun 帧，降低 CPU、磁盘与文件体积开销 |

`--jetson` 实际采用：

```text
InstantHMR small + fit1
RF-DETR nano
最多 2 人
每 2 帧执行一次检测
TensorRT 工作空间上限 2 GiB
cuDNN 卷积搜索 HEURISTIC
启用 ONNX 非对称填充兼容改写
禁用 RF-DETR TorchScript 编译
默认无本地 Rerun 窗口
```

### 6.2 正常运行时应该看到什么

启动阶段应依次出现：

1. `loading InstantHMR (small, kinematic fit on 1 crop) on cuda and RF-DETR nano...`；
2. `Jetson compatibility: asymmetric-padding rewrite enabled...`；
3. `InstantHMR runs on CUDAExecutionProvider`；
4. Gemini 2 的序列号、固件、分辨率和内参；
5. 每隔约 30 个源帧输出人物数量、detector 耗时和 InstantHMR 耗时；
6. 到达 30 秒后输出中位耗时并自动退出。

首次初始化 RF-DETR 和 InstantHMR 会比稳态帧慢。不要把第一帧耗时当作长期帧率。

### 6.3 检查录制结果

```bash
ls -lh jetson-run.rrd
test -s jetson-run.rrd && echo "RRD recording OK"
```

看到非零文件大小和 `RRD recording OK` 即完成第一阶段验收。

然后再次检查过流计数：

```bash
grep "" /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null
```

若计数增加、系统明显卡顿或相机掉线，不要立刻扩大负载。先降低为单人、骨架模式：

```bash
python demo.py \
  --orbbec --jetson --device cuda \
  --max-persons 1 --no-mesh \
  --no-viewer --save-rrd jetson-lite.rrd \
  --duration 30 --rerun-stride 3
```

### 6.4 录制到手动停止

确认 30 秒运行稳定后，可以去掉 `--duration`：

```bash
python demo.py \
  --orbbec --jetson --device cuda \
  --pose-filter one-euro \
  --no-viewer --save-rrd jetson-run-long.rrd \
  --rerun-stride 2
```

使用 `Ctrl+C` 正常结束。不要使用 `Ctrl+Z`；`Ctrl+Z` 是挂起进程，进程仍然占用资源，RRD 也可能没有按预期结束。

需要精确处理帧数时可使用：

```bash
--max-frames 300
```

### 6.5 为什么不能只写 `--no-viewer`

如果命令只有 `--no-viewer`，没有 `--save-rrd`，程序会跳过 Rerun 初始化和逐帧序列化，只在终端打印耗时。这适合纯性能或稳定性测试，但不会生成可回放文件。

要录制结果，必须同时具备：

```text
--no-viewer --save-rrd 文件名.rrd
```

## 7. 把 RRD 复制到电脑并查看效果

### 7.1 查询 Jetson IP

```bash
hostname -I
```

假设 Jetson IP 为 `192.168.1.50`，用户名为 `jetson`。

### 7.2 在回放电脑上复制文件

Linux、macOS 或已启用 OpenSSH 的 Windows PowerShell：

```bash
scp jetson@192.168.1.50:~/ws/InstantHMR/jetson-run.rrd .
```

也可以用局域网共享、SFTP 软件或移动存储介质复制。RRD 可能包含人的原始 RGB 图像，不应上传到不受控的公共网盘。

### 7.3 在回放电脑安装 Rerun

建议为回放单独创建环境。电脑端不受 Jetson 的 NumPy 1.26 限制，可安装与 RRD 兼容的同版或更新版 Rerun：

```bash
python -m venv .venv-rerun
```

Linux/macOS 激活：

```bash
source .venv-rerun/bin/activate
```

Windows PowerShell 激活：

```powershell
.\.venv-rerun\Scripts\Activate.ps1
```

安装并打开：

```bash
python -m pip install --upgrade pip rerun-sdk
rerun jetson-run.rrd
```

### 7.4 回放界面中有什么

默认布局包含：

- **Camera**：RGB 帧，以及已经绘制在画面上的人体框和二维骨架；
- **3D**：相机坐标系中的三维关键点、骨架和半透明人体网格；
- **Latency (ms)**：RF-DETR 与 InstantHMR 的逐帧耗时；
- **时间轴**：暂停、拖动、逐帧检查和循环播放。

若只看到骨架没有网格，检查运行命令中是否使用了 `--no-mesh`。若某一帧没有人物，可能是检测置信度不足、人物过小、遮挡或模型没有检测到人，并不一定是 RRD 损坏。

> **运行结果示意图生图提示词｜建议放在回放说明旁**
>
> 正向提示词：一张干净的桌面软件界面概念图，左侧面板显示 Gemini 2 拍摄的人体 RGB 画面及黄色二维骨架，中间是深灰背景上的青色半透明三维人体网格与绿色关节点，底部是两条标为检测与人体恢复耗时的折线图，最下方有可拖动时间轴；体现离线 RRD 回放、逐帧检查和三维轨道视角，信息层级清晰，专业工程软件 UI，16:9，所有文字区域留白供后期标注。
>
> 负向提示词：多人重叠，错误骨架，人体网格破裂，夸张 HUD，密集小字，乱码，虚假性能数字，游戏界面，霓虹过度。

## 8. 验收清单

首次部署只有同时满足以下条件，才算完整跑通：

- [ ] `uname -m` 为 `aarch64`；
- [ ] 当前 Python 位于 `.venv-jp6-v2`；
- [ ] `python tools/check_jetson.py` 最终为 `result: OK`；
- [ ] ORT 至少包含 `CUDAExecutionProvider`；
- [ ] `python tools/check_jetson.py --camera` 能取得一帧并打印相机内参；
- [ ] `python tools/check_jetson.py --hmr-smoke` 完成真实 CUDA 预测；
- [ ] 推荐命令能够自动完成 30 秒录制；
- [ ] `jetson-run.rrd` 非空；
- [ ] 电脑端 Rerun 能看到 RGB、二维骨架、三维骨架/网格和耗时；
- [ ] 运行前后过流计数没有增加，Jetson 没有冻结或重启；
- [ ] 长时间运行前已经确认散热、磁盘剩余空间和电源稳定。

## 9. 避坑指南：本项目实机遇到过的问题

本节是教程部分的最后一节。遇到问题时先按“现象”定位，不要看到一个依赖告警就升级整套环境。

### 9.1 本地打开 Rerun 后载板过流、死机，必须断电重启

**现象**：相机与推理能运行，但在 Jetson 本地同时打开 Rerun 后载板过流，系统冻结或黑屏，无法正常 SSH，只能断电重启。

**原因**：推理、相机 USB 供电、桌面合成、本地三维渲染、NVMe 写入和 CPU 处理在同一时刻叠加，超过载板供电路径的瞬态能力。这不等同于平均功耗过高，也不一定是软件崩溃。

**正确处理**：

```bash
python demo.py \
  --orbbec \
  --jetson \
  --device cuda \
  --no-viewer \
  --save-rrd jetson-run.rrd \
  --duration 30 \
  --rerun-stride 2
```

把 RRD 复制到电脑回放。不要在当前载板上使用 `--local-viewer`，不要启用 `MAXN SUPER` 或 `jetson_clocks`。检查电源、线缆和载板规格；必要时使用独立供电 USB 3 Hub。不要修改电流保护阈值来“解决”过流。

### 9.2 `pyorbbecsdk2` 与 PyAV 冲突

**现象**：

```text
pyorbbecsdk2 2.1.1 requires av==12.3.0
but you have av 14.2.0
```

**原因**：`pyorbbecsdk2` 2.1.1 精确要求 PyAV 12.3.0，而较新的 Supervision 0.30+ 要求更高版本 PyAV。两者不能在此环境中同时满足。

**正确组合**：

```text
pyorbbecsdk2==2.1.1
av==12.3.0
supervision==0.29.1
rfdetr==1.5.1
httpcore==1.0.9
h11==0.16.0
```

重新执行：

```bash
python -m pip install -r requirements-jetson.txt
```

不要单独升级 `av`。仓库同时锁定 `httpcore` 与 `h11` 的兼容组合；如果安装日志只显示
`Successfully installed httpcore-1.0.9`，这本身不是故障，应以完整导入检查和
`tools/check_jetson.py` 的结果为准。

### 9.3 RF-DETR 导入时提示 PyTorch 版本过低或找不到 `BackboneConfigMixin`

**现象**：

```text
[transformers] Disabling PyTorch because PyTorch >= 2.6 is required
but found 2.5.0a0+...nv24.8
ImportError: cannot import name 'BackboneConfigMixin' from 'transformers'
```

**原因**：较新的 RF-DETR/Transformers 5 进入了环境，而 JetPack 6 的 NVIDIA PyTorch 是 2.5.0a0。版本链不兼容。

**正确组合**：

```text
rfdetr==1.5.1
transformers==4.57.3
tokenizers==0.22.1
huggingface-hub==0.36.0
```

若堆栈来自 `~/.local/lib/python3.10/site-packages`，直接创建新的 `.venv-jp6-v2`，不要继续修补被污染的用户环境。

### 9.4 `pip check` 报 airo、TensorFlow、PyNaCl 冲突

**实机曾出现**：

- airo 系列要求 NumPy 2，但本项目使用 NumPy 1.26.4；
- TensorFlow 2.16 NVIDIA 构建与旧 `tf-keras`、`tensorflow-decision-forests`、`tensorflowjs` 伴生包冲突；
- PyNaCl 报当前平台不支持。

这些不是 InstantHMR 推理链路的依赖。因为虚拟环境通过 `--system-site-packages` 继承 JetPack/ROS 包，`pip check` 会同时检查许多不相关的系统包。

**正确处理**：保持 NumPy 1.26.4，不要为了清空全局告警升级到 NumPy 2，也不要为了本项目降级 NVIDIA TensorFlow。以 `python tools/check_jetson.py` 的结果和实际冒烟测试为验收标准。

### 9.5 Matplotlib 提示无法导入 `Axes3D`

```text
UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib...
```

这是 Ubuntu 系统 Matplotlib 与 pip Matplotlib 混装产生的警告，不影响 RF-DETR、InstantHMR、无界面录制或 RRD 内容。不要仅为消除这条警告重建 CUDA 环境。

### 9.6 Albumentations 提示有新版本

这只是在线更新通知，不代表当前版本不可用。RF-DETR 1.5.1 的依赖组合已经锁定，不要看到提示就升级。可用以下环境变量关闭通知：

```bash
export NO_ALBUMENTATIONS_UPDATE=1
```

### 9.7 cuDNN 报 `HEURISTIC_QUERY_FAILED` 或 `No valid engine configs for ConvFwd`

**典型日志**：

```text
CUDNN_FE failure 8: HEURISTIC_QUERY_FAILED
No valid engine configs for ConvFwd
/net/stem/stem2b/conv/Conv
```

**原因**：cuDNN 9.6 前端在该模型首个 FP16 非对称填充卷积上无法找到可用执行引擎。它不是相机故障，也不是模型文件一定损坏。

**仓库内修复**：`--jetson` 自动启用 v2 图改写，把非对称填充卷积转换为数学等价的“对称填充卷积 + Slice”，缓存到 `models/onnx_compat/`。早期的“显式 Pad + Conv”写法可能被 ORT 1.20 再次融合回问题卷积，当前 v2 方案避免依赖该优化行为。

**正确处理**：更新仓库、安装 `requirements-jetson.txt` 中的 ONNX 1.17.0，并保持 `--jetson`。不要单独替换 JetPack 的 cuDNN。

仅用于诊断时可尝试：

```bash
python tools/check_jetson.py --hmr-smoke --cuda-conv-algo DEFAULT
```

### 9.8 已缓存模型仍然访问 Hugging Face，并反复超时

**典型日志**：

```text
HTTPSConnectionPool(host='huggingface.co'... connect timeout=10)
Retrying in 1s...
```

旧逻辑即使已有缓存，也可能先发送 `HEAD` 请求。当前代码会先使用 `local_files_only=True` 查询本地快照，只有真正缺失时才联网。

**处理顺序**：

1. `git pull --ff-only origin main` 更新代码；
2. 确认模型缓存完整；
3. 缓存完整时设置 `HF_HUB_OFFLINE=1`；
4. 完全离线部署时，把 `instanthmr-small-fit1.onnx` 与 `mhr-body.onnx` 放在同一目录，并使用 `--model` 指定本地路径。

### 9.9 TensorRT 首次构建长时间没有输出，被误认为卡死

首次 TensorRT 运行需要在本机为模型构建和缓存引擎，在 Orin Nano 上可能持续数分钟，个别情况下可达 10～20 分钟。

- 不要按 `Ctrl+Z`；它会把构建进程挂起为 `Stopped`；
- 先用 CUDA 完成整套验收，再尝试 TensorRT；
- TensorRT 是可选加速路径，CUDA 是当前稳定路径；
- 不要把 RTX 工作站生成的 TensorRT 引擎复制到 Jetson；
- JetPack、TensorRT、ORT、模型、fit 或最大人数发生变化后，需要在本机重建缓存。

查看挂起任务：

```bash
jobs -l
```

只处理已经明确识别出的旧任务，不要盲目结束系统进程。TensorRT 测试命令仍应保持无界面：

```bash
python demo.py \
  --orbbec --jetson --device tensorrt \
  --no-viewer --save-rrd jetson-trt.rrd \
  --duration 30 --rerun-stride 2
```

若构建失败，继续使用 `--device cuda`，不要为了追求 TensorRT 而破坏已经可用的 JetPack 包。

### 9.10 Rerun 新版要求 NumPy 2

Jetson 环境应使用：

```text
rerun-sdk==0.22.1
numpy==1.26.4
```

Rerun 0.23+ 的 NumPy 2 要求会与本 JetPack 6 / ROS / OpenCV 环境冲突。电脑端回放可以建立独立环境使用更新版，但 Jetson 推理环境不要跟着升级。

### 9.11 相机找不到、无权限或退化为 USB 2

排查顺序：

1. `lsusb` 是否能看到 Orbbec；
2. `lsusb -t` 是否为 `5000M`；
3. 是否执行了 `shared/setup_env.py`；
4. 相机是否重新插拔；
5. 线缆是否是真正的 USB 3 数据线；
6. 无源 Hub 是否导致掉线或供电不足；
7. 多相机时是否指定了正确序列号。

若请求的 `RGB 1280x720@30` 不可用，代码会退回 SDK 默认彩色配置并打印 `(SDK default profile)`。回退可以工作，但若落到 MJPEG，Jetson 会增加解码开销。

### 9.12 内存不足、进程被系统终止

按以下顺序减负：

```text
保持 --no-viewer
增加 --rerun-stride 2 或 3
使用 --max-persons 1
使用 --no-mesh
使用 --model nano --fit 0
增加 --detector-stride 3 或 4
关闭浏览器、桌面动画和无关进程
降低相机分辨率或帧率
```

交换空间可能帮助 CPU 内存分配，但不能增加 GPU 带宽，也不能解决载板过流。

---

# 第二部分：技术原理解读

## 10. 整体数据链路

本项目不是“相机拍一张图后直接得到网格”的单一网络，而是一条由采集、检测、人体恢复、几何建模、时序处理和记录构成的链路：

```text
Gemini 2 RGB 帧 + 彩色相机内参 + 硬件时间戳
                │
                ▼
       RF-DETR Nano 人体检测
                │  人体框、置信度
                ▼
   裁剪 288×288 人体图 + CLIFF 相机条件
                │
                ▼
       InstantHMR small-fit1 ONNX
                │  MHR 参数、形状、相机平移、2D 关键点
                ▼
        图内一次运动学拟合修正
                │
                ▼
     MHR Body ONNX：70 关节 + 18,439 顶点
                │
                ▼
    One Euro/Alpha-Beta 时序滤波（可选）
                │
                ▼
 Rerun 序列化：RGB、2D/3D 骨架、网格、耗时 → RRD
```

> **技术原理配图提示词｜系统总架构图**
>
> 正向提示词：白色背景、论文级矢量系统架构图，从左到右七个模块：Gemini 2 彩色帧与内参、RF-DETR 人体检测、人体裁剪与 CLIFF 条件、InstantHMR ONNX、单步运动学拟合、MHR 人体模型、RRD 离线记录与电脑端回放；每个模块使用简洁真实的小图标，模块间箭头标出数据类型：RGB、bbox、288×288 crop、camera condition、204D pose、45D shape、70 joints、18439 vertices；Jetson 边界用浅绿色虚线框，电脑端 Rerun 用浅蓝色框，严谨、平面、无装饰性电路，16:9，预留中文标注空间。
>
> 负向提示词：网络结构胡乱连线，错误数据维度，云端依赖，大量小字，三维卡通，霓虹光效，乱码，虚假 logo。

## 11. Gemini 2 取流、时间戳与相机内参

### 11.1 为什么不用普通 OpenCV 相机接口

普通 `cv2.VideoCapture(0)` 可以取得图像，却不一定能可靠得到与当前彩色流完全对应的内参、固件信息、硬件时间戳和稳定格式。`--orbbec` 使用 Orbbec SDK v2：

- 优先请求 RGB 1280×720@30；
- 若精确配置不可用，退回 SDK 默认彩色配置；
- 支持 RGB、BGR、MJPEG、YUYV、UYVY、I420、NV12、NV21 转换；
- 读取当前彩色 profile 的真实内参；
- 优先使用相机微秒时间戳，并保证输出时间单调递增。

### 11.2 内参为什么影响三维位置

针孔相机模型可写为：

\[
u = f_x\frac{X}{Z}+c_x,\qquad
v = f_y\frac{Y}{Z}+c_y
\]

其中：

- \((X,Y,Z)\) 是相机坐标中的三维点；
- \((u,v)\) 是图像像素坐标；
- \(f_x,f_y\) 是像素单位焦距；
- \((c_x,c_y)\) 是主点。

模型发布图只接受一个焦距，因此适配层使用：

\[
f=\frac{f_x+f_y}{2}
\]

主点仍使用真实的 \((c_x,c_y)\)。焦距错误会使人物的估计深度与尺度产生系统偏差；主点错误则会使画面边缘人物的相机条件不准确。

> **技术原理配图提示词｜针孔相机与内参**
>
> 正向提示词：严谨的针孔相机教学矢量图，左侧三维坐标系中有人体关键点 P(X,Y,Z)，中间针孔相机光心，右侧图像平面显示像素 p(u,v)，用两条投影线连接；清楚表现光轴、焦距 fx/fy、主点 cx/cy、图像宽高，坐标方向为相机 x 向右、y 向下、z 向前，白底、蓝黑线条、橙色强调量，教科书风格，留出公式和中文标注空白，16:9。
>
> 负向提示词：鱼眼透视，错误坐标轴，光线不经过光心，多个成像平面，乱码公式，卡通相机，阴影杂乱。

## 12. RF-DETR：先找到人，再恢复人

InstantHMR 接收的是人物裁剪，而不是整张图。RF-DETR 先输出每个人的紧致边界框和置信度。程序按置信度排序，只保留 `--max-persons` 指定的人数。

Jetson 预设使用 RF-DETR nano，因为在完整流水线中，人体检测往往比 InstantHMR 本体更耗时。`--detector-stride 2` 表示：

- 第 0、2、4……帧执行 RF-DETR；
- 中间帧根据上一帧二维关键点重新估计人体框；
- 以较少检测计算换取相对连续的人体跟随。

检测步长越大，算力越省，但人物快速进出画面、相互遮挡或运动突变时，框更新可能滞后。场景变化大时应保持 2；动作较慢、镜头固定时可测试 3 或 4。

## 13. 人体裁剪与 CLIFF 相机条件

对每个检测框，程序会：

1. 适当扩大 RF-DETR 较紧的框，使之接近训练时的人体框分布；
2. 以人体框中心生成正方形裁剪；
3. 再扩大 1.2 倍保留人体边缘上下文；
4. 仿射缩放到模型输入尺寸 288×288；
5. 构造 CLIFF 相机条件。

在有真实相机内参时，CLIFF 条件为：

\[
\left[
\arctan\frac{c_x^{box}-c_x}{f},
\arctan\frac{c_y^{box}-c_y}{f},
\frac{s_{box}}{f}
\right]
\]

它告诉模型“这个人体裁剪来自原图的什么方向、占多大视角”，避免把所有裁剪都误认为位于相机正中央。

> **技术原理配图提示词｜人物裁剪与相机条件**
>
> 正向提示词：四阶段横向算法示意图：原始 1280×720 RGB 图中的人物检测框、扩大后的正方形框、仿射裁剪成 288×288 人体图、由框中心偏移与焦距构成的三元素相机条件；用相机主点十字、框中心、角度弧线和尺寸比例清楚解释 CLIFF 条件，白底矢量科研插图，蓝灰主色、橙色关键量，16:9。
>
> 负向提示词：错误裁剪比例，切掉头脚，多个无关人物，乱码向量，不准确透视，照片拼贴杂乱。

## 14. InstantHMR 与 MHR 人体表示

InstantHMR 网络从人体图像与相机条件预测：

- `mhr_params`：204 维 MHR 姿态和骨架参数；
- `shape_params`：45 维人体形状系数；
- `cam_trans`：人体从局部坐标移动到相机坐标的三维平移；
- `joints_2d`：70 个二维关键点。

随后 `mhr-body.onnx` 根据姿态与形状生成：

- `joints_3d_local`：人体局部坐标中的 70 个三维关节；
- `vertices_local`：人体局部坐标中的 18,439 个网格顶点。

相机坐标结果为：

\[
\mathbf J_{cam}=\mathbf J_{local}+\mathbf t_{cam}
\]

\[
\mathbf V_{cam}=\mathbf V_{local}+\mathbf t_{cam}
\]

坐标约定为 x 向右、y 向下、z 向前，距离单位为米。`--no-mesh` 不只是隐藏显示，它会跳过 18,439 顶点的输出，因此能显著降低内存、计算和 RRD 序列化负担。

## 15. 图内运动学拟合为什么能改善姿态

网络同时预测三维人体参数和二维关键点。项目观察到，网络直接预测的二维关键点通常比“三维骨架重新投影到二维”的位置更准确，因此 fit 图会执行一次带置信度权重的 Levenberg–Marquardt 更新，让三维骨架投影更贴合二维关键点。

可概括为最小化：

\[
E(\theta)=\sum_i w_i\left\|\pi(\mathbf J_i(\theta))-\hat{\mathbf j}_i\right\|^2
\]

其中：

- \(\theta\) 是待修正的人体姿态参数；
- \(\mathbf J_i(\theta)\) 是第 \(i\) 个三维关节；
- \(\pi\) 是使用真实焦距和主点的相机投影；
- \(\hat{\mathbf j}_i\) 是网络预测的二维关键点；
- \(w_i\) 是关键点置信度权重。

`--fit 1` 使用一个人体裁剪，适合 8GB Jetson 的质量与负载平衡；`--fit 2` 使用两个不同尺度裁剪并融合二维信息，质量通常更好，但计算、显存和启动成本更高。

> **技术原理配图提示词｜运动学拟合**
>
> 正向提示词：论文风格的三联图，左侧显示网络初始三维骨架投影与二维观测关键点存在小偏差，中间显示带置信度权重的重投影误差箭头与一次 Levenberg-Marquardt 更新，右侧显示更新后投影与二维关键点对齐；三维骨架青色、二维目标橙色、误差向量红色，白底、线条精确、公式区域留白，16:9。
>
> 负向提示词：错误关节连接，骨架变形，多步迭代迷宫，无法阅读的公式，照片背景，夸张动画效果。

## 16. 时序滤波：减少抖动而不过度滞后

视频逐帧独立推理会产生细小抖动。程序先用检测框交并比（IoU）把当前人物与上一帧轨迹匹配，再按人物分别维护滤波状态。

### 16.1 `--pose-filter one-euro`

One Euro Filter 会根据运动速度动态改变截止频率：静止时加强平滑，快速运动时提高响应速度。当前实现对 MHR 姿态角进行滤波，并正确处理角度跨越 \(-\pi\) 与 \(\pi\) 时的环绕问题。它在构建网格之前工作，因此骨架、网格和姿态参数保持一致。

### 16.2 `--pose-filter alpha-beta`

Alpha-Beta 是简单的恒速度预测—校正滤波器，计算量小，适合资源受限设备。它同样作用于姿态角。

### 16.3 `--smooth`

`--smooth` 直接平滑二维关键点、三维关键点、网格顶点和相机平移，视觉上可能更稳，但快速动作更容易产生可见滞后。不要同时把“更平滑”误认为“更准确”。

## 17. ONNX Runtime、CUDA 与 TensorRT

ONNX Runtime 使用 Execution Provider（执行提供程序）把图交给不同后端：

- `CUDAExecutionProvider`：当前实机的稳定首选；
- `TensorrtExecutionProvider`：可选加速，首次需要构建本机引擎；
- `CPUExecutionProvider`：作为回退，不适合 Orin Nano 上的完整实时链路。

CUDA 路径中的 fit 图具有固定输入形状。代码为输入输出分配固定 GPU 缓冲区，通过 I/O Binding 避免不必要的主机—设备复制，并启用 CUDA Graph：首次执行捕获图，后续帧重放，从而降低 Python 与内核启动开销。

TensorRT 路径使用 FP16、计时缓存和引擎缓存。引擎与 GPU 架构、TensorRT、CUDA、ORT、模型和 shape profile 强相关，所以只能在目标 Jetson 上生成。

> **技术原理配图提示词｜ORT 执行后端**
>
> 正向提示词：清晰的分层技术架构图，上层是 InstantHMR ONNX 图，中层是 ONNX Runtime 调度器，下层并列 CUDA Execution Provider、TensorRT Execution Provider、CPU fallback；CUDA 分支标出 I/O Binding、固定显存缓冲、CUDA Graph replay，TensorRT 分支标出 FP16、engine build、timing cache，最底层是 Jetson Orin GPU 与共享内存；白底深蓝矢量风格，逻辑箭头准确，16:9，留出后期中文标签。
>
> 负向提示词：错误层级，CUDA 与 TensorRT 串成同一必经路径，云端 GPU，虚假加速倍数，密集乱码，装饰性电路板。

## 18. cuDNN 9.6 非对称填充问题与 v2 修复

问题卷积的前后填充不同，例如二维空间维度上的前侧为 0、后侧为 1。ONNX 允许这种卷积，但在实机 cuDNN 9.6 前端、FP16、特定卷积形状和 HEURISTIC 搜索组合下，可能找不到执行引擎。

设原卷积某一维前填充为 \(a\)，后填充为 \(b\)。适配代码取：

\[
s=\max(a,b)
\]

先执行前后均为 \(s\) 的对称填充卷积，再从输出中裁掉因为额外填充产生的边界。得到的有效输出与原非对称卷积完全一致：

```text
原图：Asymmetric Conv(a, b)

改写：Symmetric Conv(s, s) → Slice(裁掉多余边界)
```

为什么不用“Pad → 无填充 Conv”？因为 ORT 1.20 的图优化器可能把显式 Pad 再融合回非对称 Conv，重新触发 cuDNN 问题。v2 让 cuDNN 实际看到的是对称卷积，后面的 Slice 保持独立。

缓存文件名包含源路径、大小、修改时间和补丁版本的指纹。源模型变化后会自动生成新缓存，原始文件始终不修改。

> **技术原理配图提示词｜非对称卷积等价改写**
>
> 正向提示词：白底论文级对比图，左侧用网格矩阵展示输入特征图执行非对称填充卷积，顶部和底部填充数量不同并导致 cuDNN 9.6 引擎选择失败；右侧展示相同输入先执行两侧相等的对称填充卷积，再通过 Slice 裁掉额外边缘，最终输出矩阵与左侧逐元素相同；用绿色等号连接结果，用红色小警示只标在原卷积执行引擎处，二维矩阵边界清晰，16:9，预留文字区域。
>
> 负向提示词：矩阵尺寸矛盾，Slice 裁错方向，结果不相等，大段乱码代码，三维卷积动画，霓虹特效。

## 19. RRD 为什么适合当前设备

如果在 Jetson 本地实时显示，系统要同时承担：

```text
相机采集 + RF-DETR + InstantHMR + MHR 网格生成
+ Rerun 序列化 + GUI 事件循环 + GPU 三维渲染 + 桌面合成
```

无界面录制把后半部分拆开：

```text
Jetson：相机 + 推理 + 顺序写 RRD
电脑：读取 RRD + 解码 + 2D/3D 渲染 + 交互回放
```

这不只降低平均负载，更重要的是减少相机、推理和渲染同时出现的峰值。`--rerun-stride 2` 仍然每帧运行模型，但只把每两个处理帧中的一个写入 RRD；它降低序列化和磁盘开销，不会让 RF-DETR/InstantHMR 自动少算一半。若希望减少检测计算，应调整 `--detector-stride`。

> **技术原理配图提示词｜实时显示与离线回放对比**
>
> 正向提示词：左右对比的工程负载图，左侧“Jetson 本地实时显示”把相机、AI 推理、磁盘写入、GUI、三维渲染五种负载堆叠到同一功耗峰值曲线上，并用克制红色标出超过载板瞬态预算；右侧“Jetson 录制 + 电脑回放”把推理与渲染分到两台设备，两个峰值明显分离，RRD 文件作为中间桥梁；扁平技术信息图、白底、蓝灰绿主色，曲线和箭头清晰，16:9。
>
> 负向提示词：伪造具体瓦数，爆炸画面，烧毁电路板，错误数据方向，云端服务器，乱码，夸张警报。

## 20. 为什么当前版本没有直接融合深度图

Gemini 2 的深度图测量的是相机看到的可见表面，例如衣服胸前、手臂外表或遮挡物表面；`cam_trans` 表示的是参数化人体在相机坐标系中的整体位置。直接用人物框中的深度中位数替换 `cam_trans.z`，会把“衣服表面深度”错误地当作“人体模型原点深度”，产生系统偏差，而且无法解决肢体遮挡和深度空洞。

严谨的 RGB-D 融合至少需要：

1. 彩色与深度流的硬件同步；
2. 深度到彩色的几何配准；
3. 深度尺度、无效值和噪声处理；
4. 把预测人体网格渲染到深度图；
5. 通过可见性掩码、鲁棒损失和时序约束优化人体平移、姿态或形状；
6. 在真实标注数据上验证是否改善，而不是只凭视觉感觉判断。

因此，当前适配优先保证对原始单目 RGB InstantHMR 的忠实复现。深度融合应作为独立、可评测的后续模块。

> **技术原理配图提示词｜为何不能用框内深度中位数替代人体中心**
>
> 正向提示词：侧视几何教学图，相机在左侧，右侧站立人体同时显示衣服可见表面、人体参数模型中心和完整三维网格；多条深度射线首先命中衣服表面，人物框深度中位数落在身体前表面，而 MHR 原点位于人体内部，两者用明显但克制的距离偏差箭头区分；右下角附正确方案小流程：RGB-D 配准、网格渲染、可见性、鲁棒优化，白底科研矢量风格，16:9。
>
> 负向提示词：X 光透视，裸露解剖，深度射线穿透人体，错误相机方向，乱码公式，恐怖风格。

## 21. 性能与质量调节的本质

| 调节项 | 降低资源占用 | 对结果的代价 |
|---|---|---|
| `--model nano` | HMR 参数量和延迟下降 | 姿态精度通常低于 small/medium |
| `--fit 0` | 跳过运动学拟合 | 二维重投影与姿态质量可能下降 |
| `--max-persons 1` | 减少 HMR 调用、网格和 RRD 数据 | 只保留最高置信度人物 |
| `--no-mesh` | 不生成 18,439 顶点 | 只保留骨架，无完整表面 |
| `--detector-stride 3/4` | RF-DETR 调用减少 | 快速进出和遮挡时跟踪变差 |
| `--rerun-stride 2/3` | RRD 写入、文件体积、可视化数据减少 | 回放时间采样更稀疏，不减少 HMR 推理本身 |
| 降低 RGB 分辨率 | 采集、转换、检测开销下降 | 远处和小人物更难检测 |
| 降低 FPS | 单位时间处理量下降 | 快速动作的时间分辨率下降 |
| TensorRT | 稳态模型推理可能更快 | 首次构建慢、缓存与版本耦合更强 |

建议始终从本文的 CUDA 安全基线出发，每次只改变一个参数，记录 RRD、控制台中位耗时、温度、过流计数和结果质量，再决定是否保留。

## 22. 参考资料与许可证提醒

- [InstantHMR 上游仓库](https://github.com/Naturalpad-studio/InstantHMR)
- [本 Jetson + Gemini 2 适配仓库](https://github.com/bladewin-smith/InstantHMR)
- [InstantHMR 模型仓库](https://huggingface.co/momolesang/InstantHMR)
- [Momentum Human Rig（MHR）](https://github.com/facebookresearch/MHR)
- [RF-DETR](https://github.com/roboflow/rf-detr)
- [Rerun](https://rerun.io)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Orbbec Gemini 2 产品页](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/)
- [Gemini 2 数据表](https://www.orbbec.com/staging/wp-content/uploads/2023/04/ORBBEC_Datasheet_Gemini-2.pdf)
- [NVIDIA Jetson Orin 系列功耗与性能](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

在发表、分发或商用之前，请完整阅读仓库根目录中的 `LICENSE`、`LICENSE-SAM` 和 README 的 Licence 部分。本文对许可证的概述不能替代法律审查。

## 附录 A：最常用命令速查

环境检查：

```bash
source .venv-jp6-v2/bin/activate
python tools/check_jetson.py
```

相机检查：

```bash
python tools/check_jetson.py --camera
```

真实 HMR CUDA 冒烟测试：

```bash
python tools/check_jetson.py --hmr-smoke
```

推荐 30 秒录制：

```bash
python demo.py \
  --orbbec --jetson --device cuda \
  --pose-filter one-euro \
  --no-viewer --save-rrd jetson-run.rrd \
  --duration 30 --rerun-stride 2
```

最低负载录制：

```bash
python demo.py \
  --orbbec --jetson --device cuda \
  --model nano --fit 0 --max-persons 1 --no-mesh \
  --no-viewer --save-rrd jetson-lite.rrd \
  --duration 30 --rerun-stride 3
```

电脑端回放：

```bash
rerun jetson-run.rrd
```

## 附录 B：关键文件与缓存位置

| 路径 | 内容 | 是否可直接删除 |
|---|---|---|
| `requirements-jetson.txt` | JetPack 6 的依赖边界 | 不应删除 |
| `tools/check_jetson.py` | 环境、相机和真实 HMR 检查 | 不应删除 |
| `instanthmr/camera_orbbec.py` | Gemini 2 原生取流与内参 | 不应删除 |
| `instanthmr/onnx_compat.py` | cuDNN 非对称填充兼容改写 | 不应删除 |
| `models/onnx_compat/` | 由原模型生成的兼容图缓存 | 可重建；删除后下次重新生成 |
| `models/trt_cache/` | 当前 Jetson 的 TensorRT 引擎和计时缓存 | 可重建；版本或模型变化后应重建 |
| `~/.cache/huggingface/hub/` | InstantHMR 模型下载缓存 | 离线使用时不要误删 |
| `*.rrd` | 图像、姿态、网格和耗时录制文件 | 属于用户数据；确认备份后再删除 |

## 附录 C：统一配图风格建议

为了让整篇教程视觉连贯，可以统一使用：

- 16:9 横版；
- 白色或深灰中性背景；
- 蓝灰为主色、青绿色表示三维人体、橙色表示测量量或关键步骤；
- 硬件场景使用写实工程摄影，算法原理使用扁平矢量论文插图；
- 不让生图模型直接生成长段中文，预留标注区后期排版；
- 所有骨架图都检查关节拓扑、左右肢体和相机坐标方向；
- 不在图片中写未经实测的 FPS、功耗、精度和认证结论。

