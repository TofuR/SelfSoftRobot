# 实机验证 App：Hereditary / Analytic B

更新：2026-09-11。默认入口是四页模型验证 App，操作详见 [实验指南](HEREDITARY_GUIDE.md)。采集训练数据的 `real_capture` 是另一个程序。

## 安装与启动

建议使用独立的 **Python 3.10、64 位环境**。源码仓库中在项目根目录运行；独立 ZIP 解压后在包含 `real_validation/` 的目录运行：

```bash
python -m pip install -r real_validation/requirements.txt
python -m real_validation.main
```

Windows 可双击 `real_validation/run_gui.bat`。基础依赖已经包含传统分割、中心线提取与 GUI；不必再重复安装 `requirements-perception.txt`。即使当前 Hereditary 数值模型主要用 NumPy，现有 GUI 共用的运行时导入仍需要 PyTorch。

按实际使用的功能添加依赖：

| 功能 | 安装命令或文件 |
| --- | --- |
| SAM2 初始化（当前默认） | `python -m pip install -r real_validation/requirements-sam2.txt`，或 Windows 双击 `install_sam2_windows.bat` |
| 两组真实串口阀 | `python -m pip install pyserial` |
| RealSense D415 / 兼容彩色流设备 | `python -m pip install pyrealsense2` |
| 普通 UVC 摄像头 | 基础 OpenCV 和操作系统视频驱动即可，不需要 RealSense SDK |
| 可选 NDI Aurora | `python -m pip install scikit-surgerynditracker`；SciPy 已在基础依赖中 |
| 一次装齐现有硬件依赖 | `python -m pip install -r real_validation/requirements-hardware.txt` |
| YOLO 分割候选的离线检查 | `requirements-yolo.txt`；**当前 GUI 尚未接入 YOLO**，安装依赖不会出现新的运行功能 |

PyTorch / torchvision 固定为当前验证使用的 2.6.0 / 0.21.0 配对。requirements 不指定 CUDA 轮子来源；需要 GPU 时按 [PyTorch 对应版本安装说明](https://pytorch.org/get-started/previous-versions/#v260)选择适合驱动的轮子，再安装本目录依赖。没有 NVIDIA GPU 可使用 CPU。服务器验证不等于每台 Windows 电脑都已验收。

## 四页实验流程

| 页面 | 正常操作 | 结果 |
| --- | --- | --- |
| 1 设备模型 | 加载控制模型，确认六腔→四输入映射，分别连接阀与相机；NDI 可选 | 加载不发压力、不伪造准备动作历史 |
| 2 部署预热 | 六腔工具先调到初始压力并保持；自动提取，检查 BASE/TIP，确认部署 | 拟合尺度/坐标与当前状态，再用连续图像预热；修正工具默认折叠 |
| 3 目标规划 | 设置完整形状、末端或局部目标，规划并播放预览 | 参数窗口可调容限、搜索上限、预算和末端余量 |
| 4 执行记录 | 确认计划后执行；可选择关闭矫正作对照 | 保存原图、反馈图、压力指令/ACK、NDI、后缀变化与逐步耗时 |

全局 **六腔控制** 可手动调压和设置每腔范围、升降速率。共享输入同压，限制与模型部署范围取交集；窗口显示实际生效速率。当前候选控制模型速率上限 50 kPa/s，支持正压，不支持实际负压。ACK 是下发应答，不是压力传感器读数。[调压问题诊断](MANUAL_PRESSURE_DIAGNOSTICS.md)。

**余量 0 是额外 0 步，不是 0 kPa，不会清零。** 默认余量 10 步计入预览，起始动作保持末条压力供闭环继续调整；全部步数结束后保持末压。正常完成不等于实机到位，结果中另报模型估计残差。

清零、手动调压或反馈暂时失效后，配准及动作历史仍有效时，第二页同一按钮可 **复用配准重新预热**。预热本身不发送新压力；相机改变或历史未知时按提示处理。

## 模型与感知文件

控制模型使用 **同名 `.npz + .json`**，不是旧 OpenLoop 的 `.pt`：

- 发布包默认：`checkpoints/hereditary_current/hereditary.npz` 与 `hereditary.json`。
- 新 5 Hz、新 10 Hz 及旧参考：发布包的 `checkpoints/candidates/`；来源与选择指标见 [候选说明](checkpoints/candidates/README.md)。
- JSON 中的 `dt` 决定控制模型频率，不根据相机帧率或文件名推断；换频率时换模型，不单改 `dt`。
- SAM2 初始化需要 `checkpoints/sam2/sam2.1_hiera_tiny.pt`、`vendor/sam2/` 上游源码及其依赖。官方导出的完整 SAM 部署 ZIP 已包含源码和分割权重；仅复制 Git 源码目录不会自动包含这些被忽略的大文件。
- YOLO 训练在服务器离线进行，目前不参与初始化或实时反馈；未来分割权重与控制权重分开放置，见 [YOLO 独立包方案](YOLO_DEPLOYMENT_PLAN.md)。

当前绿色是冻结帧分割，青色是模型估计，白点是可信可见边缘，紫色为规划预览，红色为目标。空闲且没有计划时有局部边缘反馈；手动调压期间按 ACK 历史预测；已生成计划后保持预览一致，不持续改变计划对应状态。外力摆动的连续分割跟随仍是后续工作。

## 图像链路与耗时

正常流程就是 **相机图像 → 分割 mask → 中心线与短边端点修正**。模型需要的缩放/颜色格式转换、将输出 mask 还原到原图等由框架或适配器自动完成，不增加人工步骤，也不要求 YOLO 后再调用 SAM。

记录“图像到可用形状”的耗时，是为了让界面和控制周期与真正拿到结果的时间一致。某些推理 API 已包含输出重建，则直接计一次调用，不能重复相加。相机等帧、模型加载、分割、中心线与一次性的部署配准分别记录；不把它们全部重复塞进每次推理。

## 保存位置

第一页高级设置的 **保存根目录** 为准。当前程序未手动配置时默认 `<real_validation>/runs/`，源码仓库的正式实验应将该项设为 `workspace/runs/validation/`；独立 PC 可选本机数据盘。此处如实描述当前 GUI 默认值，未改变路径解析行为。

```text
<保存根目录>/hereditary/<时间戳_唯一标识>/
  events.jsonl                 # 动作历史、配准、预热和执行事件
  initial_*.png / *_mask.png    # 初始化原图与分割
  plan_*.npz                   # 每次规划预览
  manual_diagnostics/          # 手动保存的通信诊断
  executions/<唯一执行标识>/
    initial_plan.npz
    metadata.json              # 执行模式、主/余量步数、最终估计
    commands.csv / samples.csv / timings.csv
    steps.jsonl / feedback_jobs/
    raw/cam0/ / frames/         # 原图与用于反馈的图像
    ndi.csv                    # 可选 NDI 评价记录，状态见 metadata
```

具体字段与证据含义见 [执行结果文件](HEREDITARY_GUIDE.md#执行结果文件)。清理/更换部署包前保留实验输出及设备电脑自己的配置。

## 独立包与检查

优先使用 `tools/package_hereditary.py` 生成的完整 ZIP。它包含当前 App、明确选择的控制候选、可选 SAM 源码/权重、依赖说明、HTML 指南及文件 SHA256 清单，不依赖服务器 `src/` 或 `real_capture/`。`config/hardware.json` 保存电脑本地配置，不作为他人机器的默认配置发布。

安装后可做不连接设备的启动检查：

```bash
python -c "from real_validation.gui.main_window import ValidationWindow; print('GUI imports OK')"
```

随后在第一页选择 Mock 并逐项连接，用模型相机跑通四页流程；真实端口/USB通信须在设备电脑检查。仓库内开发检查按改动范围运行聚焦测试与文档检查，不将 Mock 通过写成实机控制成功。

## 旧 OpenLoop 兼容入口

仅显式运行 `python -m real_validation.main --legacy-openloop` 时使用旧 `.pt + config.json + deploy_manifest.json`、离线 Anchor 与旧状态机，详见 [旧 GUI 指南](GUI_GUIDE.md)。这些不是默认四页流程的前置步骤。
