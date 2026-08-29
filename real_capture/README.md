# 六通道气压阀 · NDI 末端 · 相机  同步采集（real_capture）

> 自包含的实物数据采集程序：用 **Modbus 6 通道电流型比例阀（4–20mA，0–500kPa）** 驱动软臂，
> **NDI Aurora 电磁导航** 读末端 3D 位姿，**RealSense 多相机**取图像，各视角在**动作门控**下时间同步，
> 产出 `scripts/real/capture_to_npz.py` 直接能吃的 raw 数据 → 训练状态转移/形状模型。

与旧 `docs/ref/Main UI-plc/`（1-DOF 单通道）的关系：本项目自包含、可 git 提交（旧目录在
`docs/ref/` 下被 gitignore）。当前仍支持单通道操作，但统一使用六通道动作格式。

## 文件
| 文件 | 作用 |
|---|---|
| `modbus_manager.py` | Modbus RTU 协议 + 2 组×3 通道控制（原样复制，仅把 `import serial` 改成可选） |
| `realsense_cam.py` | 单台 RealSense RGB 捕获线程（按序列号选择，+ mock 合成剪影帧） |
| `nditracker.py` | NDI Aurora 封装 `ndi_load` / 多探头位姿读取 |
| `hardware_threads.py` | `NdiThread`（真）+ `MockNdiThread`（合成末端轨迹） |
| `valve_control.py` | `ValveController`（6 维→2 组）+ `MockValveController` + `ValveDriver`（随机/扫描） |
| `recorder.py` | `SaveThread` + `ValveRecorder`（动作门控同步核心）+ `build_ndi_tip_npz` / `export_summary_csv` |
| `main_capture.py` | PyQt5 GUI（6 通道控制 + NDI + 相机预览 + 采集 + 后处理） |

## 0. 先跑通（无硬件，验证流程）
```bash
pip install -r requirements.txt          # 至少 PyQt5 pyqtgraph numpy opencv-python
python main_capture.py --mock
```
应看到：假相机预览在动 + 6 路气压曲线在动 + NDI 末端 XY 轨迹在动 + 所有按钮可点 → 整条链路 OK。

## 1. 真机使用（GUI）
```bash
python main_capture.py --group1 COM3 --group2 COM46 --ndi COM9
```
1. **连接 Modbus**（两组串口；2 组 × 3 通道 = 6 路）→ **连接 NDI**（末端串口）；相机已自动开预览。
2. GUI「相机」分组设置相机数和序列号（逗号分隔；空白表示自动选择），点击「应用/重连」；右侧可选择单路或平铺预览。
3. **设通道范围**：每个通道 `min/max`（kPa）。**单通道**：只给目标通道设范围，其余 5 路 `min=max=0`。
4. 受约束实验：主通道选 `all`，在“通道来源”中为每个 `chi` 选择根通道；选择自身表示独立，
   选择其他通道表示跟随。可配置任意大小的同源组，链式选择会压平，循环会拒绝并恢复上一有效配置。
   follower 的 target/min/max/rise/fall 会镜像并锁定；必须先确认真实腔道 mapping。
5. 模式选 **Manual** / **Random** / **Sweep** / **Replay**；设置动作间隔、稳定等待、每通道
   rise/fall 速率上限（kPa/s，填 0 表示不限速），Random 可填 seed 和预生成步数，Replay 选择已有 `actions6.csv`。
6. **■ 停止采集** → **⚡ 生成 npz**（自动把 `cam0...camN` 全部传给 `capture_to_npz`）/ **📋 导出汇总 CSV**。

来源约束对 Manual/Random/Sweep/Replay 全部生效。权威合同写入
`meta.json.channel_source6`，`channel_equalities` 作为旧工具兼容派生字段；
`commands.csv` 同时保存约束前 `proposed0..5`、约束后 `requested0..5`、最终
`action_command0..5` 和每个 follower 的 `pair_residualN`。linked 通道范围/速率/当前命令不一致或最终
`applied6` 残差超限时会 fail-closed 停止。完整平面实验流程见
[`docs/real_data/planar_constrained_6ch_workflow.md`](../docs/real_data/planar_constrained_6ch_workflow.md)。

## 2. 动作门控采集（核心）
动作每 `动作间隔`（默认 **0.2s**）下发一次；下发后等 `稳定等待`（默认 **0.19s**）让软臂稳定，
再到缓存里取**最新一帧 + 多个 NDI 末端**落盘。每拍产出一组同索引 `(action_i, frame_i, ndi_i)`：
```
t=0.00s  下发 action_i  → 6 路阀 + 记气压
t=0.19s  抓 frame_i + ndi_i（缓存最新值）→ 同索引落盘
t=0.20s  下一拍
```
- `actions6.csv` 与 `frame_times.txt` **同索引、同时刻**（记录 grab 时刻 + 当时仍在指挥的 `action_i`）→
  下游 `capture_to_npz --actions-has-timestamps --frame-times` 插值退化为**精确配对**，无串扰。
- 相机/NDI 自由运行做预览 + 缓存；时钟到点只读缓存（不阻塞、不丢拍）。

## 3. 六通道 → 单通道
- 把不用的 5 路 `min=max=0`，只给目标通道设范围 → 自动只动那 1 路。
- 单通道仍生成 `actions6.csv`，未使用通道保持为 0；旧的 `pressure.csv` 不再生成。

## 输出（每个序列目录 `<本目录>/data/raw/seq_<时间戳>/`）
| 文件 | 内容 | 去向 |
|---|---|---|
| `cam0/00000.png … camN/00000.png` | 每个相机一个零填充帧目录，同一索引为同一拍 | `--view-dirs` |
| `frame_times.txt` | 每帧一行 相对秒 | `--frame-times` |
| `actions6.csv` | `t_sec, c0..c5`（首行表头） | `--actions --actions-has-timestamps` |
| `ndi.csv` | `t_sec + ndi0_* ... ndiN_*`（每探头 11 列） | → `tip.npz` → `--ndi-tip` |
| `commands.csv` | 命令时间、ACK、proposed/requested/applied 六维动作、来源 residual、分组通信状态 | 通信/约束 QC / 复现 |
| `samples.csv` | `t_grab`、总体/逐相机 `frame_age`、各 NDI age/quality | 数据质量筛选 |
| `meta.json` | t0 / 相机与 NDI / 模式 / seed / 范围与速率 / `channel_source6` / 兼容 equalities / age 阈值 / 帧数 | 复现 |
| `summary.csv` | **(可选)** 按帧对齐：6 路气压 + NDI xyz + 图像名 | 人眼检查 |

> 保存路径默认 `<main_capture.py 所在目录>/data/raw/seq_<时间戳>`（按 **py 文件位置**解析，不依赖运行 cwd；
> 跨机器不踩绝对路径坑——`real_capture_config.ini` 里的绝对路径若父目录在本机不存在会被忽略）。

## 4. 之后：出 npz + 训练
GUI「⚡ 生成 npz」按钮一键完成（先 `tip.npz`，再 `capture_to_npz`），等价于手跑：
```bash
# 1) NDI 末端 → tip.npz（recorder.build_ndi_tip_npz）
python -c "import sys; sys.path.insert(0,'.'); from recorder import build_ndi_tip_npz as f; print(f('data/raw/seq_XXXX'))"
# 2) 图像+气压+末端 → 仿真 schema .npz
python scripts/real/capture_to_npz.py \
  --view-dirs data/raw/seq_XXXX/cam0 data/raw/seq_XXXX/cam1 --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 --dt 0.0333 \
  --actions data/raw/seq_XXXX/actions6.csv --actions-has-timestamps \
  --frame-times data/raw/seq_XXXX/frame_times.txt \
  --ndi-tip data/raw/seq_XXXX/tip.npz \
  --planar-lift --clean-nan --out data/real_seq/seq_XXXX.npz
# 3) 训练（GT 单步状态转移）
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py --data_dir data/real_seq
```
> 单相机 + 平面弯曲用 `--planar-lift`；多相机三角化去掉它并给全部视角。详见
> `docs/real_data/workflow.md`。

## mock 选项（任意组合）
- `--mock`：三个组件全 mock（= `--mock-cam --mock-valve --mock-ndi`）。
- `--mock-cam` / `--mock-valve` / `--mock-ndi`：单选或任意组合，如 `--mock-ndi --group1 COM3 --group2 COM46`。
- 完整运行示例见 `main_capture.py` 顶部 docstring。

## 常见坑
- **自动驱动前必须 连接 Modbus**（否则 Random/Sweep 被拦）。
- `camera_params.npz` 要先用 `calibrate_cameras.py` 生成（一次性）。
- 串口被占用会连不上 → 别同时开其它串口监视器。
- NDI 失锁（quality NaN）记 NaN/空，`--clean-nan` + `tip.npz` 插值会补。
- 退出请用窗口关闭（安全停所有线程 + 关 Modbus + 落盘）。
- 真机依赖：`pyserial`（Modbus）、`pyrealsense2`（相机）、`scipy` + `scikit-surgerynditracker`（NDI）。
  未装时 mock / 仅部分硬件模式仍可运行（`import` 都是可选的）。
