# 六通道气压阀 · NDI 末端 · 相机  同步采集（real_capture）

> 自包含的实物数据采集程序：用 **Modbus 6 通道电流型比例阀（4–20mA，0–500kPa）** 驱动软臂，
> **NDI Aurora 电磁导航** 读末端 3D 位姿，**RealSense** 取图像，三路在**动作门控**下时间同步，
> 产出 `scripts/real/capture_to_npz.py` 直接能吃的 raw 数据 → 训练状态转移/形状模型。

与旧 `docs/ref/Main UI-plc/`（1-DOF 单通道）的关系：本项目自包含、可 git 提交（旧目录在
`docs/ref/` 下被 gitignore）。6 通道向后兼容单通道（见下）。

## 文件
| 文件 | 作用 |
|---|---|
| `modbus_manager.py` | Modbus RTU 协议 + 2 组×3 通道控制（原样复制，仅把 `import serial` 改成可选） |
| `realsense_cam.py` | RealSense RGB 捕获线程（+ mock 合成剪影帧，原样复制） |
| `nditracker.py` | NDI Aurora 封装 `ndi_load` / `get_ndi_value`（原样复制） |
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
2. **设通道范围**：每个通道 `min/max`（kPa）。**单通道**：只给目标通道设范围，其余 5 路 `min=max=0`。
3. 模式选 **Manual**（手动目标）/ **Random**（随机游走）/ **Sweep**（往返扫描）；
   填 `动作间隔`（默认 0.2s）+ `稳定等待`（默认 0.19s）→ 勾 **自动时间戳命名** → **▶ 开始采集**。
4. **■ 停止采集** → **⚡ 生成 npz**（自动先出 `tip.npz`，再调 `capture_to_npz`）/ **📋 导出汇总 CSV**。

## 2. 动作门控采集（核心）
动作每 `动作间隔`（默认 **0.2s**）下发一次；下发后等 `稳定等待`（默认 **0.19s**）让软臂稳定，
再到缓存里取**最新一帧 + 一个 NDI 末端**落盘。每拍产出一组同索引 `(action_i, frame_i, ndi_i)`：
```
t=0.00s  下发 action_i  → 6 路阀 + 记气压
t=0.19s  抓 frame_i + ndi_i（缓存最新值）→ 同索引落盘
t=0.20s  下一拍
```
- `actions6.csv` 与 `frame_times.txt` **同索引、同时刻**（记录 grab 时刻 + 当时仍在指挥的 `action_i`）→
  下游 `capture_to_npz --actions-has-timestamps --frame-times` 插值退化为**精确配对**，无串扰。
- 相机/NDI 自由运行做预览 + 缓存；时钟到点只读缓存（不阻塞、不丢拍）。

## 3. 六通道 → 单通道（向后兼容）
- 把不用的 5 路 `min=max=0`，只给目标通道设范围 → 自动只动那 1 路。
- `主通道` 下拉选 legacy `pressure.csv` 用哪一路（默认 ch0）。
- 同时生成两份动作日志：
  - `actions6.csv`：`t, c0,c1,c2,c3,c4,c5`（7 列，新管线 `--actions` 自动探测 A=6）
  - `pressure.csv`：`t, p_active, 0`（3 列，旧 1-DOF 文档命令照用）

## 输出（每个序列目录 `<本目录>/data/raw/seq_<时间戳>/`）
| 文件 | 内容 | 去向 |
|---|---|---|
| `cam0/00000.png …` | 零填充帧 | `--view-dirs` |
| `frame_times.txt` | 每帧一行 相对秒 | `--frame-times` |
| `actions6.csv` | `t_sec, c0..c5`（无表头） | `--actions --actions-has-timestamps` |
| `ndi.csv` | `t_sec, x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality`（失锁行=空） | → `tip.npz` → `--ndi-tip` |
| `pressure.csv` | `t_sec, p_active, 0`（兼容旧） | 旧 2 列命令 |
| `meta.json` | t0 / ISO / 模式 / 间隔 / settle / 各通道范围 / 帧数 | 复现 |
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
  --view-dirs data/raw/seq_XXXX/cam0 --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 --dt 0.0333 \
  --actions data/raw/seq_XXXX/actions6.csv --actions-has-timestamps \
  --frame-times data/raw/seq_XXXX/frame_times.txt \
  --ndi-tip data/raw/seq_XXXX/tip.npz \
  --planar-lift --clean-nan --out data/real_seq/seq_XXXX.npz
# 3) 训练（GT 单步状态转移）
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py --data_dir data/real_seq
```
> 单相机 + 平面弯曲用 `--planar-lift`；多相机三角化去掉它并给多视角。详见
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
