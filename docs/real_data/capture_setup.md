# 实物硬件采集系统构成与操作

> 本文只讲一件事：**硬件怎么构成、怎么采出 `cam0/`、`actions6.csv`、`ndi.csv`**。
> mask 分割 / 骨架化 / 模型训练见 [`workflow.md`](workflow.md)。
> 采集程序源码在仓库根 [`real_capture/`](../../real_capture/)（自包含、可 git 提交）；旧 1-DOF 单通道版在 [`docs/ref/Main UI-plc/`](../ref/Main%20UI-plc/)（参考，已被 gitignore）。

---

## 1. 硬件构成

被控对象是一只 **1-DOF 双段硅胶软体臂**：白半透明硅胶管，悬臂竖立，单腔道充气时在平面内弯曲。
围绕它有四类设备，全部接到同一台采集 PC：

| 子系统 | 设备 | 角色 | 接口 |
|---|---|---|---|
| **驱动 (气动)** | 6 通道电流型比例阀（2 组 × 3 路） | 向硅胶腔体打气，**气压 = 动作信号** | 2 路 USB-RS485，Modbus RTU |
| **观测 (图像)** | Intel RealSense D400 | 取 RGB 帧（仅 color stream） | USB |
| **度量 (末端)** | NDI Aurora 电磁导航 6-DOF tracker | 末端毫米级位姿（独立验证用，**非模型输入**） | USB-串口 |
| **采集 PC** | 普通 PC | 跑 `real_capture/` 同步采集程序 | — |

### 1.1 双段硅胶臂

- 白半透明硅胶管，蓝色背景，白色气管——这是后段 `white_on_blue` 分割的设计前提。
- **1-DOF 平面弯曲**：本工作流只驱动末端腔道（ch0），臂在图像平面内弯曲，z≈0（平面假设）。
- 末端贴 NDI 微型传感器，提供与图像同物理点的毫米级 GT（用于 px→mm 仿射自标定，见工作流文档 §1）。

### 1.2 驱动：6 通道电流型比例阀（Modbus RTU）

**这是真实控制量的来源**。比例阀响应的是 Modbus 指令，谁发都行——`real_capture/modbus_manager.py` 就是和阀硬件的完整契约。

- **6 通道 = 2 组 Modbus RTU × 每组 3 个阀**（`valve_control.N_CHAN = 6`）。
- 每组用 1 个 **USB-RS485 转换器**串联 3 个阀；插上 PC 后出现 **2 个新串口**。
- 电气接口：**4–20 mA 电流型**，对应 **0–500 kPa** 气压（寄存器值 4000–20000，线性换算见 `modbus_manager.ModbusRTU.pressure_to_register_value_current`）。
- 通道编号：`ch0-2` 属组 1，`ch3-5` 属组 2。**Modbus↔ch 联动**：只连组 1 → 只能控制 ch0-2。
- 通讯参数：**波特率 9600、从站地址 1**（默认；实际以对方硬件为准），8N1，CRC16 校验。
- 协议：功能码 `0x06`（写单寄存器）/ `0x10`（写多寄存器）。

> 比例阀**没有反馈**——下发即动作，落盘的 `actions6.csv` 记的是**下发值**（与仿真 action 语义一致）。

### 1.3 观测：Intel RealSense（免标定）

- 只取 **color stream**（`realsense_cam.py`，默认 640×480 @30fps，BGR8）。
- **背光剪影法**：关自动曝光、压短曝光/低 gain，让臂成纯黑剪影（`_apply_exposure`）。
- **免相机标定**：state 直接是图像骨架像素 `[col,row,0]`，全程不用相机内参 / 不投影。`config/real_camera_params.npz` 只在转 npz 的 `--planar-lift` 时用到，非必需。

### 1.4 度量：NDI Aurora 6-DOF（独立验证）

- 电磁导航，读末端 **3D 位姿**：`[x, y, z, Rx, Ry, Rz, qw, qx, qy, qz, quality]`（11 维，单位 mm / 度 / 无量纲）。
- 封装 `nditracker.py`（`ndi_load(port)` / `get_ndi_value(tracker)`），线程 `hardware_threads.NdiThread`（50 Hz）。
- **失锁处理**：`quality` 为 NaN 或哨兵 10000 时，把 `x..qz` 全置 NaN（`_normalize`），下游 `--clean-nan` + `tip.npz` 插值补。
- **仅作度量**：NDI 不进模型输入，只在 `eval_real_quant.py` 做末端 mm 验证（px→mm 仿射自标定）。

### 1.5 与旧版（TwinCAT + 注射器电机 + Arduino 气压）的关系

旧 `docs/ref/Main UI-plc/` 是 **1-DOF 单通道版**：TwinCAT PLC（pyads，`192.168.50.56.1.1:851`）驱动电机推注射器，Arduino 读 I2C 气压传感器（COM4@9600）。**当前主线已切换为 6 通道 Modbus 比例阀**（`real_capture/`），向后兼容单通道（把其余 5 路 `min=max=0` 即只动 1 路）。TwinCAT/注射器电机方案仅作历史参考保留。

---

## 2. 采集程序（`real_capture/`）

`real_capture/` 是**自包含**的采集程序，不依赖仓库 `src/`，整个文件夹可单独部署。

### 2.1 目录结构

```
real_capture/
  main_capture.py        # PyQt5 GUI 入口（界面纯代码构建，无 .ui 文件）
  recorder.py            # ★ 动作门控同步采集核心 ValveRecorder
  modbus_manager.py      # Modbus RTU 协议 + 2 组×3 通道控制（与阀硬件的契约）
  valve_control.py       # 6 维→2 组映射 + 自动驱动 ValveDriver(random/sweep)
  hardware_threads.py    # NdiThread(真) + MockNdiThread(合成末端轨迹)
  nditracker.py          # NDI Aurora 封装
  realsense_cam.py       # RealSense RGB 捕获线程（+ mock 合成剪影）
  requirements.txt       # 依赖
  DEPLOY.md              # 部署到新机器 / 接六通道阀的说明
  real_capture_config.ini # 串口/参数（机器相关，已 gitignore）
  data/
    raw/<seq>/           # ← 采集产物（cam0/, actions6.csv, ndi.csv, ...）
```

> 数据落盘到 `real_capture/data/raw/seq_<时间戳>/`（按 `main_capture.py` 所在目录解析，不依赖运行 cwd）。

### 2.2 动作门控同步（核心，`recorder.py::ValveRecorder`）

三路（阀 / 相机 / NDI）**不是各采各的再对齐**，而是由一个**采集时钟**驱动，保证 `(action_i, frame_i, ndi_i)` 同索引同时刻：

```
采集时钟 _clock（QTimer，period = 动作间隔，默认 0.2 s）
  │
  ├─ _on_tick()：
  │     1. driver.next_action() → 6 维气压向量 a
  │     2. controller.set_pressures(a) → 下发给两组 Modbus 阀 + emit action_logged
  │     3. QTimer.singleShot(settle_s, _on_grab(a))   # 默认 settle=0.19 s
  │
  └─ _on_grab(a)：settle 后到点
        - t_grab = monotonic() - t0
        - 取缓存最新一帧 → cam0/{idx:05d}.png + frame_times.txt
        - 取缓存最新 NDI 末端 → ndi.csv（与 a 同 t_grab 同 idx）
        - a 本身 → actions6.csv（同 t_grab 同 idx）
        - idx += 1
```

要点：

- **单一时钟** `t0 = time.monotonic()` 在开始录制时定，`frame_times / actions6 / ndi` 全是相对秒，共享同一原点。
- **相机 / NDI 自由运行**做预览 + 最新值缓存（GUI 线程独占读写，无锁）；时钟到点只读缓存，不阻塞、不丢拍。
- `settle_s` 必须 < `action_interval_s`（留 5 ms 给下一拍），保证抓取不跨入下一拍。
- `actions6.csv` 与 `frame_times.txt` **同索引、同时刻** → 下游 `capture_to_npz --actions-has-timestamps --frame-times` 的插值退化为**精确配对**，无串扰。
- PNG 落盘交给 `SaveThread`（异步 `cv2.imwrite`，不卡 GUI；哨兵 + 排空不丢尾部帧）。

### 2.3 阀控制层（`valve_control.py`）

- `ValveController`：把 6 维向量 `[c0..c5]` 拆成两组下发（组1=`[c0,c1,c2]`、组2=`[c3,c4,c5]`）；**只写给已连接的组**（未连接组不发命令，避免无效串口写）。每次下发 emit `action_logged(6vec, monotonic)`。
- `ValveDriver`：纯计算下一拍 action（random 有界游走，反射后强制钳位 / sweep 往返扫描）。每通道在 `[lo_i, hi_i]` kPa 内；`lo_i==hi_i`（range=0）的通道恒定 → **单通道模式就是把其余 5 通道 min=max=0**。
- 模式：`manual`（每拍重发 GUI 目标）/ `random`（随机游走）/ `sweep`（往返扫描）。

---

## 3. 采集流程与产物

### 3.1 流程（GUI）

```bash
cd real_capture
# 1) 全 mock（无硬件，先验 GUI + 整条链路）
python main_capture.py --mock
# 2) 真阀 + 假 NDI（最常用的实物调试起点）
python main_capture.py --mock-ndi --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1
# 3) 真机全用（两组 Modbus + NDI + 相机）
python main_capture.py --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1 --ndi /dev/ttyUSB2
```

操作步骤：

1. **连接 Modbus**：左上角填组1/组2 串口 + 波特/从站 → 点「组1 连接」（默认只连组1 即可控制 ch0-2）；需 ch3-5 再点「组2 连接」。**再点一次同一按钮 = 断开该组**（安全释放串口）。
2. **设通道范围**：每个通道 `min/max`（kPa）。单通道起步：主通道选 `ch0`，其余 5 路 `min=max=0`（自动灰锁）。
3. （可选）**连接 NDI**（再点 = 断开，安全释放 Aurora）；相机已自动预览。
4. 填「保存目录」、选「模式」（手动/随机/扫描）、「动作间隔」`0.2` s、「稳定等待」`0.19` s，勾「自动时间戳命名」。
5. 「▶ 开始采集」→ 采够后「■ 停止采集」。
6. 后处理：「⚡ 生成 npz」（先出 `tip.npz`，再调 `capture_to_npz`）或「📋 导出汇总 CSV」。

### 3.2 产物（每个序列目录 `real_capture/data/raw/seq_<时间戳>/`）

| 文件 | 内容 | 下游用途 |
|---|---|---|
| `cam0/00000.png ...` | 零填充帧（480×640 BGR） | `capture_to_npz --view-dirs` |
| `frame_times.txt` | 每帧一行 相对秒（grab 时刻） | `--frame-times` |
| `actions6.csv` | `t_sec, c0..c5`（**首行表头**；kPa） | `--actions --actions-has-timestamps` |
| `ndi.csv` | `t_sec, x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality`（首行表头；失锁写 `nan`） | → `tip.npz` → `--ndi-tip` |
| `pressure.csv` | `t_sec, p_active, reserved`（首行表头；旧版兼容 3 列） | 旧 1-DOF 命令；**训练以 actions6.csv 为准** |
| `meta.json` | 运行元信息（含 `hi6`/`lo6`/`mode`/间隔/settle/帧数） | 复现 + 下游归一化 |
| `summary.csv` | （可选）按帧对齐：6 路气压 + NDI xyz + 图像名 | 人眼核对 |

**`actions6.csv` 表头与示例**（首行表头，7 列）：
```
t_sec,c0,c1,c2,c3,c4,c5
0.199876,150.0000,0.0000,0.0000,0.0000,0.0000,0.0000
0.399812,143.2500,0.0000,0.0000,0.0000,0.0000,0.0000
```
> 本序列（`seq_20260627_172916`）只驱动 ch0，故 c1..c5 全 0；`t_sec` 是 grab 时刻相对秒。

**`ndi.csv` 表头**（首行表头，12 列）：
```
t_sec,x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality
0.199876,12.345,-8.901,1.234,0.5,-1.2,0.3,0.999,0.01,-0.02,0.03,0.95
```
> 失锁行写 `nan`（`np.loadtxt` 能解析；写空串会让 loadtxt 崩）。

**`meta.json` 示例**（关键：`hi6` 是每通道**操作上限**，下游 `masks_to_transition_npz` 用它做归一化）：
```json
{
  "t0_monotonic": 8821.39,
  "t0_wall": 1782552556.6,
  "start_iso": "2026-06-27T17:29:16",
  "mode": "sweep",
  "action_interval_s": 0.2,
  "settle_s": 0.19,
  "lo6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "hi6": [150.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "active_channel": 0,
  "note": "",
  "stop_iso": "2026-06-27T17:30:58",
  "frames": 501
}
```

- `hi6` = 每通道**操作上限**（kPa）：`masks_to_transition_npz` 按它把气压固定归一到 `[0,1]`（rest=0、full=`hi6`），保证跨序列一致（`c0=0.5` 永远 = 75 kPa）、气动单向无 OOD 负值。`hi6[ch]=0` 或缺失则回退到数据 max。
- `lo6` = 每通道下限（本例全 0 = 大气压 rest 态）。

---

## 4. 部署 / 连接注意事项

### 4.1 找串口名

| 系统 | 命令 / 位置 |
|---|---|
| Linux | `ls /dev/ttyUSB*` 或 `dmesg \| grep ttyUSB` |
| Windows | 设备管理器 →「端口 (COM 和 LPT)」，或 GUI 串口下拉 |
| macOS | `ls /dev/cu.usbserial*` |

> Linux 权限错：`sudo chmod 666 /dev/ttyUSB*`，或把用户加进 `dialout` 组（一劳永逸）。
> `group1` / `group2` 谁是谁由对方物理接线决定：连上一个、发小气压、看哪个阀动，标记一下。

### 4.2 ★ Modbus 每组独立连接（重要）

- **两组阀各自一条 RS485 串口 + 各自一个 `ModbusThread`**（`ModbusManager.serial_ports / modbus_threads` 按 `group_id` 独立）。
- 连接 / 断开都按组单独操作：`connect_group(gid)` / `disconnect_group(gid)`。**再点一次同一按钮 = 断开该组**，安全释放该组串口，不必关整个程序。
- `ValveController.set_pressures` **只写给已连接的组**：只连组1 → 只下发 ch0-2，ch3-5 不会被驱动（`all` 模式下没连的组那 3 路也保持 inactive）。
- 串口 `open` 可能阻塞 → 建议在后台线程调用（GUI 已包好）。
- 串口超时取 100 ms：9600 bps 下 8 字节响应约 8.3 ms 传输，100 ms 仅作"无响应"上限，收满预期字节即返回。
- 队列轮询 5 ms（压缩自原 50ms），命令下发更跟手、采样写入更及时。

### 4.3 配置持久化

- 串口名 / 波特率 / 从站地址 / 通道范围 / 间隔 / settle 存进 `real_capture/real_capture_config.ini`（**机器相关，已 gitignore**），下次自动回填。
- **换机器后第一次需手填一次串口**（`config.ini` 里的绝对路径若父目录在本机不存在会被忽略，不踩跨机器坑）。
- CLI 也可预填：`python main_capture.py --baudrate 9600 --slave 1 --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1`。

### 4.4 mock 任意组合（无硬件先验链路）

`--mock` = `--mock-cam --mock-valve --mock-ndi`，三个可任意组合混选：

```bash
python main_capture.py --mock-ndi --group1 /dev/ttyUSB0   # 真阀 + 假 NDI
python main_capture.py --mock-cam --mock-valve             # 假相机+假阀，真 NDI
```

mock 模式下：假相机合成随时间弯曲的剪影臂、假阀回放命令值、假 NDI 合成 XY 画圆 + Z 微动的末端轨迹——无硬件也能跑通 GUI + recorder + `capture_to_npz` 整条管线。

### 4.5 常见坑

| 现象 | 处理 |
|---|---|
| 串口连不上 | 确认对方波特率/从站地址；Linux 权限（§4.1）；别同时开其它串口监视器 |
| 自动驱动被拦 | Random/Sweep 前必须「连接 Modbus」 |
| NDI 失锁（quality NaN） | 记 `nan`，`--clean-nan` + `tip.npz` 插值补 |
| 退出未落盘 | 用窗口关闭（安全停所有线程 + 关 Modbus + 落盘），勿强杀进程 |
| 坐标轴出现浮点尾 | 已修（`_CleanAxis` 按间距取整）；若仍现，升级 `pyqtgraph` |

### 4.6 依赖

```bash
pip install -r real_capture/requirements.txt
```

- 全 mock：只需 `PyQt5 / pyqtgraph / numpy / opencv-python`。
- 真机额外：`pyserial`（阀）、`pyrealsense2`（相机）、`scikit-surgerynditracker` + `scipy`（NDI）。未装时 mock / 部分硬件模式仍可运行（`import` 都是可选的）。

---

## 5. 之后：出 npz + 训练

GUI「⚡ 生成 npz」一键完成（先 `tip.npz`，再 `capture_to_npz`），等价手跑：

```bash
# 1) NDI 末端 → tip.npz（recorder.build_ndi_tip_npz，失锁线性插值）
python -c "import sys; sys.path.insert(0,'real_capture'); from recorder import build_ndi_tip_npz as f; print(f('real_capture/data/raw/seq_XXXX'))"

# 2) 图像 + 气压 + 末端 → 仿真 schema .npz
python scripts/real/capture_to_npz.py \
  --view-dirs real_capture/data/raw/seq_XXXX/cam0 \
  --actions real_capture/data/raw/seq_XXXX/actions6.csv --actions-has-timestamps \
  --frame-times real_capture/data/raw/seq_XXXX/frame_times.txt \
  --ndi-tip real_capture/data/raw/seq_XXXX/tip.npz \
  --planar-lift --clean-nan --out data/real_seq/seq_XXXX.npz
```

后续 mask 分割 → 骨架化 → 清洗 → 训练 → 评估全流程见 [`workflow.md`](workflow.md)。

---

**一句话**：6 通道 Modbus 比例阀（2 组 × 3 路，4–20mA / 0–500kPa）驱动硅胶臂，RealSense 取图像，NDI Aurora 读末端；`real_capture/recorder.py` 的动作门控时钟（每 0.2 s 下发 → 等 0.19 s 稳定 → 抓帧 + NDI）把三路锁成同索引同时刻的 `cam0/*.png` + `actions6.csv` + `ndi.csv`。
