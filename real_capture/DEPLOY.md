# real_capture 部署到新机器 / 接入六通道阀系统

> 一句话：`modbus_manager.py` 就是和阀硬件的完整契约。**不需要对方那套 UI/采集程序**——
> 把 2 个 RS485 串口接到本程序即可，阀响应的是同一套 Modbus 指令，谁发都一样。

---

## 0. 这是什么

`real_capture/` 是一套**自包含**的实物采集程序：

- 用 `modbus_manager.py` 控制 **6 通道电流型气压阀**（2 组 Modbus RTU × 3 路，0–500 kPa，4–20 mA）；
- 同步采集 **NDI Aurora 末端位姿** + **一台或多台 RealSense 图像**；
- **动作门控**落盘：动作每 0.2 s 下发一次，等 0.19 s 软臂稳定后抓全部相机帧 + NDI 末端，同一拍分别写入 `camN/`；
- 输出直接对接 `scripts/real/capture_to_npz.py`（→ 仿真 schema `.npz`，可训练）。

---

## 1. 带哪些文件

纯采集**只需整个 `real_capture/` 文件夹**（自包含，不依赖仓库的 `src/`）：

| 文件 | 作用 |
|---|---|
| `main_capture.py` | GUI 入口（**界面也在这里，纯代码构建，无 `.ui` 文件**） |
| `modbus_manager.py` | 与阀硬件的契约（原样复用） |
| `valve_control.py` | 6 通道控制封装 + 自动驱动（random/sweep） |
| `recorder.py` | 动作门控同步采集核心 |
| `hardware_threads.py` | NDI 线程（+ Mock） |
| `nditracker.py` / `realsense_cam.py` | NDI / 相机封装（含 mock） |
| `requirements.txt` | 依赖 |

> 只有后面想点 GUI 里「生成 npz」按钮（转训练数据）时，才需要仓库根的
> `scripts/real/capture_to_npz.py` + `src/data/real/`。**纯采集不需要。**

---

## 2. 装依赖

```bash
pip install -r real_capture/requirements.txt
```

- **全 mock**（无任何硬件，先验 GUI + 链路）：只需 `PyQt5 / pyqtgraph / numpy / opencv-python`。
- **真机**额外需要：`pyserial`（阀）、`pyrealsense2`（相机）、`scikit-surgerynditracker`（NDI）。

---

## 3. 硬件接线（问对方确认）

6 通道 = **2 组 Modbus RTU × 每组 3 个阀**：

- 每组 1 个 **USB-RS485 转换器**（小 dongle）→ 串联 3 个阀。
- 插上电脑后出现 **2 个新串口**。
- 可选：NDI Aurora（没有就 `--mock-ndi`）、RealSense（没有就 `--mock-cam`）。

---

## 4. 找串口名

| 系统 | 命令 / 位置 |
|---|---|
| Linux | `ls /dev/ttyUSB*` 或 `dmesg \| grep ttyUSB` |
| Windows | 设备管理器 → 「端口 (COM 和 LPT)」，或直接看 GUI 串口下拉 |
| macOS | `ls /dev/cu.usbserial*` |

> Linux 下若报权限错：`sudo chmod 666 /dev/ttyUSB*`，或把用户加进 `dialout` 组（一劳永逸）。

---

## 5. ⭐ 波特率 / 从站地址 填在哪里

**在 GUI 左上角「Modbus 连接」分组里**，点「连接 Modbus」**之前**填好：

| GUI 控件 | 字段 | 默认 | 说明 |
|---|---|---|---|
| `波特` | baudrate | `9600` | 问对方或翻他们 `config.ini` |
| `从站` | slave_addr | `1` | 同上 |
| `组1串口` | group1 | `COM3` | 第 1 组 RS485 串口名 |
| `组2串口` | group2 | `COM46` | 第 2 组 RS485 串口名 |

也可 CLI 预填（等价）：

```bash
python main_capture.py --baudrate 9600 --slave 1 \
                       --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1
```

> 这些值会存进 `real_capture_config.ini`（**机器相关，已 gitignore**），下次自动恢复。
> 换机器后第一次需手填一次。

---

## 6. 启动

```bash
cd real_capture

# 1) 全 mock（无硬件，先验证 GUI + 整条链路）
python main_capture.py --mock

# 2) 真阀 + 假 NDI（最常用的实物调试起点）
python main_capture.py --mock-ndi --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1

# 3) 真机全用（两组 Modbus + NDI + 相机）
python main_capture.py --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1 --ndi /dev/ttyUSB2

# 4) 多相机：序列号按设备管理器/RealSense Viewer 查询，逗号分隔
python main_capture.py --camera-count 2 --camera-serials 123456789,987654321 \
                       --group1 /dev/ttyUSB0 --group2 /dev/ttyUSB1 --ndi /dev/ttyUSB2
```

`--mock` = `--mock-cam --mock-valve --mock-ndi`，三个可任意组合混选。

---

## 7. 单通道 vs 全部（「主通道」下拉）

GUI「主通道」下拉有 `ch0..ch5` + `全部 (all)`——**这是同一个功能（配 min/max）的两种实现**：

| 模式 | 行为 | 气压图 |
|---|---|---|
| **单通道 chN**（推荐起步） | 其余 5 路 `min/max/target` **自动归零并锁定**（灰显不可改，防误操作）；chN 设默认范围 `0–200 kPa`、target=0 | 只画 chN **1 条线** |
| **全部 (all)** | 放开**已连接组**的通道 `min/max` 可改（没连的组那几路仍锁定） | 画**已连接组**的曲线 |

- **Modbus↔ch 联动**：`ch0-2` 属组1、`ch3-5` 属组2。只连组1→只能用 ch0-2（ch3-5 灰锁且不驱动）；只连组2→只能用 ch3-5；两组都连→6 路全可用。**all 模式下没连的组那 3 路也保持 inactive**（不会被驱动）。
- **min/max/target 录制中改也实时生效**（random/sweep 下一拍即用新范围）。
- 单通道与 `all` 模式统一写 `actions6.csv`；未使用通道保持为 0，不再生成旧 `pressure.csv`。

---

## 8. 采集流程

1. 「Modbus 连接」填组1/组2 串口 + 波特/从站 → 点「**组1 连接**」（默认只连组1 即可控制 ch0-2）；需要 ch3-5 再点「**组2 连接**」。**再点一次同一按钮 = 断开该组**（安全释放串口，不必关程序）。
2. 「主通道」选 `ch0`（单通道起步）→ 确认 min/max（默认 `0–200`）。未连的组对应的通道会自动灰锁。
3. 在「**相机**」分组选择相机数量和序列号；空序列号表示自动选择当前设备。点击「应用/重连」后，右侧可选择单路预览或平铺全部视角。
4. （可选）点「**连接 NDI**」（再点 = 断开，安全释放 Aurora）。
5. 填「保存目录」、选「模式」（手动 / 随机游走 / 往返扫描 / Replay）、「动作间隔」`0.2` s、「稳定等待」`0.19` s。
6. 「▶ 开始采集」→ 采够后「■ 停止采集」。
7. 后处理：「⚡ 生成 npz」或「📋 导出汇总 CSV」。多相机会自动传入全部 `camN` 目录并使用三角化。

---

## 9. 输出（每个序列目录，对齐 capture_to_npz schema）

| 文件 | 内容 | 下游用途 |
|---|---|---|
| `cam0/00000.png ... camN/00000.png` | 每个相机一个目录；同一编号为同步拍 | `--view-dirs` |
| `frame_times.txt` | 每帧一行 相对秒 | `--frame-times` |
| `actions6.csv` | `t_sec, c0..c5`（**首行表头**） | `--actions --actions-has-timestamps` |
| `ndi.csv` | `t_sec + ndi0_* ... ndiN_*`（首行表头；失锁写 `nan`） | → `tip.npz` → `--ndi-tip` |
| `commands.csv` | 命令时间、ACK、最终命令、分组通信状态 | 通信 QC / 复现 |
| `samples.csv` | `t_grab`、总体/逐相机 `frame_age`、各 NDI age/quality | 数据质量筛选 |
| `meta.json` / `summary.csv` | 运行元信息（含相机数量/序列号）/ 按帧对齐汇总 | 人眼核对 |

---

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| 串口连不上 | 确认对方波特率/从站地址；Linux 权限（见 §4） |
| `group1`/`group2` 谁是谁 | 对方物理接线决定；连上一个、发小气压、看哪个阀动，标记一下 |
| 换机器后配置丢了 | 正常，`real_capture_config.ini` 不跨机器；重填一次串口即可 |
| 没有 NDI / 相机 | `--mock-ndi` / `--mock-cam`，其余照常用真阀 |
| 坐标轴出现 `0.30000000000000004` | 已修（`_CleanAxis` 按间距取整）；若仍现，升级 `pyqtgraph` |
