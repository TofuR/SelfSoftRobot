# Real Robot Validation Workbench

> **GUI 使用指南见 [`GUI_GUIDE.md`](GUI_GUIDE.md)** —— 五页功能、使用顺序、标准操作流程、当前能力边界。

第一版提供独立于采集 GUI 的验证工作台基础：

- 不可变 model/anchor/scene/safety/plan 数据契约；
- `IDLE → READY → ARMED → EXECUTING` 安全状态机；
- action dimension、六通道映射、压力、速率和 `K_safe` preflight；
- 控制观测与隐藏评价流的 observation policy；
- Mock ACK、错误注入、Abort 后归零与 `execution.csv`；
- 从 transition NPZ 建立带完整 H 历史的离线 anchor；
- 受压力/速率约束的 OpenLoop shooting 与逐步轨迹/动作预览；
- Qt 真阀线程桥接、只读 run replay 和基础离线评价；
- 五阶段 GUI 骨架。

## 搬到 PC

直接复制整个 `real_validation/` 目录，不需要复制项目的 `src/`、`scripts/`、
`real_capture/`、`config/` 或 `train_log/`。在 PC 上安装本目录依赖：

```bash
python -m pip install -r requirements.txt
```

需要接入 RealSense、串口阀和 NDI 的 PC 再安装：

```bash
python -m pip install -r requirements-hardware.txt
```

在线感知（分割 / 骨架 / 配准）另需：

```bash
python -m pip install -r requirements-perception.txt
```

把 checkpoint 和它所属实验的 `config.json` 放到 `checkpoints/current/`。当前
`config.json` 占位已经对应服务器 `exp_20260714_8`，因此只需复制该实验的
`best_model.pt`；更换其他模型时必须把二者一起更换。

启动 GUI：

```bash
python main_validation.py
```

Windows 也可以双击 `run_gui.bat`。所有默认路径都由 `main_validation.py` 所在目录
计算，不依赖启动时的工作目录。

## 自检

本目录的单元测试住在仓库的 `tests/`（**不随本目录拷贝到 PC**）。在 PC 上只能做
运行时自检：

```bash
python -c "import real_validation; print('contracts ok')"
```

完整测试（20 个契约测试 + 感知 parity + import 卫生）在仓库根运行：

```bash
python -m unittest tests.test_real_validation_core tests.test_perception_parity tests.test_import_hygiene -v
```

当前 GUI 中的执行明确标记为 Mock。真阀连接面板、在线骨架提取和完整交互式 Scene
Editor 接入完成前，不得用此入口控制实机。
