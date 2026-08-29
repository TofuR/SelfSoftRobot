# Real Robot Validation Workbench

> **GUI 使用指南见 [`GUI_GUIDE.md`](GUI_GUIDE.md)** —— 五页功能、使用顺序、标准操作流程、当前能力边界。

该目录提供独立于采集 GUI 的模型部署与真实控制工作台：

- 不可变 model/anchor/scene/safety/plan 数据契约；
- `IDLE → READY → ARMED → EXECUTING` 安全状态机；
- action dimension、六通道映射、压力、速率和 `K_safe` preflight；
- 控制观测与隐藏评价流的 observation policy；
- Mock ACK、错误注入、Abort 后归零与 `execution.csv`；
- 从 transition NPZ 建立带完整 H 历史的离线 anchor；
- 受压力/速率约束的 OpenLoop shooting 与逐步轨迹/动作预览；
- 当前相机的实验级 ROI、参考背景、在线分割和 15 节点整臂中心线；
- 相机像素、ROI 局部像素与机器人毫米坐标的可追溯正反变换；
- 执行后真实形态目标残差、前向预测误差和下一窗口 Anchor；
- Qt 真阀线程桥接、只读 run replay 和基础离线评价；
- 五阶段 GUI 骨架。

## 搬到 PC

直接复制整个 `real_validation/` 目录 —— **完全自包含**,不需要项目的 `src/`、
`scripts/`、`real_capture/`、`config/` 或 `train_log/` 等任何其它目录并排部署。
在 PC 上安装本目录依赖：

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

把同一次训练试次的 checkpoint、`config.json` 和 `deploy_manifest.json` 放到
`checkpoints/<model_name>/`。部署清单声明动作压力尺度、通道来源、训练时基、毫米状态坐标、
`K_safe` 和位移统计；加载时会校验 checkpoint 哈希。

离线锚定用的 transition NPZ 放 `data/npz/`(示例 15 节点数据已内置;GUI_GUIDE
§2.2 有『从 NPZ 建 Anchor』的完整操作入门)。

启动 GUI：

```bash
python -m real_validation.main
```

Windows 也可以双击 `run_gui.bat`。所有默认路径都由 `real_validation/` 包所在目录
计算，不依赖启动时的工作目录。

## 自检

本目录的单元测试住在仓库的 `tests/`（**不随本目录拷贝到 PC**）。在 PC 上只能做
运行时自检：

```bash
python -c "import real_validation; print('contracts ok')"
```

完整测试在仓库根运行:

```bash
python -m unittest discover -s tests -v
```

GUI 根据硬件 profile 显式显示 Mock 或 Real 执行。真实计划需要部署合同、当前相机配置、
Anchor、Scene 和 Safety 全部通过 Preflight，再经操作员 Arm 放行。
