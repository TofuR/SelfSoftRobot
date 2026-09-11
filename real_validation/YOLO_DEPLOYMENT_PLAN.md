# real_validation 的 YOLO 初始化分割

2026-09-11：n/s 微调、测试和导出均已完成，YOLO 已接入四页验证 App 的第二页。分割得到的完整形状用于人工检查、部署配准和状态预热；执行中的部分观测反馈继续使用局部边缘观察器。

## 使用

1. 解压新版 App，双击包根 `install.bat` 安装运行环境。
2. 双击包根 `run.bat`，加载控制模型并连接设备。
3. 第二页点击 **自动提取当前完整形状**。默认 YOLO n 版整图推理，绿色是冻结帧 mask，黄线是中心线；检查 BASE/TIP 后点击 **确认形状并部署预热**。
4. 要换 s 版或 CPU/CUDA，展开 **修正形状 / 高级设置 → 分割设置…**。分割方法下拉框可切回 SAM2 或传统分割。完整发行包的安装脚本已包含 SAM2 依赖。

YOLO 没有提示点接口；SAM 正/负提示按钮只在 SAM2 模式显示。无检测、多个机器人候选或不完整轮廓会明确提示，可手绘或切换 SAM2。手绘后的“结合图像微调草稿”也使用当前选择的分割方法。近景和背景改变后的结果仍需检查。

## 目录：源码、训练记录和部署权重

根目录 `sam2/sam2_src/` 是 Meta 上游源码 checkout；YOLO 上游实现由 `ultralytics` Python 依赖提供。`workspace/runs/training/` 是训练图像、日志和权重的归档位置，两者用途不同。

部署所需文件统一由 App 相对路径寻找，用户不必进入训练目录：

```text
real_validation/
  perception/yolo_initial.py           # 本项目的推理适配器
  checkpoints/
    hereditary_current/               # 默认控制模型，npz + json
    candidates/                       # 原生 5/10 Hz 控制候选
    sam2/sam2.1_hiera_tiny.pt           # SAM2 备用权重
    perception/
      yolo26n-seg_20260911_001/         # 默认，更快
      yolo26s-seg_20260911_001/         # 可选，mask 指标稍高
        best.pt                       # GUI 实际加载
        inference.json                # 文件 hash 与原图坐标合同
  vendor/sam2/                        # 完整包内的 SAM2 源码
  requirements.txt                    # 单一运行依赖清单
  PACKAGE_MANIFEST.json               # 文件 hash、模型来源与选择指标
  licenses/Ultralytics-LICENSE.txt
```

两个分割模型都可服务于 5/10 Hz 控制模型，没有控制 dt 或腔道映射。服务器保留完整训练 study，以及 `workspace/models/deployment/robot_yolo26seg_20260911_001/` 发布副本；`real_validation/checkpoints/perception/` 是便于本机和独立包加载的发布副本，不包含训练数据。旧权重和旧独立包保留。

## 完成的训练与测量

来源：`robot_yolo26seg_20260911_001`，9 条同平台序列，训练 6404、验证 1097、测试 1405 张；监督是处理后的伪 mask。两个模型上限均 100 epoch，patience=25，n/s 分别在 59/57 epoch 早停。最佳 checkpoint 依据 **Ultralytics 的 mask fitness + box fitness 验证指标之和**。原训练进程部分输出误写 mask-only，发布元数据按实际代码修正。

| 候选 | 测试 mask mAP50–95 | CPU 图像→mask 中位 / P95 | CUDA 图像→mask 中位 / P95 |
| --- | --- | --- | --- |
| YOLO26n-seg | 0.7758 | 26.59 / 27.84 ms | 9.66 / 9.75 ms |
| YOLO26s-seg | 0.7899 | 49.33 / 54.17 ms | 13.36 / 13.93 ms |

服务器、640 输入、batch=1、20 张暖启动图像、CPU 4 线程。计时含 API 预处理、模型与原图 mask 重建，不含中心线、相机、通信和首次加载。默认 n 版考虑速度，s 版作为质量对照；伪标签分数和服务器时间不代表新 D415 现场精度或部署电脑耗时。

## 运行与发布合同

适配器延迟加载本地 `.pt`，首次校验权重 hash，缓存选定模型。Ultralytics 接收原始 uint8 BGR 图像，自行 letterbox、颜色转换、归一化，`retina_masks=True` 返回原图尺寸 mask；适配器检查尺寸并拒绝多候选，再复用中心线和短边处理。正常路径一次 YOLO 分割即可。

`initial_shape_draft` 事件保存原图、mask、中心线、后端、权重 hash、置信度、设备、加载/图像到 mask/mask 到节点/总时延；失败草稿也保留分割诊断。首次调用包含框架初始化，可能数秒；后续缓存调用可更快。

发布入口：

```bash
python scripts/evaluation/export_robot_segmentation.py --study workspace/runs/training/robot_yolo26seg_20260911_001 --out <新的发布目录>
python -m real_validation.tools.package_hereditary --bundle <控制模型.npz> --out <新包.zip> --perception-dir <发布目录> --yolo-license <Ultralytics-LICENSE> [原有候选、SAM2、指南参数]
```

导出拒绝覆盖；打包验证候选合同/hash，清单单列 perception_models 和模型来源；运行包只复制 `.pt` 和 `inference.json`，ONNX、测试指标与样例保留在服务器归档。上游依赖安装在 Python 环境中，训练集、整个 venv 和日志不随包复制。Windows 安装脚本创建包根 `.venv`，requirements 是联网安装清单；目标平台运行结果需在现场确认。
