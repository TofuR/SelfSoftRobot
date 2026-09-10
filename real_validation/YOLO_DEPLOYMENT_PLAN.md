# real_validation 内置 YOLO 分割方法接入与随包部署方案

交付对象始终是 **同一个 real_validation 独立部署包**。YOLO 是 App 内部的分割方法，与 SAM2 一样从现有分割设置选择；所需权重、配置和运行依赖随验证 App 一起提供。用户启动同一个 App，在第二页直接自动提取并确认部署，不需要启动另一套 YOLO 程序或下载单独的 YOLO 部署包。

状态：**预先规划，尚未接入 GUI 或实时跟踪**。2026-09-11。当前 App 仍使用 SAM2/传统初始化及局部边缘反馈。服务器离线微调入口与数据分组见 [训练说明](../docs/experiments/robot_segmentation_student_20260911.md)。是否训练完成以对应 study 的 `status.json` / `COMPLETE` 为准。

## 最小运行流程

对使用者保持现有操作：**获取相机图像 → 模型输出机器人 mask → 中心线采样与 BASE/TIP 短边端点修正 → 检查并确认部署**。YOLO 替换初始化分割后端，先复用现有 `extract_initial_shape(..., supplied_mask=...)` 与确认流程；该函数当前把 supplied mask 的来源写为 sam2，接入时一并改为真实后端名称。

缩放、letterbox、BGR/RGB、输入数值尺度以及输出裁剪/还原由适配器负责。若使用 Ultralytics 高层 API 已完成这些操作，则直接使用其原图坐标结果，不再做一次逆变换。直接使用 ONNX Runtime 时按导出合同解码，只执行模型实际需要的步骤。已经导出为端到端检测的图不能盲目再加 NMS。正常 YOLO 路径不串联 SAM、GrabCut 或额外迭代优化。

部署配准是初始化的一次性几何与状态拟合，不在每帧重做。规划前持续跟随外力摆动暂不实施，未来另行设计观测状态与模型状态的关系。

## 目录与文件

以下为拟采用结构，**不是当前可直接加载的模型合同**：

```text
<real_validation 完整部署包>/
  START_HERE.md
  PACKAGE_MANIFEST.json
  real_validation/
    main.py / run_gui.bat / ...             # 完整 App 与感知适配器
    README.md / HEREDITARY_GUIDE.md / HEREDITARY_GUIDE.html
    YOLO_DEPLOYMENT_PLAN.md
    requirements*.txt
    config/hardware.example.json
    checkpoints/
      hereditary_current/hereditary.npz    # 控制模型，与分割模型独立
      hereditary_current/hereditary.json
      candidates/                          # 原生 5/10 Hz 及旧控制参考
      perception/soft_arm_yolo26n_<版本>/
        best.pt                            # 若发布 PyTorch/Ultralytics 后端
        model.onnx                         # 若发布 CPU ONNX 后端
        model.onnx.data                    # 仅在实际导出引用外部张量时包含
        inference.json                     # 与权重绑定的推理/坐标合同
        provenance.json                    # 训练来源与选择指标
        metrics.json / latency.json        # 实测结果与测试条件
        labels.json                        # 类别 0 = soft_arm
    validation_samples/segmentation/
      frame_*.png / reference_mask_*.png
      expected.json                        # 样例来源、坐标、检查容差
  third_party_notices/                      # 组件与模型许可证、来源
  locks/                                   # 在目标平台验收过的精确依赖版本
  wheels/                                  # 仅离线安装版需要
```

第一版优先比较 `.pt` 与 CPU `.onnx`；只启用已经验收的后端。一个发行包不必携带所有 n/s 候选：选择满足需求的一个为默认，另一权重按对照需要附带。若保留 SAM 作为人工修正备用，则继续包含现有 SAM2 源码、权重、依赖及许可证，并在清单声明；若决定不带 SAM，则不能保留指向缺失默认权重的设置。

`.pt` 与 `.onnx` 均为分割模型，不得放入 Hereditary 控制候选列表直接加载；分割模型没有控制 `dt`、六腔映射或压力单位。同一个分割模型可服务于原生 5/10 Hz 控制模型。

## 每项保存什么

| 项目 | 必须记录的内容 | 原因 |
| --- | --- | --- |
| 推理权重 | 已选最佳权重；ONNX 及它实际引用的全部外部文件；各自 SHA256 | 防止漏文件、混用候选或只有主 ONNX 文件 |
| 推理合同 | 后端/库版本、任务、输入输出名称/形状/类型、图像尺寸、batch、动态轴、opset、精度、类别、置信阈值、是否已有 NMS、mask 阈值及重建方式 | 不靠文件名猜测导出输出结构 |
| 坐标约定 | 原始 BGR 帧尺寸、颜色转换、缩放/填充/裁剪和原图还原规则；mask 输出所在坐标系 | 避免把 640 输入或历史 300×300 裁剪坐标直接画到相机上 |
| 形状后处理 | 节点数由控制模型读取，BASE→TIP 次序、短边中心约定、筛选规则、失败判据 | 无结果、多个候选或断裂时报告失败/歧义，不默认选最大碎片作全臂 |
| 来源 | 官方预训练模型/版本/hash、训练脚本版本、配置、数据清单 hash、序列分组、最佳权重选择指标、排除样例规则 | SAM 伪标签指标不能写成人工真值或新 D415 泛化结果 |
| 验证 | mask 指标、漏检/误选情况、中心线与短边检查，以及测试电脑/后端/精度/线程数/输入尺寸 | 对软臂细边界的质量检查不能只看通用 COCO 数值 |
| 时延 | 加载与预热单列；内存图像到 mask、mask 到节点、合计的 P50/P95/max；样本数；相机等帧另外记录 | 不重复计算 API 已包含的后处理，也不把模型前向时间当整个结果可用时间 |
| 最小样例 | 少量全臂、近景、易混支架、无臂/不完整臂图像与预期结果；标明 mask 是否伪标签 | 不带全量训练数据，也能离线检查坐标和失败处理 |
| 依赖和通知 | 在目标 OS/Python/CPU 或 CUDA 上验证的版本；Ultralytics 和模型的实际适用许可、来源 | 不能把 Linux 环境或 TensorRT engine 直接视为 Windows 通用产物 |

现在的 `requirements-yolo.txt` 固定已用于离线候选环境的 Ultralytics / torchvision / CPU ONNX Runtime 版本。它用于提前安装或离线检查，不代表 GUI 已集成，也不是已验收的 Windows 全依赖锁文件。ONNX 导出器 `onnx` 是服务器导出依赖，PC 仅推理时无需专门安装。TensorRT engine 如后续采用，必须记录并匹配目标 GPU、TensorRT/CUDA 版本，不作为第一版通用必需文件。

## 运行记录与不应带入的内容

接入后，每次初始化保存分割后端/模型 hash、实际预处理参数、原图/mask/中心线、端点修正来源和分段耗时。执行日志继续关联控制模型、配准、压力历史及已有逐步反馈记录；后续若启用逐帧 YOLO，再明确记录哪些帧真的跑了 YOLO。没有运行的模型不生成虚假的耗时或预测记录。

不随通用包携带全部训练图片、标签缓存、optimizer 状态、整个 conda/venv、旧试次目录或服务器绝对路径依赖。设备电脑自己的 `config/hardware.json`、端口、相机序列号、实验压力与上次输出不作为其他机器的默认状态。离线安装版 wheelhouse 要在目标 Windows/Python 架构上准备；普通联网版用已核对的 requirements 即可。

## 发布前需要完成的接入工作

1. 候选训练与独立序列评估完成；检查薄边、短边、支架混淆和实际 D415 样例，按质量与时延选择后端。
2. 增加一个统一的分割适配器，返回原图尺寸 mask、来源和耗时；复用现有中心线/确认逻辑，不新增必经页面或手动画框步骤。
3. 在现有“修正 / 高级设置”中配置分割模型；初始化正常只保留自动提取、检查和确认。无结果时显示原因，保留人工修正，不能自动宣称配准成功。
4. 扩展打包器的感知模型清单及 hash/路径检查。当前 `--extra` 虽能复制任意附件，但**不会验证 YOLO 合同或使 GUI 获得 YOLO 能力**，不能仅塞入 `.pt` 就标注“YOLO 可用”。
5. 在无仓库源码依赖的解压目录检查加载、原图坐标、中心线端点、CPU 路径、明确选择的 GPU 路径和旧四页流程。发布包含 YOLO 方法的新版 real_validation 完整 ZIP，保留此前 App 包及控制权重。验收以用户在同一 App 中选择 YOLO、自动提取并完成部署预热为准，不能只验收外部推理脚本。

本轮仅更新依赖/文档与上述方案；实时跟随、YOLO GUI 接入和最终分割发行权重仍待后续完成。
