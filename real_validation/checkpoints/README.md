# 验证 App 的模型文件

默认四页 Hereditary 与旧 OpenLoop 使用不同合同，不能互换。

| 用途 | 文件 | 加载位置 |
| --- | --- | --- |
| 默认控制模型 | `hereditary_current/hereditary.npz` + 同名 `hereditary.json` | 第1页“加载模型” |
| 5 Hz / 10 Hz / 旧控制参考 | `candidates/<名称>.npz` + 同名 JSON | 第1页选择对应候选；[候选来源](candidates/README.md) |
| SAM2.1 Tiny 初始化 | `sam2/sam2.1_hiera_tiny.pt`，配合 `vendor/sam2/` | 第2页修正/高级设置中的 SAM 设置 |
| YOLO 分割候选（待接入） | 拟放 `perception/<模型版本>/`，权重与推理合同一起 | 尚无 GUI 入口；见 [部署方案](../YOLO_DEPLOYMENT_PLAN.md) |
| 旧 OpenLoop | `<名称>/best_model.pt`、`config.json`、`deploy_manifest.json` | 仅 `--legacy-openloop`；见 [旧指南](../GUI_GUIDE.md) |

Hereditary JSON 包含参数 hash、原生 `dt`、四维压力尺度、六腔映射和限制。复制/改名时 NPZ 与 JSON 一起处理，加载会核对 hash。不能修改文件名或 JSON 的 dt 来转换 5/10 Hz 模型。分割模型不控制压力，也不拥有控制 dt。

权重是运行资产，不提交到 Git；源码 checkout 不保证这些大文件存在。独立发行包由打包工具明确复制所选权重并登记 SHA256，保留旧候选，不依赖服务器路径。
