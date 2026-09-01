# SelfSoftRobot 仓库整理与重构设计

> 状态：Active migration（目标设计；实际落地范围以“实施状态”表为准）
> 日期：2026-08-31
> 范围：仓库结构、数据谱系、训练共识、实验产物和文档治理
> 原始非目标：首轮设计阶段不删除、不移动现有数据；2026-09-01 经明确授权后进入历史资产物理迁移，仍不覆盖历史训练、不改变模型科学结论

## 实施状态（2026-09-01）

| 阶段 | 状态 | 已落地 | 尚未完成 |
|---|---|---|---|
| Phase 0 清单 | 完成 | 历史 raw/intermediate/processed/training/analysis 只读 inventory | 定期刷新策略 |
| Phase 1 路径与 manifest | 完成 | `ProjectPaths`、URI、schema、原子写入、路径配置模板 | 历史 manifest 批量升级 |
| Phase 2 数据双读单写 | 完成 | `real_capture`、预处理、dataset/fixture selector、组合/清洗、SAM2 与 QC 均统一读写 workspace | 后续新增脚本持续遵守同一合同 |
| Phase 3 训练合同 | 部分完成 | 新 run 根、轻量 run manifest v2、旧 checkpoint 双读、engine 验证/选择/早停循环及 transition adapter 已落地；GT/OpenLoop 排序等价、真实流水线切换及切换后完整短跑已通过 | 用 canonical dataset 完成首个 formal run 与 frozen test 验收 |
| Phase 4 历史导入 | 部分完成 | 67 项资产已原子迁入 workspace；14 个 dataset 均有历史/观察型 manifest，104 个训练 run 已索引，4 个真实主线试次已有 `legacy_run_manifest.json` | 严格 v2 升级、归档 run manifest 和 15 条缺失数据引用继续审计 |
| Phase 5 文档治理 | 部分完成 | 唯一导航及数据划分、训练评价、试次、证据标准 | workflow 合并、旧文档逐份裁决、front matter lint |
| Phase 6 物理清理 | 部分完成 | 经授权并完成 GT/OpenLoop 2+2 epoch 主线短跑后，67 个旧兼容链接及空旧根已移除 | 缓存、测试展示产物和废弃工作树另行逐类清理 |

本表是实现状态，不替代下文的目标设计。提交历史按功能分层保存；任何尚未完成项都不能从目标描述推断为已实现。

## 0. 结论先行

本项目不适合通过一次“大搬家”解决混乱。当前问题的根因不是目录名称不够整齐，而是源代码、原始数据、可重建中间产物、数据集发布版、训练试次、评价结果和知识文档缺少统一的身份与边界。建议采用以下四项核心决策：

1. **建立一个可配置的运行工作区 `workspace/`，所有非源码产物通过同一个路径注册器访问。** 默认可以位于仓库内并被 git 忽略，也可以通过本地配置映射到大容量磁盘。业务代码不得再自行拼接 `real_capture/data/...`、`data/real_seq/...`、`train_log/...` 等路径。
2. **用不可变 manifest 串起数据谱系。** 原始采集序列、处理中间版本、可训练数据集和实验试次分别拥有稳定 ID；目录名只承担定位，不再承担全部语义。
3. **把训练规则拆成三层：项目强制协议、模型声明、具体试次配置。** 数据划分、选择指标、早停、checkpoint 和测试集使用遵守共同合同；GTObserved、OpenLoop、Hereditary、渲染/多阶段模型仍可声明各自不同需求。
4. **文档按“事实类型”治理，而不是按产生工具或临时任务堆放。** 每个主题只有一个权威入口；过时内容保留但明确 `superseded` 或归档，实验过程记录与长期共识分开。

迁移采用“**先注册、后双读、再单写、最后清理旧入口**”的方式。历史路径在完成校验前保持只读兼容，不批量重命名历史试次，也不复制大体积原始图像来制造第二份真相。2026-09-01 的实际迁移使用同文件系统原子 rename；主线短跑通过后已移除旧入口。

---

## 1. 当前项目的实际组成

### 1.1 领域主线

项目已经从早期 PyElastica/神经场仿真逐步转向真实软臂的视觉自建模和控制，当前主要链路是：

```text
真实采集（RealSense / valve ACK / NDI）
  -> 分割、裁剪、骨架化、坐标与动作合同
  -> GTObserved / OpenLoop / Hereditary 等状态模型
  -> 定量评价、视野诊断、规划
  -> real_validation 安全执行与独立观测评价
```

其中 `real_capture` 是采集应用，`real_validation` 是部署验证工作台；二者不应分别拥有一套独立的数据事实。NDI 应继续保持为独立评价信号，不进入模型或规划器输入。

### 1.2 2026-08-31 目录快照

以下数字用于确定重构优先级，不是清理目标：

| 当前目录 | 约占空间 | 当前角色 | 主要问题 |
|---|---:|---|---|
| `real_capture/` | 19 GB | 采集应用 + 原始/中间数据 | 应用代码拥有了项目最大的数据根 |
| `train_log/` | 4.0 GB | 模型权重、日志、部分评价 | 新旧模型布局不同；单阶段与完整流水线语义不同 |
| `output/` | 1.1 GB | 分析、图片、报告、临时验证 | 可再生结果与需保留的正式报告混放 |
| `.claude/` | 725 MB | 工具工作树 | 非领域资产；多个工作树复制仓库内容 |
| `sam2/` | 617 MB | 本地脚本、第三方源码、checkpoint、mask | 工具代码、第三方代码、模型权重和数据产物混放 |
| `data/` | 166 MB | 仿真数据和真实训练 NPZ | 真实数据版本信息被编码进长目录名 |
| `docs/` | 119 MB | 193 个已跟踪文档及本地参考材料 | 长期共识、状态快照、论文草稿、外部仓库混放 |
| `real_validation/data/` | 828 KB | 离线锚定 NPZ 副本 | 与正式 processed dataset 形成重复入口 |

仓库内还存在大量 `__pycache__`、根目录遗留 `*.pyc`、`tests/` 下的 GIF 和根目录图片。这些不一定占主要空间，却会持续模糊“源码”和“运行产物”的边界。

### 1.3 已确认的路径分裂

同一真实序列目前至少跨越以下位置：

```text
real_capture/data/raw/<seq>/                 # 不可重建的采集事实
real_capture/data/derived/<seq>/             # crop、候选、QC、部分 mask
sam2/masks/<seq>_full/                       # SAM2 mask
data/real_seq/<seq>_<processing-tags>/       # 可训练 train/val NPZ
real_validation/data/npz/                    # 离线锚定副本
train_log/...                                # 训练和部分评价
output/...                                   # 额外分析和可视化
real_validation/runs/run_*/                  # 真实验证运行
```

这种布局造成三个直接后果：

- 代码必须知道某种处理由哪个工具产生，才能猜出路径；
- `seq_..._n15_sam2_robot_mm` 这类目录名逐渐承担 manifest 的职责，却仍不能完整描述参数、源码版本和父数据；
- 删除、替换或比较某个版本时，很难可靠回答“它由什么生成、被哪些训练使用、还有没有唯一副本”。

### 1.4 已确认的训练语义分裂

当前 `UnifiedTrainer`：

- 只创建训练 DataLoader；
- 用训练集 epoch 平均 loss 更新 `best_model.pt`；
- `ReduceLROnPlateau` 同样观察训练 loss；
- 没有统一的验证循环和早停。

完整真实数据训练脚本另行启动 `watch_best_checkpoint.py`，每隔若干 epoch 在 `val/` 上计算全节点平均误差并保存 `best_eval_model.pt`。因此：

- 直接运行 `train_transition.py` 与运行 `train_real_transition.sh` 的“best”含义不同；
- 验证集选择只在部分流水线存在；
- `test` 集还没有形成统一路径或使用合同；
- “scheduler patience”和“early stopping patience”目前容易被混为一谈，事实上前者只控制降学习率，后者尚未统一实现。

### 1.5 已确认的文档问题

- 根 `Readme.md` 只有标题，真正导航在 `docs/README.md`；
- `docs/README.md` 自称最后更新于 2026-07-20，却没有覆盖 8 月新增的训练、验证和 Hereditary 文档；
- `docs/HANDOFF.md` 的标称更新时间、分支和“当前最佳实验”已经与后续实验并存，动态状态与长期知识没有分离；
- `docs/paper/` 是一套论文结构/草稿，`docs/papers/` 同时容纳文献笔记、相关工作、论文素材和另一批草稿，目录名无法表达权威性；
- `docs/real_data/` 中通用工作流、单序列处理记录、部署说明和基准报告并列，读者不容易知道何者可复用；
- `docs/ref/` 是被忽略的外部工程/旧仓库集合，甚至包含嵌套 `.git`，它不是文档；
- 同一结论会同时出现在 HANDOFF、status、workflow、deployment 和序列记录中，修改一个位置无法保证其他位置同步。

---

## 2. 整理原则

### P1. 一份事实，一个权威拥有者

- 原始相机帧、动作 ACK 和 NDI 原始记录只属于 raw sequence；
- 处理参数和派生关系只属于 dataset manifest；
- 训练参数、代码版本和选择结果只属于 run manifest；
- 项目当前状态只属于一个 status 页面；其他文档链接到它，不复制数字。

### P2. 路径不是身份

路径可迁移，`sequence_id`、`dataset_id`、`run_id` 和内容 hash 才用于引用。文档允许展示人类可读路径，但机器合同不得只依赖路径字符串。

### P3. raw 不可变，derived 可重建，release 不可覆盖

- raw：采集结束后只允许补充审计元数据，不修改帧和控制记录；
- intermediate：可删除重建，但每个 recipe 版本写到新目录；
- processed release：通过 QC 后冻结；任何参数变化生成新 `dataset_id`；
- run：一次启动一个新目录，禁止覆盖已有 log/checkpoint。

### P4. 共同协议只统一“可比较性”，不强迫模型同构

所有模型必须声明数据、选择指标和产物；但不同模型可以使用不同阶段、损失、验证频率、早停策略和诊断指标。共识的目标是让差异显式，而不是消灭差异。

### P5. 先兼容再迁移

先让新代码通过统一注册器读旧目录；新产物只写新目录；验证引用和 hash 后，再把旧入口改成只读别名。任何删除另行提出，不属于本设计的自动步骤。

### P6. 科学证据和工程证据分开

- 单序列连续 80/20 可以证明管线能运行，但不能证明跨序列泛化；
- recorded-GT 对比是前向模型误差，不等于真实控制成功；
- Mock ACK 是执行管线证据，不等于物理执行证据；
- 真机控制结论必须关联真实 run、执行后观测和目标误差。

---

## 3. 目标仓库结构

```text
SelfSoftRobot/
├── Readme.md                         # 极短项目入口，链接权威导航
├── pyproject.toml                    # 后续统一包、测试和工具配置
├── configs/
│   ├── paths.example.toml            # 可提交的路径示例
│   ├── data/                         # 数据处理 recipe
│   ├── training/                     # 训练协议/模型默认值
│   ├── deployment/                   # 部署合同模板
│   └── local.toml                    # 本机覆盖，gitignore
├── src/                              # 可复用领域库
│   ├── data/
│   ├── perception/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── planning/
│   ├── hardware/
│   └── registry/                     # paths、manifest、run registry
├── apps/
│   ├── real_capture/                 # 采集 GUI/进程入口，不拥有数据根
│   └── real_validation/              # 验证 GUI/进程入口，不复制数据集
├── scripts/                          # 薄 CLI，核心逻辑调用 src
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   └── maintenance/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── hardware/
│   └── fixtures/                     # 小而明确的测试夹具，不放展示 GIF
├── notebooks/                        # 探索，正式结论回写 docs/runs
├── docs/                             # 见第 7 节
├── external/                         # 第三方源码/旧工程；每项有 provenance
└── workspace/                        # 默认被 git 忽略；也可映射到仓库外
```

这不是要求第一阶段立刻把 `real_capture` 和 `real_validation` 改名到 `apps/`。优先级应是先解除它们对内部数据路径和共享算法的所有权，再做物理移动。

### 3.1 共享代码边界

当前 `scripts/real/masks_to_transition_npz.py` 直接导入 `real_validation.perception`，说明可复用感知逻辑实际被应用目录拥有。目标应为：

```text
src/perception/*       <- 被 preprocessing 和 real_validation 共同使用
src/data/real/*        <- manifest、坐标、动作和 NPZ 合同
src/training/*         <- 训练协议与统一 engine
apps/*                 <- GUI、生命周期编排、平台适配
scripts/*              <- 参数解析和调用，不承载主要业务算法
```

`sam2/sam2_src`、旧 PLC 工程和参考仓库应进入 `external/` 或由依赖安装脚本管理；SAM2 checkpoint、mask 和临时 JPEG 则属于 workspace，不跟第三方源码放在一起。

---

## 4. 统一工作区与路径注册器

### 4.1 逻辑布局

```text
workspace/
├── data/
│   ├── raw/
│   │   ├── real/<sequence_id>/
│   │   └── simulation/<sequence_id>/
│   ├── intermediate/
│   │   └── real/<sequence_id>/<recipe_id>/
│   │       ├── crop/
│   │       ├── masks/
│   │       ├── skeleton/
│   │       └── qc/
│   ├── processed/
│   │   ├── real/<dataset_id>/
│   │   │   ├── manifest.json
│   │   │   └── splits/{train,val,test}/
│   │   └── simulation/<dataset_id>/
│   └── fixtures/                     # GUI demo/离线锚定的小数据
├── runs/
│   ├── training/<study_id>/<run_id>/
│   ├── validation/<run_id>/
│   └── analysis/<analysis_id>/<run_id>/
├── models/
│   └── pretrained/sam2/
├── reports/                          # 选定后发布的人类可读结果
├── cache/                            # 可安全重建
└── registry/
    ├── datasets.jsonl
    └── runs.jsonl
```

### 4.2 配置优先级

路径根只在一个地方解析，建议优先级为：

1. CLI 显式 `--workspace-root`；
2. 环境变量 `SSR_WORKSPACE_ROOT`；
3. `configs/local.toml`；
4. 默认 `<repo>/workspace`。

示例：

```toml
# configs/paths.example.toml
[paths]
workspace_root = "workspace"

[compat]
enable_legacy_reads = true
legacy_raw_root = "real_capture/data/raw"
legacy_derived_root = "real_capture/data/derived"
legacy_processed_root = "data/real_seq"
legacy_training_root = "train_log"
```

`ProjectPaths`/`ArtifactRegistry` 应提供 `raw_sequence(id)`、`dataset(id)`、`training_run(id)` 等方法。除这一层外，禁止出现新的仓库相对数据常量。manifest 内优先写相对 workspace 的 URI，例如 `artifact://data/raw/real/seq_...`，不写 `/Data5/ddf/...` 绝对路径。

### 4.3 当前路径到目标路径的映射

| 当前路径 | 目标逻辑路径 | 迁移方式 |
|---|---|---|
| `real_capture/data/raw/<seq>` | `workspace/data/raw/real/<seq>` | 原目录注册或移动后留只读链接；不复制 |
| `real_capture/data/derived/<seq>` | `workspace/data/intermediate/real/<seq>/<recipe_id>` | 按 recipe 拆版本；旧目录整体注册为 `legacy` |
| `sam2/masks/<seq>_*` | 对应 intermediate 的 `masks/sam2/` | 记录 SAM2 版本/checkpoint hash/anchors |
| `data/real_seq/<long-name>` | `workspace/data/processed/real/<dataset_id>` | 从现有 `dataset_manifest.json` 导入 registry |
| `real_validation/data/npz` | `workspace/data/fixtures` 或已发布 dataset | GUI 使用 dataset selector，不维护正式副本 |
| `train_log/<model>/<exp>` | `workspace/runs/training/<study>/<run>` | 历史只读导入；不重命名旧实验 |
| `real_validation/runs` | `workspace/runs/validation` | 保留现场配置、命令 ACK、观测和评价 |
| `output/<topic>` | `workspace/runs/analysis/<topic>/<run>` | 每次分析必须声明输入 checkpoint/dataset |
| `sam2/checkpoints` | `workspace/models/pretrained/sam2` | checkpoint hash 与来源写 provenance |
| `docs/ref` | `external/` 或仓库外 reference root | 不再把外部项目伪装成文档 |

### 4.4 数据写入规则

- 所有写入先创建临时目录，完成后原子改名为最终 ID；
- 目标目录存在即拒绝，除非命令明确是 `resume` 且合同匹配；
- intermediate 可以被垃圾回收，但 registry 必须能指出重建命令；
- raw 和 processed release 默认 chmod/read-only 只是附加保护，真正约束来自“新版本不覆盖旧版本”；
- 任何跨盘软链接必须由 registry 记录真实目标，业务代码不直接解析链接结构。

---

## 5. 数据 manifest 与谱系共识

### 5.1 四种身份

| 身份 | 示例 | 含义 |
|---|---|---|
| `sequence_id` | `real_20260819_172644` | 一次不可变采集 |
| `recipe_id` | `sam2_skeleton_v3-a13c9e2` | 处理代码版本 + 关键参数 hash |
| `dataset_id` | `real_172644_mm_n15-r004` | 通过 QC、可供训练引用的发布版 |
| `run_id` | `trial_20260831_007` | 一次不可覆盖的执行 |

ID 只需可读且唯一，不应继续把所有参数都塞进名字。完整语义在 manifest 中。

### 5.2 processed dataset manifest 最低字段

```json
{
  "schema_version": 2,
  "dataset_id": "real_172644_mm_n15-r004",
  "created_at": "...",
  "status": "released",
  "sources": [
    {"sequence_id": "real_20260819_172644", "raw_manifest_sha256": "..."}
  ],
  "recipe": {
    "name": "real_sam2_to_transition",
    "version": 3,
    "git_commit": "...",
    "parameters": {},
    "commands": []
  },
  "contracts": {
    "state": {},
    "action": {},
    "timing": {},
    "observation": {}
  },
  "split_policy": {
    "name": "chronological_purged_v1",
    "group_key": "sequence_id",
    "seed": null,
    "embargo_frames": 40
  },
  "splits": {
    "train": [],
    "val": [],
    "test": []
  },
  "quality_control": {},
  "files": [{"uri": "artifact://...", "sha256": "...", "bytes": 0}]
}
```

现有 manifest 已经覆盖 state/action/timing/QC 的大量字段，应增量升级而不是推倒重写。首要改动是消除绝对路径、补充 source hash、recipe 版本、明确 split policy 和可选 test。

### 5.3 数据划分共同协议

#### 强制规则

1. **先划分，再构造窗口/episode。** 同一帧及由其生成的历史窗口不得跨 split 重用。
2. **默认按采集序列或独立实验组分组。** 不允许把全部帧打乱后随机切分时序数据。
3. 同一连续序列需要工程性 train/val 时，使用连续块并在边界设置 purge/embargo；建议最小值为 `max(history_window, rollout_window)`。
4. 归一化、统计阈值和可学习预处理只在 train 上拟合，结果序列化后原样用于 val/test。
5. split 写入 dataset manifest 后冻结。更换比例、seed、分组或 embargo 均产生新的 dataset release。
6. test 在模型、epoch、超参数和阈值全部冻结后只做最终报告；任何基于 test 的选择都必须把它降格为 val，并建立新 test。

#### 三种允许的证据等级

| 等级 | 划分 | 可以支持的结论 |
|---|---|---|
| `smoke` | 少量固定夹具 | 代码和 schema 能运行 |
| `within_sequence` | 单序列连续分块 + embargo | 同一采集条件下的插值/时间外推工程验证 |
| `cross_sequence` | sequence/group holdout，必要时跨日期/工况 | 科学比较和泛化主结论 |

当前单序列末尾 20% 验证属于 `within_sequence`。它可以继续用于工程迭代，但报告中不能替代跨序列 test。

#### 模型可声明的差异

- 渲染/多视角模型可要求按场景或相机 rig 分组；
- GTObserved 可用 one-step/observed-state 指标；
- OpenLoop 必须包含窗口 rollout 指标，且 `K_eval` 与部署合同一致；
- Hereditary 除预测误差外可以输出算子谱和消融诊断，但不能用解释性诊断替代 held-out 预测指标；
- 真实控制 test 必须按独立执行 run 划分，不能以离线 recorded action 回放代替。

---

## 6. 训练、验证、早停与试次共识

### 6.1 三层配置

```text
ProjectTrainingProtocol   # 所有模型必须遵守的可复现与选择合同
        +
ModelTrainingSpec         # 模型声明 phase、dataset、loss、metric、rollout 需求
        +
RunConfig                 # 本次 dataset、seed、epoch、lr、资源与显式覆盖
```

现有 `TrainingSpec/PhaseSpec` 是第二层的良好基础，但还应增加验证和选择字段；现有 `manage_training_trial.py` 是第一、三层试次归档的良好基础，应推广而不是另造多套目录。

### 6.2 所有正式训练的强制字段

每个 run 在启动前写入并冻结：

- `run_id`、父 study、创建时间、状态；
- dataset manifest URI + sha256；
- git commit、dirty diff hash（dirty 时保存 patch 或明确拒绝正式 run）；
- 实际命令、解析后的完整 config、Python/CUDA/关键依赖环境；
- seed、确定性设置和设备信息；
- 各 phase 的输入 checkpoint 及 sha256；
- 选择指标、方向、评价频率、早停策略；
- 预期产物和完成条件。

### 6.3 建议的统一 run 布局

```text
<run_id>/
├── run_manifest.json
├── config.resolved.json
├── commands.sh
├── environment.txt
├── source.patch                     # 工作树非干净时
├── status.json
├── stages/
│   └── <phase>/
│       ├── train.log
│       ├── metrics.csv
│       └── checkpoints/
│           ├── last.pt
│           ├── best_train.pt        # 可选，仅诊断
│           ├── best_val.pt          # 正式选择权重
│           └── epoch_XXXX.pt
├── evaluations/
│   ├── validation/
│   └── test/                        # 冻结后一次性生成
├── diagnostics/
├── artifacts.json
└── COMPLETE                         # 只有完整合同满足才生成
```

不再用含义模糊的 `best_model.pt` 作为跨工具接口。兼容期可以保留它，但 manifest 必须注明它等价于 `best_train` 还是 `best_val`。OpenLoop 热启动必须从 manifest 中显式取得同一试次 GT 权重，不自动扫描“最新”目录用于正式训练。

### 6.4 验证和 checkpoint 选择

每个 `PhaseSpec` 建议增加：

```text
validation_dataset_role: val
selection_metric: validation.node_mean_mm
selection_mode: min
eval_interval_epochs: 5
min_delta: 0.01
warmup_evaluations: 2
early_stopping_patience_evaluations: null
lr_scheduler_metric: validation.node_mean_mm
restore_best_at_end: true
```

规则如下：

- `best_train` 只反映优化目标，不能默认用于论文或部署；
- `best_val` 才是模型选择结果；
- scheduler 默认观察 validation selection metric；若模型有充分理由观察训练 loss，必须显式声明；
- 指标不可比时（如单位或节点合同变化）必须拒绝在同一 study 排名；
- OpenLoop 的选择指标应基于部署窗口内的 rollout，而非只看 one-step；
- 多目标选择先声明主指标，安全/稳定性作为 gate，不在运行结束后临时挑对自己有利的指标。

### 6.5 早停共识

早停不是所有模型都必须启用，但语义必须统一：

- patience 以“完成了多少次验证”计数，不以训练 batch 或模糊 epoch 计数；
- `warmup_evaluations` 结束前不早停；
- 只有超过 `min_delta` 才重置 patience；
- 达到 patience 后先原子保存 `last.pt` 和状态，再结束；
- 完成后恢复 `best_val` 进入后续 phase/最终 val；
- OpenLoop 的 teacher-forcing 退火尚未结束时默认不得早停，除非模型 spec 明确允许；
- 两阶段模型必须逐 phase 声明，不能把后一阶段的验证规则套给前一阶段；
- Hereditary 等长周期模型可以关闭早停，以固定预算做公平消融，但仍要周期 val 和 `best_val`。

`scheduler_patience` 与 `early_stopping_patience` 必须使用两个字段，不共享 CLI 名称。

### 6.6 测试与最终报告

1. 用 val 完成超参数、epoch 和 checkpoint 选择；
2. 冻结 run manifest 和选择结果；
3. 在 test 上运行预注册的统一评价；
4. 结果写到当前 run 的 `evaluations/test/`，不写全局 `output/exp_name`；
5. 若进行真实执行，再创建关联的 validation run，记录 checkpoint hash、现场配置、动作 ACK、执行后骨架和独立 NDI；
6. 报告引用 `dataset_id + run_id + checkpoint sha256`，不引用“最新模型”。

### 6.7 快速试验与正式试次

允许保留直接 Python CLI 以快速试学习率，但必须标为 `run_kind=exploratory`。需要共同分析、写入报告或作为后续模型初始化的实验，应提升为正式试次，采用独立目录并保存实际命令、config、checkpoint 和一致评价。快速试验也不得覆盖已有目录。

---

## 7. 文档信息架构与治理

### 7.1 目标结构

```text
docs/
├── README.md                         # 唯一导航，CI 检查链接
├── overview/
│   ├── project.md                    # 稳定目标和范围
│   ├── architecture.md               # 当前系统边界
│   └── status.md                     # 唯一“现在到哪了”，短且定期更新
├── standards/
│   ├── data_lifecycle.md
│   ├── dataset_split.md
│   ├── training_and_evaluation.md
│   ├── experiment_layout.md
│   └── evidence_language.md
├── workflows/
│   ├── real_capture.md
│   ├── real_preprocessing.md
│   ├── training.md
│   └── real_validation.md
├── architecture/
│   ├── decisions/                    # ADR：为什么这样做
│   └── designs/                      # 尚未实现/正在实现的设计
├── experiments/
│   ├── protocols/                    # 实验前声明
│   └── records/                      # 实验后摘要；大产物引用 run_id
├── research/
│   ├── directions/
│   ├── literature/                   # 文献笔记与综述
│   └── paper/                        # 唯一活跃论文树
├── presentations/
└── archive/
    └── manifest.md
```

### 7.2 当前文档的建议归位

| 当前内容 | 建议 |
|---|---|
| `HANDOFF.md` | 缩成入口和安全不变量；动态数字链接 `overview/status.md` |
| `overview/status.md` | 成为唯一当前状态；避免复制完整教程和论文论证 |
| `real_data/workflow.md`、`general_6ch_postprocess.md` | 合并出一个通用 preprocessing workflow；旧单通道内容归档 |
| `seq_*.md` | 移到 `experiments/records/<sequence_id>/`，只描述该次处理证据 |
| `paper/` | 作为唯一活跃 manuscript，重命名到 `research/paper/` |
| `papers/notes`、综述 | 移到 `research/literature/`；不要与论文草稿同名 |
| `papers/*_draft.md` | 明确并入 active paper 或标为 superseded/archive |
| `directions/` | 移到 `research/directions/`，overview 标记 active/parked/rejected |
| `superpowers/specs/plans` | 已落地者转 ADR/归档，未落地者进入 designs，不以工具名分类 |
| `docs/ref` | 移出 docs；在 `external/manifest.toml` 记录来源、commit、license |
| HTML 报告和大型图 | 正式发布放 reports，运行生成物留 workspace run |

### 7.3 文档 front matter

所有长期文档建议包含：

```yaml
---
title: ...
kind: overview | standard | workflow | design | protocol | record | paper
status: draft | active | superseded | archived
updated: 2026-08-31
scope: ...
supersedes: []
superseded_by: null
sources: []
---
```

实验记录再增加 `dataset_ids`、`run_ids`、`claim_level`。文档中的“当前”“最好”“最新”若没有日期和 run ID，CI lint 应警告。

### 7.4 单一事实来源规则

| 事实 | 唯一拥有者 | 其他文档如何使用 |
|---|---|---|
| 项目当前主线/里程碑 | `overview/status.md` | 链接，不复制 |
| 数据 schema/划分规则 | `standards/data_lifecycle.md`、dataset manifest | 链接或引用版本 |
| 训练选择/早停规则 | `standards/training_and_evaluation.md` | 模型文档只写差异 |
| 某次运行参数和结果 | run manifest + experiment record | 不手抄完整 config |
| 模型原理 | 模型 design/源码 docstring | status 只写一句定位 |
| 文献证据 | literature note/综述 | 论文草稿引用，不重复检索过程 |

### 7.5 文档保留与归档判断

对每份文档执行五问：

1. 它描述的是长期规则、设计、操作流程，还是一次运行？
2. 其中的事实是否已有更权威拥有者？
3. 命令在当前代码上是否仍可运行？
4. 是否有后续文档明确取代它？
5. 删除它会不会损失无法从 git 或 run manifest 恢复的科学证据？

裁决只有四种：`keep`、`merge`、`supersede`、`archive`。首轮不做物理删除；exact duplicate 也先在 archive manifest 标明来源和目标，再单独审查。

---

## 8. 分阶段迁移方案

### Phase 0：冻结规则和建立清单（低风险）

- 接受本文的目标边界；
- 生成当前 raw sequence、processed dataset、training run、validation run 清单；
- 标出唯一副本、可重建产物、活跃进程和未完成试次；
- 不移动任何目录，不改正在运行的 tmux 实验。

**验收：** 每个现有主线 dataset 能映射到原始 sequence；每个需保留 run 能映射到 dataset/checkpoint；未知项进入 quarantine 清单而不是猜测。

### Phase 1：实现路径与 manifest 基础设施（最高优先级）

- 新增 `ProjectPaths`、artifact URI 和 schema 校验；
- 新增 `configs/paths.example.toml` 与本机覆盖；
- 为现有 dataset manifest 写只读升级器；
- 为路径注册器、legacy fallback、跨盘路径和 hash 校验添加测试。

**验收：** 同一命令可以在默认 workspace 和外部 workspace 运行；核心代码不依赖 cwd；新 manifest 无绝对项目路径。

### Phase 2：数据管线双读、单写

- `real_capture` 新采集只写 `workspace/data/raw/real`；
- 预处理器能读旧 raw，但新 intermediate/processed 只写 workspace；
- `real_validation` 通过 dataset/fixture selector 选择 NPZ，不要求复制到应用目录；
- SAM2 mask 进入对应 recipe 目录，记录 checkpoint hash。

**验收：** 对一个小序列做旧路径与新路径 A/B，帧数、时间戳、动作、节点、split 和关键数组 hash 一致；旧数据保持未修改。

### Phase 3：统一训练 engine 和试次合同

- 把完整真实流水线的 val checkpoint 选择能力收进公共 engine；
- 扩展 `PhaseSpec` 的 validation/selection/early-stop 字段；
- 统一 run manifest、完成标记、resume 和显式 checkpoint 解析；
- GT/OpenLoop/Hereditary 先迁移，再评估旧渲染模型是否仍 active。

**验收：** focused test 覆盖无 val、带 val、早停、resume、多 phase、OpenLoop 退火和 Hereditary 固定预算；同一试次所有阶段和评价保存在一个目录，不覆盖历史 log。

### Phase 4：历史数据和实验只读导入

> 2026-09-01 实施说明：已用
> `docs/maintenance/2026-09-01_legacy_asset_migration.json` 登记并原子迁移
> 67 项历史资产。按本次轻量迁移决策，不对同文件系统 rename 重复计算逐文件
> SHA-256；ledger 记录 source/target 与状态，目标存在时拒绝覆盖。
>
> 同日新增 `workspace/registry/workspace_asset_index.json`：只读取目录、小型 JSON
> 和完成标记，不 hash NPZ/checkpoint 等 payload。它提供 dataset→run 反向查询，
> 但不能替代严格不可变 manifest。4 个已有 `real_pipeline` 试次已补非覆盖的
> `legacy_run_manifest.json`，并明确记录 validation 选择或训练 loss/未知语义。
> 其余 10 个原本无清单的旧 dataset 已补观察型 `legacy_dataset_manifest.json`；
> 来源仅在 NPZ 文件名可确认时记录，训练就绪状态仍标为 `unknown`。索引发现的
> 15 条缺失 dataset 引用保持 unresolved，不能据此删除关联归档 run。

- 登记 source/target、资产类型和迁移状态；跨文件系统复制时再增加内容 hash；
- 在 registry 登记旧路径，不强制重命名；
- 对需要长期保留的 run 补 `legacy_run_manifest.json`；
- `output/` 中可再生项只记录生成命令，正式结果关联到对应 run。

**验收：** registry 可查询“dataset 被哪些 run 使用”和“run 来自哪些 sequence”；随机抽样打开 checkpoint、NPZ、图片和日志均成功。

### Phase 5：文档合并与状态重写

- 先建立新 `docs/README.md` 和 front matter lint；
- 重写唯一 status；
- 处理 `paper/papers`、通用/单序列 real_data、HANDOFF 重复；
- 建立 archive manifest，保留旧链接重定向或 stub 一段过渡期。

**验收：** 新读者从根 README 在三次点击内找到当前状态、数据处理、正式训练和真实验证；所有内部链接通过；不存在两个 active 文档同时声明同一“当前最佳模型”。

### Phase 6：物理清理（最后、单独审批）

> 2026-09-01 实施说明：历史资产部分已获授权完成。完整真实流水线使用 911 帧
> 数据集完成 GT 2 epoch + OpenLoop 2 epoch、周期/最佳评价和叠图；随后移除
> 67 个兼容链接与空旧根。试次保存在
> `workspace/runs/training/real_pipeline/seq_20260819_182253/trial_20260901_000/`。

- 仅在 canonical 目标存在、目标不覆盖保护和回滚路径确认后处理旧入口；
- 清理 `__pycache__`、根目录生成图、tests GIF、废弃工作树等可重建内容；
- 旧路径先改只读链接，再经过主线短训练与评价确认可运行后移除。

**验收：** 清理列表逐项记录“删除对象、原大小、可恢复位置/重建命令”；不对 workspace 根或仓库根执行宽泛递归删除。

---

## 9. 优先级与首批任务

### P0：先修定义，暂不搬数据

1. 决定 workspace root 配置名和默认位置；
2. 定稿 dataset/run manifest v2；
3. 为现有真实 dataset 生成 registry 索引；
4. 写训练选择合同，明确 `best_model.pt` 和 `best_eval_model.pt` 的现状；
5. 禁止新增硬编码 `real_capture/data`、`train_log`、`output`（已由可执行字面量非递增 baseline 守门）。

### P1：打通一条参考主线

> 2026-09-01：已发布 `real_transition_reference_10hz_v1`，明确 train/val 与
> 独立序列 frozen test；发布和验收记录见
> [`../maintenance/2026-09-01_reference_dataset_release.md`](../maintenance/2026-09-01_reference_dataset_release.md)。
> 训练、最终 test 和 offline fixture 验收继续使用该不可变 release。

选择一个已完成、非正在训练的真实序列，完整验证：

```text
legacy raw
 -> new registered intermediate
 -> immutable processed release(train/val/test contract)
 -> GT stage
 -> OpenLoop stage
 -> val selection
 -> frozen test
 -> offline real_validation fixture
```

这一条参考主线通过后，再迁移 Hereditary 和多序列组合。不要同时改所有旧模型。

### P2：文档和非主线资产

- 合并导航和状态；
- 分离 active paper 与 literature；
- 迁出 external reference；
- 分类旧模型为 `active`、`reproducibility-only`、`archived`，再决定是否移动源码。

---

## 10. 验证、回滚与风险控制

### 10.1 每次路径迁移必须验证

- 源/目标文件数、总字节数；
- manifest 和关键文件 sha256；
- NPZ key、shape、dtype、有限值、坐标/动作/节点合同；
- split 互斥和 lineage；
- 代表性 loader、训练 dry-run、评价 dry-run；
- 文档和 run manifest 引用扫描。

### 10.2 回滚策略

- 兼容期 registry 同时保存 legacy URI 和 canonical URI；
- 新写入失败时不回写旧路径；
- 物理移动使用同一文件系统原子 rename，跨文件系统则 copy -> verify -> switch registry，源目录最后处理；
- 任一 consumer 验证失败就把 canonical URI 切回 legacy URI，不修改数据内容；
- 旧目录未经过完整训练和验证周期不得删除。

### 10.3 主要风险

| 风险 | 控制 |
|---|---|
| 19 GB raw 复制导致空间翻倍或中断 | 优先原地注册/rename，不盲目 copy |
| 绝对路径写入历史 manifest | 升级时保留原值到 `legacy`，新字段使用 artifact URI |
| 正在训练的 run 被重命名 | Phase 0 标记 active process，历史 run 只读导入 |
| “统一训练”破坏模型特有需求 | 强制协议与 ModelTrainingSpec 分层 |
| val 被反复调参后当 test | manifest 明确 role；锁定独立 test/新序列 |
| 文档合并丢失证据 | 先 supersede + archive manifest，不先删除 |
| 软链接隐藏跨盘失效 | registry 记录 resolved target，启动时 preflight |

---

## 11. 完成定义

这次整理只有同时满足以下条件，才算真正完成，而不是“看起来更整齐”：

- 新增业务代码不再硬编码多个数据/输出根；
- 任一 processed dataset 都能追溯到 raw sequence、处理 recipe、参数、代码版本和 QC；
- 任一正式 checkpoint 都能追溯到 dataset、完整 config、命令、源码状态和选择指标；
- train/val/test 的角色和证据等级明确，test 不参与选择；
- 所有模型使用共同训练合同，同时通过 ModelTrainingSpec 保留合理差异；
- 一次实验的阶段、评价、配置、命令和完成标记保存在同一 run 下，且不覆盖历史 log；
- `real_capture`、`real_validation`、SAM2 和训练脚本通过 registry 共享数据，不维护各自的正式副本；
- 文档只有一个当前状态入口、一个训练共识入口和一个数据生命周期入口；
- 所有历史内容在删除前都具有可验证的迁移记录和回滚方案。

本设计建议先实施 Phase 0–1。它们能建立后续整理的坐标系，且不触碰现有 26 GB 工作资产；在路径注册、manifest 和训练合同落地之前直接移动目录，只会把隐含混乱换成新的隐含混乱。
