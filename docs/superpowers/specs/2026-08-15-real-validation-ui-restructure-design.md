# 设计:real_validation 界面重构 —— 左右两栏 + 跨页摄像头 + 紧凑排版 + 多选编辑

> 日期:2026-08-15 · 分支 `feat/real-data-transition`
> 目标:①布局改为**左右两栏**(左固定显示区 + 右 5 页 Tab,参考 real_capture);②左显示区为**主显示视图**(摄像头 + 可叠加多层,带图层勾选框);③摄像头流**窗口级共享**(Observe/Plan/Execute 都可见);④Setup/Plan 参数**紧凑排版一次显示全**(安全配置 6×5 不滚动);⑤场景编辑**多选删除**(ExtendedSelection + 批量删 + Del)。
> 前置:real_validation 已完全自包含 + 目录重构完成(contracts/core/planning/runtime/execution/hardware/gui/tools 子包)。

---

## 1. 目标与非目标

### 1.1 要实现

1. **左右两栏布局**:窗口主区改为 `QSplitter(Horizontal)`——左固定显示区(~520px)+ 右 5 页 Tab。
2. **主显示视图**(增强 `CameraViewWidget`):摄像头 + 叠加骨架/场景目标障碍/预测轨迹/实际骨架/NDI,带图层勾选框。
3. **跨页摄像头**:`_CameraThread` 提升为窗口级共享,`_on_camera_frame` 喂主显示视图;Observe 保留自身 camera_view 用于锚定交互。
4. **紧凑排版**:安全配置 6×5 压缩行高一次显示全;Setup/Plan 参数压缩到一屏。
5. **多选删除**:scene_editor 改 ExtendedSelection + 批量删 + Del 快捷键。

### 1.2 非目标

- 不改数据契约(contracts/)、不改规划/执行/感知逻辑——纯 GUI 布局 + 交互增强。
- 不做新的规划/执行功能(预测轨迹/实际骨架叠加是**显示已有数据**,不新增数据源)。
- 不重构已有的子包结构(布局改动只碰 gui/ 层)。

---

## 2. 整体布局架构

```
QMainWindow
└── 顶部安全栏(不变: Run/State/Hardware + 归零/中止)
└── 主 QSplitter(Horizontal)
    ├── 左: 主显示区(固定宽 ~520)
    │   ├── MainDisplayView(摄像头 + 多层叠加)
    │   └── 图层勾选框行(骨架/场景/预测/实际/NDI)
    └── 右: QTabWidget(5 页,保持现有内容)
        ├── 1 Setup  2 Observe & Scene  3 Plan  4 Execute  5 Results
```

- 左栏固定宽(用户可拖 splitter 调,默认 ~520)。
- 右栏 tabs 保持现有页面结构(不重排页面内部,只压缩空间)。
- 摄像头流启动后常驻左栏显示区;Observe 页内的 camera_view 仍保留用于**锚定交互**(图上点选目标/障碍),但摄像头帧由窗口级流共享喂给两者。

---

## 3. 主显示视图(增强 CameraViewWidget)

**决策**:增强现有 `real_validation/gui/widgets/camera_view.py` 的 `CameraViewWidget`,不新建重名 widget(避免与 Observe 页重复实例混淆)。

现有能力(已实现):`set_frame`(BGR 图)、`set_skeleton`(青线+圆点)、`set_anchor`(绿三角)、`set_scene`(目标红 x/障碍琥珀虚线)、鼠标点选(select/add_target/add_obstacle/add_target_skeleton)。

**新增能力**:
- `set_predicted_states(states: np.ndarray)`:`(K,N,2)` 像素坐标 → 叠加规划预测轨迹(灰虚线连节点,随 k 步)。来源:`plan.predicted_states_path` 的 `states_model`(规划后由 GUI 读入喂给主显示视图)。
- `set_actual_skeleton(skeleton: np.ndarray)`:`(N,2)` → 执行中每步实际骨架(红点)。来源:`_on_camera_frame` 的 `extract_skeleton_2d` 结果,执行时每步更新。
- `set_ndi_position(xy: tuple[float,float] | None)`:`ndi_mm` → NDI 末端(mm,Mock 下 None/占位)。只显示,不进模型(遵守隐藏评价流约束)。
- **图层开关**:`set_layer_visible(layer: str, visible: bool)`,图层 key:`skeleton/scene/predicted/actual/ndi`。GUI 加一排 QCheckBox 控制各层可见性(默认:骨架 ✓、场景 ✓、预测 ✓、实际 ✓、NDI ✗ 默认关)。

**图层与数据流**:
| 图层 | 数据 | 何时有值 |
|---|---|---|
| `skeleton` | `set_skeleton` | 摄像头帧 + 模型已加载(实时) |
| `scene` | `set_scene` | 锚定/设目标障碍后 |
| `predicted` | `set_predicted_states` | 规划完成(读 predicted_states.npz) |
| `actual` | `set_actual_skeleton` | 执行中每步 |
| `ndi` | `set_ndi_position` | 真机 NDI(可选) |

**Observe 页复用**:Observe 页的 `self.camera_view` 改为**引用主显示视图**(同一实例),保留点选交互;或保留独立 camera_view 仅用于锚定。**决策**:Observe 页 camera_view = 主显示视图实例(左栏),Observe 页内不再单独建 camera_view,而是把锚定/场景工具栏移到左栏主显示区下方或 Observe 页内引用。为最小改动:Observe 页保留自身的 camera_view(交互不变),主显示区是**另一个增强实例**用于常驻显示。两者都订阅窗口级摄像头流。**(保留方案:两个实例,Observe 用于锚定交互,主显示用于常驻叠加显示。)**

---

## 4. 跨页摄像头流

- `_CameraThread` 已在窗口级(`self._camera_thread`),保持。
- `_start_camera` 启动流 → `_on_camera_frame` 现在喂**主显示视图**(`self.main_display.set_frame(bgr)`)AND 若 Observe 页有独立 camera_view 也喂(`self.camera_view.set_frame(bgr)`)。
- 骨架提取(`extract_skeleton_2d`)在 `_on_camera_frame` 做一次,分发给主显示(骨架层)+ Observe(锚定用)。
- Plan/Execute 页无需新增相机 widget——左栏主显示区常驻可见,天然看到摄像头 + 叠加。

**数据流**:摄像头帧 → `_on_camera_frame` → `main_display.set_frame` + (可选)Observe camera_view.set_frame + `extract_skeleton_2d` → 主显示 set_skeleton + 执行时 set_actual_skeleton。

---

## 5. 紧凑排版

### 5.1 安全配置(Setup 页,核心痛点)
当前:`QTableWidget(6,5)` 每格 QDoubleSpinBox,6 行在默认高度需滚动。
改:
```python
self.safety_table.verticalHeader().setDefaultSectionSize(26)   # 压缩行高
self.safety_table.verticalHeader().setVisible(True)            # 保留 ch0-ch5 标签
self.safety_table.horizontalHeader().setDefaultSectionSize(96) # 压缩列宽
self.safety_table.setMinimumHeight(6 * 26 + 30)                # 一次显示全 6 行
```
6×26=156px + 表头 30 ≈ 186px——一屏放下,无需滚动。

### 5.2 Setup 页整体
- 卡间距压缩(`layout.setSpacing(8)`),控件间 `setContentsMargins(8, 10, 8, 8)`。
- model_summary(QPlainTextEdit)限高(如 120px),不再撑开。
- 目标:Setup 页一屏显示完全部(实验/模型/安全/硬件连接四卡)。

### 5.3 Plan 页
- QFormLayout 压缩行距(`form.setVerticalSpacing(6)`)、SpinBox 紧凑。
- PlanPreviewWidget 高度适当(可缩小默认,或用户拖 splitter)。
- 目标:Plan 参数卡 + 预览一屏。

### 5.4 通用
- 所有 QGroupBox 内边距统一压缩(`setContentsMargins(8, 10, 8, 8)`)。
- 窗口默认 `resize(1400, 860)`(略增宽给左栏)。

---

## 6. 多选删除(scene_editor)

`real_validation/gui/widgets/scene_editor.py`:
```python
self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)   # Ctrl 多选 / Shift 范围 / Ctrl+A 全选
```
- 删除按钮批量删选中项:
```python
def _on_delete(self):
    selected_rows = sorted({idx.row() for idx in self.list.selectedIndexes()}, reverse=True)
    primitives = list(self._scene.primitives)
    for row in selected_rows:
        # 逐个 without_primitive(按 primitive_id),一次 emit scene_edited
```
- 保留单选时的 `_on_select` 属性表单(点选单个编辑名称)。
- **Del 快捷键**:在 `SceneEditorPanel` 加 `keyPressEvent`(焦点在列表时 Del 触发 `_on_delete`);窗口级 `keyPressEvent` 转发或列表自身处理。
- 批量删后 emit 一次 `scene_edited(updated)`,GUI `_apply_scene_edit` 走 session.set_scene(B16 守卫 + 落盘)。

---

## 7. 文件改动清单

| 文件 | 改动 |
|---|---|
| `gui/widgets/camera_view.py` | 增强:set_predicted_states/set_actual_skeleton/set_ndi_position/set_layer_visible |
| `gui/widgets/scene_editor.py` | ExtendedSelection + 批量删 + Del 键 |
| `gui/main_window.py` | `_build_ui` 改左右两栏 + 主显示视图实例 + 图层勾选框 + `_on_camera_frame` 分发 + 紧凑排版(Setup/Plan)+ 安全配置压缩 + 窗口 resize |
| `gui/widgets/plan_preview.py` | (可选)紧凑化 |
| `real_validation/__init__.py` | **不改**(保持 stdlib-only) |
| `tests/` | 更新 GUI 回归测试(布局相关断言) |

---

## 8. 验证

1. 全量测试:`python -m unittest discover -s tests -v` 全绿(130 + 新增/更新)。
2. import 卫生:`tests/test_import_hygiene.py` 全绿(包根 stdlib-only 不破)。
3. offscreen 冒烟:`QT_QPA_PLATFORM=offscreen` 构造 ValidationWindow,断言:
   - 主区有 QSplitter(Horizontal)、左栏 MainDisplayView、右栏 tabs
   - 安全配置表 6 行高度足够(setMinimumHeight 生效)
   - scene_editor 列表 ExtendedSelection 模式
4. 人工目检(有显示环境):两栏布局、摄像头跨页常驻、图层勾选框切换、多选删除。

---

## 9. 风险与约束

| 项 | 说明 |
|---|---|
| 摄像头双实例 | 主显示 + Observe 各自实例都订阅窗口级流;`_on_camera_frame` 需喂两个(或共享帧缓冲)。若帧率高双 set_frame 开销可接受(Mock 200ms/帧) |
| 锚定交互保留 | Observe 页 camera_view 的点选/工具逻辑不变;主显示区是只读显示(不参与锚定点击) |
| import 卫生 | 改动只碰 gui/ 层,包根 __init__ 不动 |
| 向后兼容 | 现有页面控件/信号不删,只改布局与新增叠加层;GUI 回归测试需同步 |
| 紧凑排版 | 压缩后需人工确认参数仍可读(行高 26px SpinBox 是否舒适) |
