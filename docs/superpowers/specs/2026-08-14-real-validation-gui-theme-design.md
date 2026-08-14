# 设计:real_validation 工作台 GUI 主题美化

> 日期:2026-08-14 · 分支 `feat/real-data-transition`
> 目标:把 `real_validation/` GUI 从默认 Qt 灰外观升级为 **real_capture 同款"现代扁平化医疗风格"**(浅蓝灰背景 + 白色圆角卡片 + 医疗青主按钮 + 警示红危险按钮 + 彩色状态),保留五页 Tab 结构、全部信号接线与逻辑,**纯视觉升级**。
> 参考实现:`docs/ref/Main UI-plc/ui_single_motor_v1.py` 内嵌 QSS、`docs/ref/Main UI-plc/main_capture.py`(pg 白底配置)。

---

## 1. 目标与非目标

### 1.1 要实现的能力

1. 一套统一 QSS 主题:窗口背景 `#F0F4F8`、QGroupBox 白底圆角卡片、医疗青/警示红/停止紫按钮语义、圆角输入框、中文字体回退链。
2. 五页(Tab)每页控件用 QGroupBox 卡片分组,信息层级对齐 real_capture 的控制面板观感。
3. 顶部安全栏升级:状态色块 QLabel(随 session state 变色)+ 归零/中止大按钮。
4. pyqtgraph 全部绘图区改白底(`background="#FFFFFF"`,前景 `#334E68`),曲线颜色沿用现有 `COLORS`。
5. 冒烟 + 既有测试全绿。

### 1.2 明确的非目标

- 不改五页 Tab 结构 / 布局骨架 / 信号槽接线 / 各 widget 逻辑。
- 不改 `real_validation/__init__.py`(**必须保持 stdlib-only**,theme 只被 `main_validation.py` 依赖)。
- 不加新功能、不加新依赖、不做深色主题。
- 不动纯逻辑模块(`models.py` / `session.py` / `preflight.py` / `executor.py` / `perception/`)。

---

## 2. 现状盘点

| 维度 | 当前 | 参考(real_capture) |
|---|---|---|
| 全局样式 | 无 QSS,默认 Qt 灰 | `ui_single_motor_v1.py` 现代扁平化医疗 QSS |
| 分组 | 控件平铺,无卡片 | QGroupBox 白卡 + 圆角 + 分组标题 |
| 按钮 | 默认灰,个别 setStyleSheet 变色 | 主按钮医疗青、急停警示红、语义清晰 |
| 状态 | 顶部一行 QLabel + 两个按钮 | 全局彩色状态栏 |
| 绘图 | pyqtgraph 默认黑底 | `pg.setConfigOptions(background="#FFFFFF", foreground="#334E68")` |
| 字体 | 默认 | `Microsoft YaHei` 雅黑 |

---

## 3. 配色方案(单一来源,放 `real_validation/widgets/theme.py`)

从 `ui_single_motor_v1.py` 提取 + 扩展现有 `main_validation.py` 已用的颜色(它们本来就对齐同一套):

| 角色 | 值 | 用途 |
|---|---|---|
| 窗口背景 | `#F0F4F8` | QMainWindow 背景 |
| 卡片背景 | `#FFFFFF` | QGroupBox / 文本框 / 表格 |
| 卡片边框 | `#D9E2EC` | QGroupBox / 输入框边框 |
| 主文本 | `#334E68` | QLabel / 分组标题 / 绘图前景 |
| 强调文本 | `#102A43` | 输入框文字 / 重要值 |
| 次级文本 | `#486581` | 占位提示 / 次要标签 |
| 输入框底 | `#F8FAFC` | QLineEdit / QSpinBox 背景 |
| **医疗青(主动作)** | `#2CB1BC` | 规划 / 执行 / 锚定 / 加载等主按钮;hover `#38BEC9`;pressed `#14919B` |
| **警示红(危险)** | `#EF4E4E` | 归零 / 中止 / 急停;hover `#F86A6A`;pressed `#E02424` |
| **停止紫(次要动作)** | `#667EEA` | 停止 / 取消类;hover `#7F9CF5`;pressed `#5A67D8` |
| 成功绿 | `#38A169` | 状态 OK / 已锚定 |
| 白底按钮文字 | `#FFFFFF` | 彩底按钮文字 |

**pyqtgraph 配置**:`PGG_OPTS = dict(background="#FFFFFF", foreground="#334E68")`,全局 `antialias=True`。

**字体回退链**(PC 是 Windows,开发是 Linux):`"Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", sans-serif`。

---

## 4. 组件 QSS 规则

### 4.1 按钮语义(通过 objectName + 选择器)

QSS 中定义三种彩底按钮样式,`main_validation.py` 为相应按钮设 `setObjectName`:

- `QPushButton#primary` — 医疗青底白字(规划 / 执行 / 锚定 / 加载 / Arm / 应用安全)
- `QPushButton#danger` — 警示红底白字(归零 / 中止 / 急停)
- `QPushButton#accent` — 停止紫底白字(取消规划 / Pause / 停止类)
- 其余按钮 — 白底 `#FFFFFF` + 边框 `#D9E2EC` + 主文本色(次要按钮,如"加载 scene.json"、"导入 plan.json")

**规则**:任何 setStyleSheet 内联改色(`background:#...;color:white`)的现有按钮改为统一走 objectName + QSS,内联样式在 `_build_ui` 里删除,颜色归 theme。

### 4.2 QGroupBox 卡片

```css
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    color: #334E68;
    font-weight: bold;
    font-size: 13px;
}
```

### 4.3 输入类控件

```css
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #D9E2EC;
    border-radius: 4px;
    padding: 4px 6px;
    background-color: #F8FAFC;
    color: #102A43;
    selection-background-color: #2CB1BC;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border: 1px solid #2CB1BC; }
```

### 4.4 表格(六通道安全表)

```css
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 6px;
    gridline-color: #D9E2EC;
    alternate-background-color: #F8FAFC;
}
QTableWidget::item { padding: 2px; }
QHeaderView::section {
    background-color: #E3E8EE;
    border: none;
    padding: 4px;
    color: #334E68;
    font-weight: bold;
}
```

### 4.5 文本区(日志 / 摘要 / 结果)

```css
QPlainTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 6px;
    padding: 6px;
    font-family: Consolas, "Microsoft YaHei", "Noto Sans CJK SC", monospace;
    color: #334E68;
}
```

### 4.6 Tab 页

```css
QTabWidget::pane { border: none; background: #F0F4F8; }
QTabBar::tab {
    background: #E3E8EE; color: #486581;
    padding: 8px 16px; border-radius: 6px 6px 0 0; margin-right: 2px;
}
QTabBar::tab:selected { background: #2CB1BC; color: #FFFFFF; font-weight: bold; }
```

---

## 5. 顶部安全栏升级

当前:一行 `QLabel(state_label)` + `归零` + `中止`。

改为:
- `state_label` 设 objectName `state_badge` → QSS 白底圆角 + 彩色边框 + 彩色文字,`_refresh()` 里按 state 设 4 种配色:
  | state | 文字色 / 边框色 |
  |---|---|
  | IDLE / no_session | `#486581` |
  | READY / COMPLETED / ZEROED | `#38A169` |
  | ARMED / PAUSED | `#F6AD55`(橙) |
  | EXECUTING / ABORTING / ERROR / PLANNING | `#EF4E4E`(红) |
- `归零`/`中止` 设 objectName `danger`(红底)。
- 文本格式保留 `Run: <name>    State: <state>    Hardware: MOCK`,再加一个模型摘要 QLabel 可选(`action_dim/n_nodes`),若模型已加载。

> 设计决策:不做多行的全局"硬件/模型/实验"三条状态条(那是 real_capture 控制面板的横向布局,与本工作台五页流程结构不符)。保持单行安全栏 + 状态色块,信息已覆盖。

---

## 6. 页面分组(每页 QGroupBox 卡片)

`main_validation.py` 现有 `_setup_page` / `_observe_page` / `_plan_page` / `_execute_page` / `_results_page` 的控件归属:

### 6.1 Setup 页
| 卡片 | 控件 |
|---|---|
| **实验与运行** | run_root、New Experiment、Open Run(Replay) |
| **模型与部署契约** | checkpoint、data_dir、k_safe、device、Load Model、model_summary |
| **安全配置(六通道 kPa / kPa·s⁻¹)** | safety_table + "应用安全配置"按钮 |

### 6.2 Observe & Scene 页
| 卡片 | 控件 |
|---|---|
| **离线锚定** | anchor_npz、帧索引、"从 NPZ 建立 Anchor"、加载 anchor.json |
| **目标与障碍** | 目标 x/y/半径、"设置末端目标"、障碍 x/y/半径、"添加圆障碍" |
| **实时相机与 Warmup** | Start Camera、Warmup、从相机取流锚定、工具选择(select/点加目标/点加障碍) |
| **场景编辑** | camera_view + scene_editor splitter、anchor_status |

> 设计决策:工具按钮("点加目标"/"点加障碍")归属"实时相机与 Warmup"卡,因为它们是相机视图上的交互工具,与 real_capture 把操作控件归同类卡片一致。

### 6.3 Plan 页
| 卡片 | 控件 |
|---|---|
| **规划参数** | K、优化迭代、多起点、动作周期(s)、模型维度→硬件通道 |
| **规划与预检** | 运行 OpenLoop Planner、取消规划、导入 plan.json、运行 Preflight、plan_summary |
| **(无卡片)** | PlanPreviewWidget(占满剩余,自带白底绘图) |

### 6.4 Execute 页
| 卡片 | 控件 |
|---|---|
| **执行控制** | Arm/Confirm、Mock Execute、Pause、Resume |
| **执行日志** | execution_log(QPlainTextEdit) |

### 6.5 Results 页
| 卡片 | 控件 |
|---|---|
| **结果与指标** | results(QPlainTextEdit) |

---

## 7. 绘图区白底

- `theme.configure_pyqtgraph()` 调 `pg.setConfigOptions(antialias=True, background="#FFFFFF", foreground="#334E68")`,在 `main_validation.py` **创建任何 PlotWidget 之前**调用(即在 `ValidationWindow.__init__` 的 `_build_ui` 前,或模块 import 后 main() 开头)。
- `plan_preview.py` / `camera_view.py`:
  - `showGrid(x=True, y=True, alpha=0.15)`
  - 标题文字默认即用 `foreground`(全局配置),曲线颜色沿用现有 `COLORS = ("#2CB1BC", "#667EEA", "#EF4E4E", "#F6AD55", "#38A169", "#805AD5")`,白底对比度 OK。
  - `camera_view` 的骨架青 `#2CB1BC` 不变。

> 注意:pyqtgraph `setConfigOptions` 是**全局**的,必须确保调用发生在 PlotWidget 实例化之前。theme 模块被 `main_validation.py` import 后立即(在 `_build_ui` 构造 PlotWidget 前)调一次即可,无需逐页设置。

---

## 8. 文件改动清单

| 文件 | 改动 |
|---|---|
| `real_validation/widgets/theme.py` | **新增**:`QSS` 常量 + `PGG_OPTS` + `configure_pyqtgraph()` + `STATE_BADGE_COLORS` 映射。纯字符串 / 仅 import pyqtgraph(可延迟),**无副作用** |
| `real_validation/main_validation.py` | `_build_ui` 各页套 QGroupBox 卡片;按钮设 objectName;顶部安全栏改色块;`app.setStyleSheet(theme.QSS)`;`configure_pyqtgraph()` 提前调用;`_refresh()` 更新 state_badge 配色 |
| `real_validation/widgets/plan_preview.py` | grid alpha 0.15(可选,微调) |
| `real_validation/widgets/camera_view.py` | grid alpha 0.15(可选) |
| `tests/` | **不改**(无 GUI 自动化测试;验证靠冒烟 + 既有测试全绿) |

---

## 9. 验证

1. 冒烟:`python -m real_validation.main_validation` → 白底界面、五页卡片、六通道安全表可编辑、规划预览白底。
2. 全量测试:
   ```bash
   python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts \
       tests.test_perception_parity tests.test_perception_quality \
       tests.test_perception_registration tests.test_perception_probe \
       tests.test_import_hygiene -v
   ```
   必须全绿(尤其 `test_import_hygiene` —— theme 绝不能进 `__init__` 闭包)。
3. 因无 GUI 自动化测试,以人工目检为准 + 断言 main() 可构造不崩(可用 `QApplication` + 立即 `closeEvent` 冒烟脚本,可选)。

---

## 10. 风险与约束

| 项 | 说明 |
|---|---|
| import 卫生 | theme 只被 `main_validation.py` import,**绝不**加进 `real_validation/__init__.py` 或 `widgets/__init__.py`(后者会连累 `from real_validation.widgets import ...` 的既有使用?——不会:widgets/__init__ 已 import PyQt5,但 theme 若 eager import pyqtgraph 不影响卫生,因卫生只测 `import real_validation` 和 `import real_validation.perception`)。为稳妥,theme 的 pyqtgraph 调用放函数内,模块级只存字符串常量 |
| 全局 pg 配置 | 若在别处(如其它脚本)创建 PlotWidget 会受影响;但 theme 只在 GUI 入口调用,不污染训练/评估脚本 |
| 内联样式清理 | 删 `setStyleSheet` 内联颜色后,行为必须由 QSS 兜住,否则按钮退回默认灰 → 逐按钮核对 |
| 兼容 | 不破坏任何现有调用签名;分支 `feat/real-data-transition`;提交前询问用户 |
