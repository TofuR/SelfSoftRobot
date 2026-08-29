# real_validation 在线规划与控制工作流

## 1. 两份配置各自负责什么

模型部署包由 `checkpoint + config.json + deploy_manifest.json` 组成，声明模型结构、
动作通道、压力尺度、训练时基、状态坐标、`K_safe` 和位移统计。GUI 加载时校验
`checkpoint_sha256`，确保三份文件属于同一次训练试次。

每次真实验证实验在 `run_*/validation_setup.json` 保存当前现场配置：

- 相机 backend、主相机索引和 serial/身份；
- 源图尺寸与 `roi_xywh=[x,y,w,h]`；
- 在线分割方法、完整参数、参考背景路径和哈希；
- 机器人直径 `16 mm`、两段长度比例；
- 首次 Anchor 得到的 mask 面积参考和相机像素到机器人毫米坐标变换；
- 当前 checkpoint 哈希和配置身份。

训练数据的 mask 来源记录模型如何获得监督骨架。在线感知配置记录当前现场如何从
相机帧得到同定义的 15 节点骨架。两者通过节点顺序、端点修正、坐标单位和质量门保持一致。

## 2. 实际操作闭环

```text
加载部署包
    ↓
连接主相机 → 保存/加载当前空场背景 → 框选或自动建议 ROI → 确认现场配置
    ↓
当前帧分割 → 两段整臂中心线 → 15 节点 → 首次 Anchor 建立毫米坐标
    ↓
在相机画面点击毫米目标和障碍 → 自动 K → OpenLoop shooting → Preflight
    ↓
Arm → 逐拍下发 → ACK applied6 更新动作历史
    ↓
机器人稳定 → 再观测 → 真实目标残差 + 前向预测误差 → 下一窗口 Anchor
```

### 2.1 当前相机如何获得 ROI

1. 相机安装完成后新建 run，并选定主相机。
2. 选择当前现场的在线分割方法。
3. 白色机器人/蓝背景场景先保存一张空场背景。相机、支架、光源和曝光随后保持固定。
4. 点击“框选 ROI”，在右侧完整相机画面上拖动正方形，使两段机器人在全部计划形态下均位于框内。
5. 机器人已经出现在画面时，可以点击“当前 mask 自动建议”。程序按当前 mask 包围盒和约
   3 个机器人直径的余量产生方形候选，候选仍可拖动调整。
6. 点击“确认 ROI”。配置写入本次 run，绿色边框表示正式 ROI。

ROI 改变会生成新的现场配置身份，并使已有相机 Anchor 和 Plan 失效。切换主相机、相机断开
或模型变化时，GUI 引导重新确认现场配置。

### 2.2 ROI 坐标如何进入模型

ROI 局部点 `p_roi=(x_roi,y_roi)` 先恢复为源相机像素：

`p_camera = p_roi + (roi_x, roi_y)`。

首次 Anchor 使用机器人 mask 的 16 mm 实体直径估计 `pixels_per_mm`，以基座为原点、
基座指向末端为轴向建立 `SkeletonFrameTransform`：

`p_robot_mm = Rᵀ (p_camera - p_base_camera) / pixels_per_mm`。

该变换在一次 run 内固定复用。相机画面的目标、障碍、预测轨迹和执行后骨架都通过同一变换
正反映射。ROI 仅影响图像处理范围，不改变机器人毫米坐标的定义。

相机重新安装后，新 run 的首次 Anchor 会吸收新的图像平移、平面旋转和尺度。后续 Anchor
使用首个 Anchor 全图参考帧做注册检查，相机位姿变化超过部署阈值时拒绝更新状态。

### 2.3 在线骨架与两段机器人

在线 mask 使用 `skeletonize → 最长主路径 → tip/base 端帽修正 → 分段弧长重采样`。
它支持 S 形和局部水平形态，并保持 `node0=base`、`node14=tip`。默认两段长度比例为
`(1,1)`，15 节点形成 14 个区间，每段 7 个区间，连接处为 node7。模型仍接收一条完整
15 节点骨架；分段信息用于稳定节点分配和后续引入多段先验。

每次正式 Anchor 自动保存：

- `camera_frame.png`：源相机帧；
- `camera_roi_overlay.png`：源相机帧与正式 ROI；
- `roi_frame.png`：ROI 输入；
- `mask.png`：在线分割结果；
- `segmentation_stages/`：white、moved、gated、morph、pretrim、final；
- `skeleton_overlay.png`：15 节点叠图；
- `quality.json`：mask 面积、端点修正、注册和判定。

目录位于 `run_*/perception/observations/anchor_*/`，可以直接判断偏差来自 ROI、分割、
中心线还是坐标变换。

## 3. 规划与执行接口

### 3.1 Anchor

Anchor 包含当前 15 节点形态和模型需要的最近 H 步动作。动作历史优先使用阀控制器 ACK
返回的 `applied6`，再按 `channel_source6/channel_map` 投影到模型动作维度。显式零历史起步
会在 Anchor 中标记 OOD，后续逐拍 ACK 会滚动替换零填充。

### 3.2 场景

画面点选在毫米模型加载后需要先完成相机 Anchor。这样每个 `target_point`、
`target_skeleton` 和 `obstacle_circle` 在写入 `scene.json` 前已经位于 model/mm 坐标。
数值输入框同样直接输入模型单位。

目标点控制 nodeN-1。完整形态目标控制全部 15 个对应节点。场景保留一个活动目标和多个障碍，
Planner 对所有障碍累计碰撞代价。

### 3.3 自动 K

自动 K 先计算当前 Anchor 与目标的形态差距：

- 点目标：末端到目标区域边界的距离；
- 骨架目标：各对应节点距离的最大值减去容差。

然后在部署清单的 `planning_displacement_p95[K]` 中选择覆盖该差距的最小 K，并同时受 GUI
上限和 `K_safe` 约束。距离超过单窗口统计范围时使用当前安全上限，执行后重新观测并规划下一窗口。

### 3.4 Preflight 与执行

Preflight 校验 checkpoint、Anchor、scene、安全配置、动作维度、通道来源、压力范围、变化率、
训练时基、`K_safe` 和预测障碍间距。通过后才允许 Arm。

执行器逐拍发送六通道 kPa，并等待 ACK。历史缓冲只记录最终 `applied6`。中止、归零和相机配置
变化都会按会话状态机使旧计划失效。

## 4. 执行后结果如何解释

执行完成后等待机器人稳定，再点击“采集执行后形态并评价”。输出分为三类：

1. `observed_control_result`：执行后真实相机骨架到目标的距离，以及观测骨架与障碍的实际间距。
   Real 相机来源时，这是本次真实形态控制结果。
2. `forward_model_prediction_error`：Planner 预测终态与执行后观测骨架的全节点均值和末端误差。
   它评价前向模型在该动作序列上的预测准确度。
3. Planner 的预测目标残差：动作下发前的模型内优化结果，用于候选计划筛选。

Mock 相机输出用于验证数据流、坐标和落盘接口。结果保存为：

- `post_control_metrics.json`；
- `post_control_overlay.png`，红色为观测，青色为预测；
- `anchor_post_execution.json`；
- `perception/observations/post_*/` 的逐阶段感知文件。

执行后观测会成为下一规划窗口的 Anchor，因此长距离任务可以按
“规划一个安全窗口 → 执行 → 再观测”循环推进。

## 5. 操作门与问题定位

| 界面提示 | 检查内容 |
|---|---|
| 当前相机 ROI 与参考背景 | 保存当前空场背景、框选并确认 ROI |
| 当前相机身份不一致 | 检查主相机索引和 RealSense serial |
| 当前模型与 ROI 配置不一致 | 按新 checkpoint 重新确认现场配置 |
| 注册位移超限 | 固定相机与支架，随后新建 run 并建立新坐标 |
| mask 面积超限 | 查看本次 observation 的 ROI、mask 和 overlay |
| K_safe / 时基门 | 重新导出当前 checkpoint 的 deploy_manifest |
