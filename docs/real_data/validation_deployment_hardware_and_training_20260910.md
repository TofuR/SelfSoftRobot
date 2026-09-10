# 验证App：阀电气接口、颜色条件与5/10 Hz权重

核对日期：2026-09-10。这里描述当前 `real_validation` 的 Hereditary/Analytic B 入口，旧 `checkpoints/current` 中的 OpenLoop checkpoint 不代表当前HOV2.2部署权重。

## 1. 现在已经是电流控制

软件链路为：**GUI压力kPa → 六通道限幅/限速 → USB-RS485 Modbus RTU → 模拟量输出 → 4–20 mA比例阀 → 机器人腔体**。源码按华控电子寄存器表实现；仓库资料没有可靠记录到比例阀完整订货型号，不能仅由代码确定实际阀铭牌。

| 参数 | 当前实现 |
|---|---|
| 分组 | 2组，每组3路；c0–c2为组1，c3–c5为组2 |
| 模拟量 | 两组均为4–20 mA；0–500 kPa对应4000–20000 |
| 换算 | `I_mA = 4 + 16 * P_kPa / 500`；`register = int(4000 + 32 * P_kPa)（正数截断）` |
| 示例 | 0 kPa→4 mA→4000；100 kPa→7.2 mA→7200；150 kPa→8.8 mA→8800 |
| 寄存器 | 两组均使用0x000A、0x000B、0x000C；组别由串口区分 |
| 通信 | 默认9600、8N1、从站1；单路0x06，多路0x10，CRC16 |
| 当前模型实验范围 | 0–150 kPa，并与GUI每路上下限/速率取交集；500 kPa只是底层接口量程 |
| 数据含义 | 保存的是下发/ACK确认值；没有独立实测腔压，因此不能把它叫实测气压闭环 |

实际调用在[modbus.py](../../real_validation/hardware/modbus.py)的 `ModbusManager.set_pressure` 和 `set_all_pressures`，均使用 `ModbusRTU.pressure_to_register_value_current`。同文件的 `pressure_to_register_value_voltage` 是旧0–5 V兼容函数，当前下发不调用它。采集App的对应实现为[modbus_manager.py](../../real_capture/modbus_manager.py)。

### 换设备要改什么

- **换成同协议、同0–500 kPa量程的4–20 mA设备**：当前换算本身无需改。确认硬件输出模块已配置为电流输出，并核对串口、从站、寄存器及接线。软件发送的Modbus数值不会自动把物理电压输出模块变成电流输出模块。
- **仍是4–20 mA，但压力范围/寄存器单位不同**：在 `ModbusRTU` 修改 `P→I→register` 和逆换算，按设备说明书更新 `ModbusManager.channel_registers`；通用公式为 `I=4+16*(P-Pmin)/(Pmax-Pmin)`。同时核对 `valve.py` 的硬件范围与GUI限制、导出包的动作物理尺度和允许范围。训练动作始终应表示kPa，不把电流数字作为模型动作。
- **换成另一种通信协议/模拟量板卡**：实现与 `ValveController` 相同的下发、ACK、连接、最后下发值和归零接口，再由硬件管理器接入；无需为此改p/h模型和规划目标函数。
- **更换量程或支持负压**：不能只改一个寄存器公式。当前App/模型数据域和停止归零逻辑仍按正压实现，必须共同更新并重新确认训练/执行合同。

## 2. 背景和遮挡物颜色确实重要

当前在线反馈并没有运行SAM2。SAM2用于离线训练标签；在线[partial_edges.py](../../real_validation/perception/partial_edges.py)沿预测臂身两侧搜索浅色内部到较暗外部的边缘，包含明确阈值：灰度差≥18、法向梯度≥6、梯度方向一致性≥0.65，臂内HSV饱和度≤135且亮度≥95。

| 场景 | 可能后果 | 前期部署建议 |
|---|---|---|
| 浅色臂、均匀深色哑光背景 | 更容易得到可信边缘 | 当前优先使用；蓝色不是硬编码必需，关键是明暗/颜色对比 |
| 浅色背景或与臂同色遮挡物 | 真实轮廓对比不足，遮挡边缘可能被误当作臂边缘 | 避免作为首次实机配置；专门列为后续鲁棒性测试 |
| 黑色/深色遮挡物 | 通常使对应边缘缺失，但遮挡物边界仍可能误关联 | 可以作为可重复起步条件，不能假设“黑色就一定可靠” |
| 花纹、线缆、支架、反光 | 多个候选或错误对应，误校正比漏检更危险 | 避免同色粗线穿过臂身；固定曝光和柔和照明，检查反馈覆盖与残差 |
| 黑色剪影、亮背景 | 初始化可选dark，但当前在线边缘仍按bright处理 | 初始化明暗选项不等于整套反馈极性切换；当前闭环部署优先亮臂暗背景 |

初始化[initial_shape.py](../../real_validation/perception/initial_shape.py)是全图阈值/连通域提取，可能被白色支架或大块亮区域干扰；失败时使用手动中心线并确认，不能因此省略后续边缘质量检查。无需手画遮挡区域，但程序识别的是“可信边缘减少/丢失”，不是理解任意遮挡物类别。

本次新权重学习的是气压历史→几何形态，**扩大这部分训练数据不会自动修好颜色阈值前端**。若实验必须更换复杂背景，主要应改进在线边缘/外观关联和初始化，而不是仅重训动力学模型。

## 3. 现有权重的数据规模

此前工作台验证使用 `package_20260909_000/hereditary.npz`，源checkpoint为 `hov22_local14_unit_balanced_s42_20260905_000/phase_hereditary_geometry/model/best_eval_model.pt`，SHA256为 `b7fdb9b1d2493c98d15e67bdb6e228c8bc4dd05d524d2598f04fbc082c02d501`。

- 原生10 Hz；2条采集序列：`182253`、`182519`。
- 实际fit为728+1967＝**2695帧**；dev存储755帧，其中675帧为评分区，其余为旧协议上下文；该旧上下文与fit有重叠，不能称独立序列验证。
- 模型564参数、32维历史状态、15节点；最大训练200 epoch，batch64；所选checkpoint来自验证指标而非任意“最后一次”。

新训练从原始独立序列重新组织，不从旧2695帧fit继续追加，也不把同一批数据的多个派生版本重复计数。

## 4. 本次全平台、原生频率训练

当前独立study：[`hov_full_native_20260910_001`](../../workspace/runs/training/hov_full_native_20260910_001/)。

| 频率 | 同平台原始序列 | 原始图像/动作行 | 处理与分配 |
|---|---|---:|---|
| 5 Hz | 172644、181044、181548、183351、183526、183547、183740、184036 | 17479 | 已补齐6条动态序列的SAM2/骨架；39帧全零序列183526保留校准资料，不足以组成训练/验证历史窗口 |
| 10 Hz | 182253、182519 | 4211 | 合并两条原始序列各自已有train/val，重新分配，不丢掉原先未参与fit的时段 |

这里“全量”指全部同平台、同采样率的合格序列，并保留验证数据：各动态序列前80%训练，隔开一个历史窗口后余下部分验证。10 Hz为3368训练/763验证/80隔离帧；5 Hz为13950训练/3350验证/140隔离帧（另39帧仅校准）。来自同次采集，证据等级仍是 `within_sequence`。既有“test”标签在本次全量部署开发中不再提供独立测试资格，正式科学结论要用新的独立实机试验。

不将5 Hz插值成10 Hz，也不随意抽取10 Hz每隔一帧来丢弃中间动作。将各序列原来的归一化动作先还原到kPa，再统一按150 kPa存储；用同一采集日相机基点(370,110)、1.25 px/mm的固定坐标变换重新组织形状，避免把不同序列自动估计的轴方向误差当作动力学。

训练均使用HOV2.2无残差、局部14角+2长度、2 play+6 Maxwell、seed42、Adam lr=0.001、训练300 epoch、batch256。5 Hz的history/episode为20步，10 Hz为40步，均为4 s；dt分别0.2/0.1 s，Maxwell时间常数也按对应dt创建。每5 epoch验证一次，以完整验证序列的node_mean_mm选权重；关闭早停，完整训练300 epoch，仍保留学习率调度。较大的batch主要减少小模型调度开销，实际效果仍由验证曲线判断。

本次按用户要求停止旧任务后从头重训，不恢复旧checkpoint。旧study `hov_full_native_20260910_000` 的数据处理已全部完成，5 Hz训练已停止、10 Hz旧200 epoch训练已完成，旧日志与权重保留；新study通过 `--datasets-from` 复用经哈希核对的数据。

### tmux、状态与最终加载

```bash
tmux attach -t hov_full_native_20260910
cat workspace/runs/training/hov_full_native_20260910_001/status_5hz.json
cat workspace/runs/training/hov_full_native_20260910_001/status_10hz.json
```

会话执行全流程脚本，训练细节在study内 `training_5hz.log` / `training_10hz.log`，数据和SAM2日志复用旧study `hov_full_native_20260910_000`，其中SAM2日志在 `preprocessing/`。连接tmux后可用Ctrl-b、d脱离；不要用Ctrl-C结束训练。代码为[train_full_hereditary_deployment.py](../../scripts/experiments/train_full_hereditary_deployment.py)，命令、数据哈希和选择结果均落盘。

每种频率只有在预处理检查、训练、导出和加载检查通过后，才将状态写为 `complete`；失败写为 `failed` 并保留原因，不悄悄减小训练语料。本次两种频率均已完成300 epoch及导出。5 Hz按验证集选择第150轮（node_mean_mm=1.714694），10 Hz选择第300轮（1.342242）；验证集合不同，不能据此直接比较频率优劣。候选副本已放入 `real_validation/checkpoints/candidates/`，旧权重保留。

预期完成路径（完成前可能不存在）：

```text
hov_full_native_20260910_001/
  deploy_5hz/hereditary.npz + hereditary.json
  deploy_10hz/hereditary.npz + hereditary.json
  real_validation_5hz.zip
  real_validation_10hz.zip
```

在第一页“选择模型”选择对应频率的 `hereditary.npz`，同名JSON必须放在旁边；模型dt会随包加载。也可解压相应zip使用默认带模型的App。不要只手改JSON的dt来把5 Hz权重冒充10 Hz权重。模型加载不发送气压；仍按自己设定的初始气压进行形状对齐和模型预热。

## 5. 腔道与实拍对照

见[四组动作的接线实拍对照](chamber_mapping_reference_20260910.md)，或下载[单文件HTML图册](../../workspace/reports/validation_deployment_20260910_000/chamber_reference.html)。每组给出原图、序列、帧号、六路下发压力。默认 `c0,c1,c2,c3,c4,c5 ← u0,u1,u1,u2,u2,u3`；成对驱动的两腔无法从现有组级图像单独区分。
