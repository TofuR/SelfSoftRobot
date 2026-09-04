# Hereditary v2 训练与验证记录

更新日期：2026-09-03

## 1. 目标与结论边界

Hereditary v1 在训练 episode 中从静息状态烧入，而连续链式评估会让慢
Maxwell 状态逐渐热化。对于 `episode_len=40`、`dt=0.1 s` 和
`tau_max=10 s`，训练利用了慢模态的冷亏损作为伪静态容量，造成训练与长链式
推理的状态分布失配。v2 首先消除这一初始化伪影，再把算子容量放到当前数据可
辨识的时间带内。

代码冒烟检查只能说明实现可运行。性能改善、residual 容量以及 play/Maxwell
定量谱都必须以 v2 重新训练后的 checkpoint 为准。

## 2. v2 修改合同

| 项目 | v1 | v2 默认值 | 目的 |
|---|---|---|---|
| `burnin_mode` | `rest`（旧配置缺省语义） | `equilibrium` | 在窗口首动作 `a0` 处令 `p=h=e(a0)`，使 `q=d=0` |
| `tau_max` | 10.0 s | 2.0 s | 6 个 Maxwell 元素全部落入 0.3--2.0 s 可辨识带 |
| `n_play` | 8 | 2 | 保留率无关迟滞的最小可证伪容量 |
| `residual_scale_max` | 固定 0.3 | 参数化，默认 0.3 | 重训后检查 residual 是否仍受上限约束 |

新的 Maxwell 网格预期为：

```text
[0.30, 0.44, 0.64, 0.94, 1.37, 2.00] s
```

旧 checkpoint 没有 `burnin_mode` 时必须按 `rest` 加载，以保留其训练协议；
新 checkpoint 必须在 `config.json` 中显式记录所有上述字段。

## 3. 首次 v2 主训练

本轮固定随机种子 42，并显式给出全部 v2 参数，避免未来默认值变化影响复现。
通用训练参数由当前配置解析为 `n_epochs=500`、`batch_size=4`、`lr=0.001`；
这些解析后值也保存在实验 `config.json` 中，后续公平重训应显式保持一致。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_transition.py \
  --mode hereditary \
  --data_dir data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/train \
  --episode_len 40 \
  --dt 0.1 \
  --n_play 2 \
  --n_maxwell 6 \
  --tau_max 2.0 \
  --burnin_mode equilibrium \
  --residual_scale_max 0.3 \
  --n_epochs 500 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42 \
  --experiment-dir train_log/hereditary/exp_20260831_000
```

运行位置：

- tmux 会话：`hereditary_v2_main_s42`
- 实验目录：`train_log/hereditary/exp_20260831_000`
- 终端日志：`train_log/hereditary/exp_20260831_000/session.log`

查看方法：

```bash
tmux attach -t hereditary_v2_main_s42
tmux capture-pane -pt hereditary_v2_main_s42 -S -80
tail -f train_log/hereditary/exp_20260831_000/session.log
```

## 4. 训练前验证状态

执行命令：

```bash
python -m unittest tests.test_hereditary_model tests.test_hereditary_operators -v
```

截至 2026-08-31：29 项全部通过。`test_episode_rollout_matches_manual_stepping`
已分别覆盖 `equilibrium` 与 `rest`：前者的独立手工参照从首个历史动作的
`p=h=e(a0)` 平衡态开始，后者从零状态开始；两者随后消费相同的剩余历史与
episode 当前动作，并与 trainer 的线程化状态逐项一致。

## 5. 后续消融训练矩阵

所有实验保持数据、训练轮数、batch size、学习率、checkpoint 选择规则和 seed
一致，每次只改变表中字段。

| 标签 | `burnin_mode` | `tau_max` | `n_play` | residual max | 问题 |
|---|---:|---:|---:|---:|---|
| v1-retrain | rest | 10.0 | 8 | 0.3 | 公平复现 v1 |
| F1-only | equilibrium | 10.0 | 8 | 0.3 | 单独验证平衡初始化 |
| F1+F2 | equilibrium | 2.0 | 8 | 0.3 | 验证可辨识时间带 |
| v2-main | equilibrium | 2.0 | 2 | 0.3 | v2 主模型 |
| F5-capacity | equilibrium | 2.0 | 2 | 0.5 | residual 容量对照 |

先用 seed 42 完成筛查；若主结论成立，再对关键配置使用 42、43、44 三个 seed，
报告均值、标准差和配对差异。

## 6. 评估协议

“冷重启”必须写明具体操作。建议同时保留以下三项：

1. **连续链式状态**：序列起点初始化一次，随后连续传递 operator state。当前
   `eval_real_quant.py --mode gt` 对 hereditary 的实际含义是连续状态；模型不读取
   传入的真实 skeleton。
2. **K=40 窗口重播种**：每 40 步从记录动作窗口重新初始化，窗口内连续传递状态。
   对应 `--mode open_loop --window-len 40`，是 windowed OpenLoop 的主要部署比较。
3. **逐帧冷重启**：每一帧都从其动作历史重新 burn-in，用于测量严格匹配 episode
   初始化分布的局部能力。旧分析由临时脚本完成，正式复验前需固化为仓库脚本。

定量评估：

```bash
CKPT=train_log/hereditary/exp_20260831_000/phase_hereditary/model/best_model.pt

python scripts/evaluation/eval_real_quant.py \
  --checkpoint "$CKPT" \
  --data_dir data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/val \
  --mode gt --no-ndi

python scripts/evaluation/eval_real_quant.py \
  --checkpoint "$CKPT" \
  --data_dir data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/val \
  --mode open_loop --window-len 40 --no-ndi
```

谱与叠加可视化：

```bash
python scripts/evaluation/plot_hysteresis_spectrum.py --checkpoint "$CKPT"

python scripts/evaluation/visualize_real_overlay.py \
  --checkpoint "$CKPT" \
  --data_dir data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/val
```

## 7. 核心判读标准

### F1：状态热化

- 连续链式误差不再像 v1 一样从约 1.68 mm 单调升至约 7.05 mm；
- 后半序列误差显著下降；
- 连续链式与逐帧冷重启/K=40 重播种的协议差距显著缩小；
- base 节点的异常漂移随之消失。

约 1.3 mm 是根据 v1 冷重启结果提出的预期值，不作为实现正确性的硬阈值。

### F2/F3：时间带与谱

- 重新报告各 Maxwell/play 分量的实际输出 RMS、LOO 误差增量和跨 seed 稳定性；
- 旧 `rest + tau_max=10 s` 谱含伪静态 aliasing，只能用于根因对照；
- v2 谱稳定后才讨论 play/Maxwell 的定量份额；
- 物理上的率无关/率相关区分仍需独立变速率实验裁决。

### F4：play 容量

`n_play=2` 相对 `n_play=8` 若验证误差无显著退化，且 play 的 realized
contribution/LOO 增量继续很小，才支持缩减 play bank。

### F5：residual 容量

检查值：

```python
from src.utils.model_loader import load_model

info = load_model(
    "train_log/hereditary/exp_20260831_000/phase_hereditary/model/best_model.pt",
    device="cpu",
)
m = info["model"]
print("residual_scale =", m.residual_scale.item(), "/ max", m.residual_scale_max)
```

若 0.3 配置仍达到上限，再结合 0.5 对照是否降低独立验证误差判断容量需求。
仅参数触顶表示优化压力；“触顶且放宽后稳定改善”才是更强的容量证据。

## 8. 结果边界

当前 val 是同一真实采集序列的留出尾段，可用于验证 F1 根因和模型内消融。
论文中的跨轨迹泛化与物理谱结论还需要独立序列、变速率/保持段及未参与选择的
test split。记录骨架对比是前向模型误差；它不等同于真实机器人控制成功率。

## 9. 并行消融批次（2026-08-31）

2026-08-31 02:49（Asia/Shanghai）在主训练继续占用物理 GPU 0 的同时，使用
物理 GPU 1、2、3 启动六个互不写同一目录的训练。所有新任务与主训练保持
`n_epochs=500`、`batch_size=4`、`lr=0.001`、`episode_len=40`、`dt=0.1`、
`n_maxwell=6` 和相同数据划分；每张 GPU 并行两个任务。

| tmux 会话 | 物理 GPU | 实验目录 | 标签 | 与配对实验相比只改变 |
|---|---:|---|---|---|
| `h2_v1_s42_g1` | 1 | `exp_20260831_001` | v1-retrain, seed 42 | v1 完整协议基线 |
| `h2_f1_s42_g1` | 1 | `exp_20260831_002` | F1-only, seed 42 | 相对 `_001` 仅 `rest -> equilibrium` |
| `h2_f1f2_s42_g2` | 2 | `exp_20260831_003` | F1+F2, seed 42 | 相对 `_002` 仅 `tau_max: 10 -> 2 s` |
| `h2_f5_s42_g2` | 2 | `exp_20260831_004` | F5-capacity, seed 42 | 相对 `_000` 仅 residual max `0.3 -> 0.5` |
| `h2_v2_s43_g3` | 3 | `exp_20260831_005` | v2-main, seed 43 | v2 跨 seed 复现 |
| `h2_v2_s44_g3` | 3 | `exp_20260831_006` | v2-main, seed 44 | v2 跨 seed 复现 |

由此形成四组直接配对：F1 用 `_001` 对 `_002`，F2 用 `_002` 对 `_003`，
F4 用 `_003` 对 `_000`（仅 `n_play: 8 -> 2`），F5 用 `_000` 对 `_004`；
v2 稳定性使用 `_000/_005/_006` 的 seeds 42/43/44。不要用跨多项变化的两次
训练替代这些配对比较。

启动器为：

```text
scripts/experiments/run_hereditary_v2_experiment.sh
```

它拒绝复用已经存在的实验目录，并在每个目录写入 `run_manifest.txt`、
`config.json`、`session.log` 和 checkpoint。训练成功结束后自动调用：

```text
scripts/experiments/evaluate_hereditary_v2_checkpoint.sh
```

自动评估分别写入 `evaluations/continuous/`、`evaluations/window40/` 和
`evaluations/spectrum/`，并生成 `evaluations/residual_scale.txt`；全部成功后才
创建 `evaluations/COMPLETE` 和实验根目录的 `RUN_COMPLETE`。现有 `_000` 主训练
由 `h2_eval_v2_s42` 等待，并在完整训练结束后运行同一评估包。

启动后核验显示六个主进程均正确绑定目标 GPU，每个训练进程约占 338 MiB；
双开后 GPU 1/2/3 各约占 705 MiB，初始利用率约为 82%/89%/75%，未出现显存压力。
为避免每 batch 的 tqdm 刷屏放大日志，新任务设置 `TQDM_DISABLE=1`，但每 epoch
汇总、CSV、checkpoint 和错误信息仍完整保留。

查看全部会话：

```bash
tmux list-sessions | grep -E 'h2_|hereditary_v2_main_s42'
tmux attach -t h2_f1_s42_g1
tmux capture-pane -pt h2_f1_s42_g1 -S -80
tail -f train_log/hereditary/exp_20260831_002/session.log
```

## 10. 2026-09-03 同 fit/dev、同选模规则对照

为与 ISHSM v4 比较，Hereditary v2 使用同一 fit/dev、seed 42、100 epoch 上限和聚合 dev
`validation.node_mean_mm` 选模，在 GPU3 新目录从 epoch 1 干净重训：

```text
workspace/runs/training/ishsm_v4_formal/hereditary_v2_equilibrium_s42_gpu3_restart
```

正式 `best_eval_model.pt` 位于 epoch 100，continuous 为
`1.7818 / 2.9004 mm`，cold-restart-40 为 `1.8331 / 3.0527 mm`
（node / endpoint）。相同 dev 上，ISHSM v4 zero-init 为 `1.7486 / 3.2866 mm`：配对
bootstrap 对 node 的差异不确定，Hereditary endpoint 则稳定改善约 `0.3862 mm`。

F1 的旧式单调链式爆炸没有重现；continuous 相对 cold-restart-40 的 endpoint 稳定改善约
`0.1523 mm`。F5 仍触发：`residual_scale_raw=0.3060`，前向有效值被 clamp 在
`0.3000/0.3000`。由于 epoch 100 仍刷新验证最优，这些数值是预算截断结果；下一步需延长训练，
并用 `residual_scale_max=0.5` 单变量重训区分训练不足与真实 residual 容量需求。

完整协议、置信区间和 ISHSM 消融见 `ishsm_optimization_validation_record.md`。谱图位于：

```text
workspace/runs/analysis/ishsm_v4/hereditary_spectrum_s42_epoch100/
```

其中原始权重的 `play/maxwell=0.10/0.90` 只能作为模型内部诊断，不能直接解释为材料迟滞份额；
定量物理结论仍需 realised contribution、LOO、跨 seed 和独立变速率实验。

## 11. 2026-09-03 200 epoch 与 residual 容量消融

在同一 fit/dev、seed 42 和选模规则下，将 Hereditary v2 的训练上限延长到 200 epoch，并只改变
`residual_scale_max`：

| residual 上限 | continuous node / endpoint | cold-restart-40 node / endpoint | raw / effective scale |
|---:|---:|---:|---:|
| 0.3 | 1.4344 / 2.4665 | 1.4458 / 2.4971 | 0.3060 / 0.3000 |
| 0.5 | **1.3366 / 2.3704** | **1.3467 / 2.3765** | 0.5004 / 0.5000 |

0.5 相对 0.3 在 continuous 的 node/endpoint 分别改善 `0.0978 mm
[0.0751, 0.1241]` 和 `0.0961 mm [0.0385, 0.1514]`；cold-restart-40 的改善方向一致。
这确认 F5 是当前开发集上的容量信号。由于 0.5 仍然钉住上限，不能仅据此无限放宽 residual；
最终值需要在冻结配置后的新独立轨迹上裁决。

机器汇总：`workspace/runs/analysis/ishsm_v5_200/analysis_summary.json`。完整 ISHSM 对照、证据边界
和当前模型选择见 `ishsm_optimization_validation_record.md`。
