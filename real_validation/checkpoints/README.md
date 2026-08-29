# 模型部署包放置约定

每个可运行模型使用一个独立目录：

```text
checkpoints/<model_name>/
  config.json
  deploy_manifest.json
  best_model.pt
```

三份文件来自同一次训练试次。`deploy_manifest.json` 声明 checkpoint 哈希、六通道来源、
模型动作通道、压力尺度、训练时基、状态坐标和节点数。毫米状态模型同时声明
`robot_diameter_mm=16`、毫米尺度、`K_safe`/认证表和规划位移统计。
当前模型合同为 `model_contract_version=2`：GL 的 `w0` 对齐当前动作，
骨架和空间 GRU 统一按 `node0=base -> nodeN-1=tip` 排列。
同一次实验生成的 `anchor.json` 与 `scene.json` 也显式保存该 `node_order`，加载时按合同校验。

在仓库根生成部署清单：

```bash
python scripts/utils/build_deploy_manifest.py \
  --exp-dir train_log/real_pipeline/<dataset>/<trial>/stages/open_loop \
  --raw-seq real_capture/data/raw/<seq> \
  --checkpoint <trial>/stages/open_loop/phase_open_loop_transition/model/best_eval_model.pt \
  --horizon-summary <openloop_horizon_summary.json> \
  --out train_log/real_pipeline/<dataset>/<trial>/deploy_manifest.json
```

随后复制到可移植工作台目录：

```bash
mkdir -p real_validation/checkpoints/<model_name>
cp <trial>/stages/open_loop/config.json \
   real_validation/checkpoints/<model_name>/config.json
cp <trial>/stages/open_loop/phase_open_loop_transition/model/best_eval_model.pt \
   real_validation/checkpoints/<model_name>/best_model.pt
cp <trial>/deploy_manifest.json \
   real_validation/checkpoints/<model_name>/deploy_manifest.json
```

GUI 在 Setup 页选择 `best_model.pt`。加载过程会向上查找同目录的配置和部署清单，并校验
`checkpoint_sha256`。当前精简运行时接受
`OpenLoopTransitionModel + fractional encoder`。
