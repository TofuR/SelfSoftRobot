# 模型放置约定

默认模型目录为 `checkpoints/current/`：

```text
checkpoints/current/
  config.json
  best_model.pt
```

仓库不提交二进制 checkpoint。部署到 PC 前，将服务器当前模型复制为：

```text
服务器:
train_log/open_loop_transition/exp_20260714_8/
  config.json
  phase_open_loop_transition/model/best_model.pt

PC 工作台:
real_validation/checkpoints/current/
  config.json
  best_model.pt
```

本目录自带的 `config.json` 对应 `exp_20260714_8`。如果替换 checkpoint，必须同时替换
它所属实验的 `config.json`，不得混用。当前精简运行时仅接受
`OpenLoopTransitionModel + fractional encoder`。
