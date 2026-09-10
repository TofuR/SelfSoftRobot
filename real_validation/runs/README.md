# 验证 App 运行输出

以第一页高级设置的“保存根目录”为准。当前 GUI 默认选择 `<real_validation>/runs/`；服务器正式试次应在界面设置为项目 `workspace/runs/validation/`，独立设备电脑可选择本机数据盘。

默认四页工作台每次加载模型创建 `hereditary/<时间戳_唯一标识>/`，其中每次执行再创建 `executions/<唯一标识>/`。模型加载目录保存配准、历史及规划；执行子目录保存原图、反馈图、命令/ACK、可选 NDI、时延与修订后缀。详见 [当前指南](../HEREDITARY_GUIDE.md#执行结果文件)。

旧 `run_YYYYMMDD_HHMMSS/` 结构属于 OpenLoop 兼容路线，不是当前四页目录规则。实验输出不提交到 Git；更换部署包前保留本机输出。
