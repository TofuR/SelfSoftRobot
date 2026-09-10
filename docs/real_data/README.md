---
title: 实物数据文档索引
kind: map
status: active
updated: 2026-09-10
scope: real-data workflows, records, and legacy notes
---

# 实物数据文档索引

## 当前流程

- [验证App电气接口、颜色条件与5/10 Hz训练](validation_deployment_hardware_and_training_20260910.md)。
- [u0–u3接线实拍对照](chamber_mapping_reference_20260910.md)：逐组原始图像、压力值与六腔映射。

- `capture_setup.md`：硬件采集和动作同步。
- `general_6ch_postprocess.md`：六通道通用前处理主线。
- `automated_real_pipeline.md`：自动前处理、训练和离线验证。
- `deployment.md`：从采集到工作台的部署总览。
- `real_validation_online_workflow.md`：在线锚定、规划、执行和再观测。
- `camera_pose_robot_frame.md`：相机变化、ROI 和机器人坐标合同。

## 实验记录

- `planar_constrained_6ch_workflow.md`：六通道来源约束平面实验。
- `online_segmentation_benchmark_20260824.md`：实时分割 benchmark。
- `seq_20260819_10hz_preprocess.md`、`seq_20260819_172644_preprocess.md`：具体序列处理记录。

## 历史兼容说明

- `workflow.md`：旧 1-DOF/免标定流程，文件内已标明 Legacy；仅用于复现旧实验。

历史记录不删除，但不再作为“当前主线”入口；当前命令和路径以本页及状态页为准。
