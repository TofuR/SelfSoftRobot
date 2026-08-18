# Legacy真实数据后处理

这里保存“仅末端段驱动、近端段在图像中静止”实验的特殊算法。它们依赖固定关节绝对位置
和跨帧静态共识，不适用于两段均运动、任意1–6D模型动作的通用流程。

- `static_proximal.py`：固定关节检测与静态近端段共识；
- 入口仍为`../clean_transition_npz.py`，必须显式传
  `--allow-legacy-static-proximal`；
- mask级旧共识保留在`../repair_masks.py`，同样需要显式legacy确认。

当前主线见`docs/real_data/general_6ch_postprocess.md`。
