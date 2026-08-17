# 本地数据目录

将用于离线锚定的 transition NPZ 放在这里。文件至少需要：

- `positions`: `(T,3,N)` 或 `(T,N,3)`；
- `actions`: `(T,D)`，动作单位为 kPa。

实机运行产生的数据写入 `real_validation/runs/`，不要放回本目录。
