# 历史本地数据目录（已废弃）

正式离线锚定数据已改由项目 dataset registry 选择，本目录不再接收 NPZ 副本。
独立部署时通过 GUI 文件选择器选择外部文件。文件至少需要：

- `positions`: `(T,3,N)` 或 `(T,N,3)`；
- `actions`: `(T,D)`，动作单位为 kPa。

实机运行数据写入 GUI 选择的“保存根目录”，默认四页的结构见 [运行目录说明](../runs/README.md)，不要放回本目录。上面的 transition NPZ 只用于旧 OpenLoop 离线 Anchor，不是当前四页 Hereditary 部署必需数据。
