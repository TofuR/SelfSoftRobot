# 3D Shape Visualizer Design

## Context

项目中有多种模型（SDF、MS-SCNF、CMSTNF、ODE-CMSTNF、Smooth-CMSTNF、MSTNF），它们的共同思路是：输入驱动参数 + 遍历空间点 → 得到 3D 形状。但目前没有统一的 CLI 工具来可视化这些模型的 3D 输出。现有代码在 notebook `09_model_visual_validation` 中，不适合在远程服务器上使用。

需要一个交互式 CLI 脚本，自动发现 checkpoint 和数据，查询模型，输出 HTML 交互式 + PNG 截图。

## Design

### 脚本位置

`scripts/evaluation/visualize_3d_shape.py`

### 交互流程

1. 扫描 `train_log/` 下所有 `best_model.pt`，列出供用户选择
2. 用 `model_loader.py` 自动检测模型类型并加载
3. 扫描数据目录的 npz 文件，用户选择数据文件 + 帧 index
4. 用户设置网格分辨率和阈值
5. 查询模型并可视化

### 模型查询逻辑

| 模型类型 | 输入 | 输出 | 形状提取方式 |
|---------|------|------|------------|
| SDF | `(N, 3)` coords | `(N, 1)` sdf | `marching_cubes(sdf_grid, level=0)` |
| NeRF 系列 | 网格点 reshape 为 rays | `(N, 2)` [vis, dens] | 密度阈值过滤点云 |
| MS-SCNF | 同 NeRF | 同 NeRF + skeleton | 密度阈值 + GT skeleton 叠加 |

### 输出

- `output/visualize/{model_type}_{exp_name}_frame{idx}.html` — Plotly 交互式
- `output/visualize/{model_type}_{exp_name}_frame{idx}.png` — PyVista offscreen

### 依赖变更

- 新增: `plotly`, `scikit-image`
- 已有: `pyvista`, `torch`, `numpy`

### 需要修改的现有文件

- `src/utils/model_loader.py` — 扩展 `_detect_model_type()` 支持 SDF 模型

### 核心函数

```
scan_checkpoints()      → List[{path, name}]
scan_data_files()       → List[{path, name}]
interactive_select()    → 终端菜单
query_sdf_model()       → mesh (vertices, faces)
query_nerf_model()      → point cloud (N, 3)
export_html()           → Plotly HTML
export_png()            → PyVista PNG
main()
```
