"""segmentation.py — 薄壳：实现已移至 real_validation/perception/segmentation.py。

部署产物持有实现（工作台需要在没有仓库 src/ 的 PC 上运行同一份分割代码），
本文件保持原有公开签名不变，供离线数据准备脚本继续使用。
"""

from real_validation.perception.segmentation import (  # noqa: F401
    _clean,
    build_median_background,
    masks_to_skeletons_2d,
    segment_backlight,
    segment_bg_subtract,
    segment_color,
    segment_views,
    segment_white_on_blue,
)

__all__ = [
    "_clean",
    "build_median_background",
    "masks_to_skeletons_2d",
    "segment_backlight",
    "segment_bg_subtract",
    "segment_color",
    "segment_views",
    "segment_white_on_blue",
]
