"""配置文件加载工具。"""

import json
import os

_CONFIG_DIR = os.path.dirname(__file__)


def load_config(name):
    """加载指定名称的配置文件。

    Args:
        name: 配置名，如 'camera', 'simulation', 'training'

    Returns:
        dict: 配置内容（自动忽略 _doc 注释键）
    """
    path = os.path.join(_CONFIG_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def get_camera_params():
    """返回相机参数的便捷函数。"""
    cfg = load_config("camera")
    cam = cfg["camera"]
    img = cfg["image"]
    rm = cfg["ray_marching"]

    focal_cfg = cfg["focal_length"]
    if focal_cfg["mode"] == "from_fov" or focal_cfg["value"] is None:
        fov_rad = img["fov_deg"] * 3.14159265 / 180.0
        focal = 0.5 * img["width"] / (fov_rad / 2)
    else:
        focal = focal_cfg["value"]

    return {
        "eye": tuple(cam["eye"]),
        "center": tuple(cam["center"]),
        "up": tuple(cam["up"]),
        "H": img["height"],
        "W": img["width"],
        "focal": focal,
        "near": rm["near"],
        "far": rm["far"],
        "n_samples": rm["n_samples"],
    }
