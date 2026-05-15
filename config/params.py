"""配置文件加载工具。"""

import json
import math
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(name):
    """加载指定名称的配置文件。

    Args:
        name: 配置名，如 'camera', 'simulation', 'training'

    Returns:
        dict: 配置内容
    """
    path = os.path.join(_CONFIG_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _compute_focal(cfg):
    img = cfg["image"]
    focal_cfg = cfg["focal_length"]
    if focal_cfg["mode"] == "from_fov" or focal_cfg["value"] is None:
        fov_rad = img["fov_deg"] * math.pi / 180.0
        return 0.5 * img["width"] / (fov_rad / 2)
    return focal_cfg["value"]


def get_camera_params():
    """返回主相机参数（单相机模式使用此函数）。"""
    cfg = load_config("camera")
    primary = cfg["primary"]
    img = cfg["image"]
    rm = cfg["ray_marching"]

    return {
        "eye": tuple(primary["eye"]),
        "center": tuple(primary["center"]),
        "up": tuple(primary["up"]),
        "H": img["height"],
        "W": img["width"],
        "focal": _compute_focal(cfg),
        "near": rm["near"],
        "far": rm["far"],
        "n_samples": rm["n_samples"],
    }


def get_all_camera_params():
    """返回所有相机参数列表 [primary, extra_0, extra_1, ...]。

    每个元素: {"eye": tuple, "center": tuple, "up": tuple, "name": str}
    额外相机通过 angle_deg 绕中心旋转计算 eye 位置。
    center/up 为 null 时继承主相机。
    """
    cfg = load_config("camera")
    primary = cfg["primary"]
    primary_center = tuple(primary["center"])
    primary_up = tuple(primary["up"])

    cameras = [{
        "eye": tuple(primary["eye"]),
        "center": primary_center,
        "up": primary_up,
        "name": "primary",
    }]

    for extra in cfg.get("extra_cameras", []):
        angle_rad = math.radians(extra["angle_deg"])
        dist = extra["distance"]
        eye = (
            dist * math.cos(angle_rad),
            dist * math.sin(angle_rad),
            extra["height"],
        )
        center = tuple(extra["center"]) if extra.get("center") is not None else primary_center
        up = tuple(extra["up"]) if extra.get("up") is not None else primary_up
        cameras.append({
            "eye": eye,
            "center": center,
            "up": up,
            "name": extra.get("name", f"extra_{extra['angle_deg']}deg"),
        })

    return cameras


def get_simulation_params():
    """返回仿真参数（杆体物理属性、阻尼、时间步）。"""
    cfg = load_config("simulation")
    rod = cfg["rod"]
    damping = cfg["damping"]

    shear_modulus = rod["youngs_modulus"] / (4.0 * (1.0 + rod["poisson_ratio"]))

    return {
        "n_elements": rod["n_elements"],
        "base_length": rod["base_length"],
        "base_radius": rod["base_radius"],
        "density": rod["density"],
        "youngs_modulus": rod["youngs_modulus"],
        "poisson_ratio": rod["poisson_ratio"],
        "shear_modulus": shear_modulus,
        "start_position": rod["start_position"],
        "direction": rod["direction"],
        "normal": rod["normal"],
        "damping_constant": damping["constant"],
        "ramp_up_time": damping["ramp_up_time"],
        "dt": cfg["time"]["dt"],
    }
