import math

from scipy.spatial.transform import Rotation as R
from sksurgerynditracker.nditracker import NDITracker


def quaternion2euler(quaternion):
    r = R.from_quat( quaternion )
    euler = r.as_euler( 'xyz', degrees=True )
    return euler


def ndi_load(port):
    settings_aurora = {
        "tracker type": "aurora",
        "verbose": True,
        "serial port": port,
        "use quaternions": True,
        # "romfiles": ["610157.rom"]
    }
    tracker = NDITracker( settings_aurora )
    tracker.start_tracking()
    return tracker


# def get_ndi_value(tracker):
#     port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
#     for t in tracking:  # t为[[wx,wy,wz,w,x,y,z]]
#         if math.isnan( quality[0] ):  # 判断是否为空
#             return [10000, 10000, 10000, 10000, 10000, 10000, 10000]  # 为空则将其重置很大的数，在坐标轴显示不出来
#         else:
#             # # 直接保存为四元素形式，便于数据处理
#             # return [t[0, 4], t[0, 5], t[0, 6], t[0, 0], t[0, 1], t[0, 2], t[0, 3]] # 重新组合为[x,y,z,qw,qx,qy,qz]
#             # 保存为三维坐标系位姿形式
#             Rxyz = quaternion2euler( t[0, 0:4] )  # t为[[qw,qx,qy,qz,x,y,z]]
#             return [t[0, 4], t[0, 5], t[0, 6], Rxyz[0], Rxyz[1], Rxyz[2], quality[0]]  # 重新组合为[x,y,z,Rx,Ry,Rz,Quality]

def _one_pose(raw, quality):
    """把一个 Aurora tracking object 规整成 11 维位姿。"""
    import numpy as np
    arr = np.asarray(raw, dtype=float).reshape(-1)
    q = float(np.asarray(quality, dtype=float).reshape(-1)[0]) if np.size(quality) else float("nan")
    if arr.size < 7 or not math.isfinite(q):
        return [float("nan")] * 10 + [q]
    try:
        Rxyz = quaternion2euler(arr[0:4])
        return [arr[4], arr[5], arr[6], Rxyz[0], Rxyz[1], Rxyz[2],
                arr[0], arr[1], arr[2], arr[3], q]
    except Exception:
        return [float("nan")] * 10 + [q]


def get_ndi_values(tracker, count=None):
    """返回前 count 个 tracking object 的扁平位姿列表。"""
    port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
    n = len(tracking) if count is None else max(0, int(count))
    poses = []
    for i in range(n):
        if i < len(tracking):
            q = quality[i] if i < len(quality) else float("nan")
            poses.extend(_one_pose(tracking[i], q))
        else:
            poses.extend([float("nan")] * 11)
    return poses


def get_ndi_value(tracker):
    """兼容旧接口：只返回第一个 tracking object。"""
    return get_ndi_values(tracker, count=1)
