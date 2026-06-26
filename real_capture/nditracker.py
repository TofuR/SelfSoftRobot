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

def get_ndi_value(tracker):
    port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
    for t in tracking:  # t为[[wx,wy,wz,w,x,y,z]]
        if math.isnan( quality[0] ):  # 判断是否为空
            return [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]  # 为空则将其重置很大的数，在坐标轴显示不出来
        else:
            # 保存为三维坐标系位姿形式 + 四元数
            Rxyz = quaternion2euler( t[0, 0:4] )  # t为[[qw,qx,qy,qz,x,y,z]]
            return [t[0, 4], t[0, 5], t[0, 6], Rxyz[0], Rxyz[1], Rxyz[2], t[0, 0], t[0, 1], t[0, 2], t[0, 3], quality[0]]  # 重新组合为[x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,Quality]