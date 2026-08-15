"""真机硬件适配层(设计 spec §3.2)。

职责:包装 real_capture 的硬件驱动,不复制驱动实现。真机接线才 import;
Mock 流程不触碰本包。**本包不进 real_validation/__init__.py 闭包**(保持包根
stdlib-only;真机依赖 pyserial/pyrealsense2/scikit-surgerynditracker 只在
requirements-hardware.txt)。
"""
