"""
elastica_env.py — 基于 PyElastica 的软体机械臂仿真环境。

提供两种使用模式：
  1. 静态采集：`create_simulation` + `get_simulation_data_pair`（每次独立仿真）
  2. 连续仿真：`ContinuousSoftArmEnv`（保持仿真状态，逐步推进）
"""

import numpy as np
import pyvista as pv
import cv2
from collections import defaultdict
from tqdm import tqdm

# --- PyVista 全局设置 ---
pv.set_plot_theme("document")
pv.OFF_SCREEN = True

# --- PyElastica 模块导入 ---
from elastica.modules import BaseSystemCollection, Constraints, Forcing, CallBacks, Damping
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import NoForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

# =============================================================================
# 物理参数常量
# =============================================================================
N_ELEMENTS = 30            # 杆体离散单元数
ROD_LENGTH = 0.5           # 杆体长度 (m)
ROD_RADIUS = 0.015         # 杆体半径 (m)
ROD_DENSITY = 1000.0       # 密度 (kg/m³)
YOUNGS_MODULUS = 1e6        # 杨氏模量 (Pa)
POISSON_RATIO = 0.5        # 泊松比
SHEAR_MODULUS = YOUNGS_MODULUS / (4.0 * (1.0 + POISSON_RATIO))

ROD_START_POS = np.zeros(3)
ROD_DIRECTION = np.array([0.0, 0.0, 1.0])
ROD_NORMAL = np.array([1.0, 0.0, 0.0])

DAMPING_CONSTANT = 0.1     # 阻尼系数
RAMP_UP_TIME = 0.5         # 扭矩渐升时间 (s)

# =============================================================================
# 渲染 / 相机参数常量
# =============================================================================
DEFAULT_IMAGE_SIZE = (100, 100)
CAMERA_EYE = (1.5, 0.0, 0.5)
CAMERA_CENTER = (0.0, 0.0, 0.25)
CAMERA_UP = (0.0, 0.0, 1.0)


# =============================================================================
# 自定义力与回调
# =============================================================================

class SimpleDistributedTorque(NoForces):
    """沿杆施加均匀分布扭矩。"""

    def __init__(self, torque_profile, ramp_up_time=RAMP_UP_TIME):
        super().__init__()
        self.torque_profile = torque_profile
        self.ramp_up_time = ramp_up_time

    def apply_torques(self, system, time: np.float64 = 0.0):
        factor = min(1.0, time / self.ramp_up_time) if self.ramp_up_time > 0 else 1.0
        system.external_torques += self.torque_profile * factor


class StoreRodDataCallback(CallBackBaseClass):
    """按固定间隔记录杆体的位置和半径。"""

    def __init__(self, step_skip, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if current_step % self.every == 0:
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())


class ProgressBarCallback(CallBackBaseClass):
    """仿真进度条回调。"""

    def __init__(self, step_skip, total_steps):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.pbar = tqdm(total=total_steps, desc="Simulating")
        self.last_step = 0

    def make_callback(self, system, time, current_step):
        if current_step % self.every == 0:
            self.pbar.update(current_step - self.last_step)
            self.last_step = current_step

    def __del__(self):
        if hasattr(self, 'pbar'):
            self.pbar.close()


# =============================================================================
# 仿真器
# =============================================================================

class SoftArmSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks, Damping):
    """PyElastica 仿真器容器。"""


def create_simulation(driving_params, total_steps=0, verbose=False):
    """创建并配置一次软体臂仿真实例。

    Args:
        driving_params: 驱动扭矩分量，形状 (2,) 或 (action_dim,)。
        total_steps: 总仿真步数（用于进度条显示）。
        verbose: 是否启用进度条回调。

    Returns:
        (simulation, callback_data, torque_force)
        - simulation: 完成 finalize 的 SoftArmSimulator。
        - callback_data: 记录位置与半径历史的字典。
        - torque_force: SimpleDistributedTorque 实例引用，可用于后续修改扭矩。
    """
    simulation = SoftArmSimulator()

    soft_arm = CosseratRod.straight_rod(
        N_ELEMENTS, ROD_START_POS, ROD_DIRECTION, ROD_NORMAL,
        ROD_LENGTH, ROD_RADIUS, ROD_DENSITY,
        youngs_modulus=YOUNGS_MODULUS, shear_modulus=SHEAR_MODULUS,
    )
    simulation.append(soft_arm)
    simulation.constrain(soft_arm).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,),
    )

    torque_profile = np.zeros((3, N_ELEMENTS))
    torque_profile[0, :] = driving_params[0]
    torque_profile[1, :] = driving_params[1]

    torque_force = SimpleDistributedTorque(torque_profile, ramp_up_time=RAMP_UP_TIME)
    simulation.add_forcing_to(soft_arm).using(
        SimpleDistributedTorque,
        torque_profile=torque_profile,
        ramp_up_time=RAMP_UP_TIME,
    )

    simulation.dampen(soft_arm).using(
        AnalyticalLinearDamper, damping_constant=DAMPING_CONSTANT, time_step=1e-4,
    )

    callback_data = defaultdict(list)
    simulation.collect_diagnostics(soft_arm).using(
        StoreRodDataCallback, step_skip=5000, callback_params=callback_data,
    )

    if verbose:
        simulation.collect_diagnostics(soft_arm).using(
            ProgressBarCallback, step_skip=100, total_steps=total_steps,
        )

    simulation.finalize()

    # finalize 后从 _forces 中取回实际实例引用
    # PyElastica finalize 会重新实例化所有注册的 forcing，需要找到对应实例
    torque_ref = None
    for force_list in getattr(simulation, '_forcing', {}).values():
        for f in force_list:
            if isinstance(f, SimpleDistributedTorque):
                torque_ref = f
                break
        if torque_ref is not None:
            break

    if torque_ref is None:
        # 回退：直接使用预创建对象（某些 PyElastica 版本不会重新实例化）
        torque_ref = torque_force

    return simulation, callback_data, torque_ref


# =============================================================================
# 渲染
# =============================================================================

def render_rod_as_image(position_data, radius_data, image_size=DEFAULT_IMAGE_SIZE,
                        cam_eye=CAMERA_EYE, cam_center=CAMERA_CENTER, cam_up=CAMERA_UP,
                        show_window=False):
    """将杆体渲染为 RGB 图像。

    Args:
        position_data: 杆体节点位置，形状 (3, N_nodes)。
        radius_data: 杆体半径数组。
        image_size: 输出图像分辨率 (W, H)。
        cam_eye: 相机位置。
        cam_center: 相机注视点。
        cam_up: 相机上方向。
        show_window: 是否短暂弹出可视化窗口。

    Returns:
        RGB 图像数组。
    """
    points = position_data.T
    n_points = points.shape[0]
    cells = np.hstack((n_points, np.arange(n_points)))
    poly_data = pv.PolyData(points)
    poly_data.lines = cells

    avg_radius = np.mean(radius_data)
    tube = poly_data.tube(radius=avg_radius)

    plotter = pv.Plotter(window_size=image_size, off_screen=not show_window)
    plotter.set_background("black")
    plotter.add_mesh(tube, color="white", lighting=False)
    plotter.camera_position = [cam_eye, cam_center, cam_up]

    if show_window:
        plotter.show()
        plotter = pv.Plotter(window_size=image_size, off_screen=True)
        plotter.set_background("black")
        plotter.add_mesh(tube, color="white", lighting=False)
        plotter.camera_position = [cam_eye, cam_center, cam_up]

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def render_to_binary(position_data, radius_data, image_size=DEFAULT_IMAGE_SIZE,
                     cam_eye=CAMERA_EYE, cam_center=CAMERA_CENTER, cam_up=CAMERA_UP,
                     threshold=127):
    """渲染杆体并返回二值图像。

    Args:
        position_data: 杆体节点位置。
        radius_data: 杆体半径。
        image_size: 输出图像分辨率。
        cam_eye: 相机位置。
        cam_center: 相机注视点。
        cam_up: 相机上方向。
        threshold: 二值化阈值。

    Returns:
        二值图像数组 (0/1)。
    """
    rgb_img = render_rod_as_image(position_data, radius_data, image_size,
                                  cam_eye, cam_center, cam_up)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(gray_img, threshold, 1, cv2.THRESH_BINARY)
    return binary_img


# =============================================================================
# 静态采集接口
# =============================================================================

def get_simulation_data_pair(driving_params, final_time=1.0, dt=1e-4,
                             verbose=True, visualize=False):
    """执行一次独立仿真并返回二值观测图像（静态采集接口）。

    Args:
        driving_params: 驱动扭矩。
        final_time: 仿真时长 (s)。
        dt: 时间步长 (s)。
        verbose: 是否输出日志。
        visualize: 是否弹出渲染窗口。

    Returns:
        (driving_params, binary_img)
    """
    total_steps = int(final_time / dt)
    simulation, callback_data, _ = create_simulation(
        driving_params, total_steps, verbose,
    )
    timestepper = PositionVerlet()

    if verbose:
        print(f"开始仿真... 参数: {driving_params}")

    integrate(timestepper, simulation, final_time, total_steps)

    if not callback_data["position"]:
        return driving_params, np.zeros(DEFAULT_IMAGE_SIZE, dtype=np.uint8)

    final_pos = callback_data["position"][-1]
    final_rad = callback_data["radius"][-1]

    binary_img = render_to_binary(final_pos, final_rad, show_window=visualize)
    return driving_params, binary_img


# =============================================================================
# 连续仿真环境
# =============================================================================

class ContinuousSoftArmEnv:
    """连续时间仿真环境，保持仿真状态逐步推进，适用于序列数据采集。

    Usage:
        env = ContinuousSoftArmEnv(dt=1e-4)
        env.set_action([torque_x, torque_y])
        env.step(steps=500)
        img, action = env.get_observation()
    """

    def __init__(self, dt=1e-4, final_time=1.0):
        self.dt = dt
        self.final_time = final_time
        self.current_time = 0.0
        self.step_count = 0

        initial_params = np.array([0.0, 0.0])
        self.simulation, _, self.torque_force = create_simulation(initial_params)

        if self.torque_force is None:
            raise RuntimeError("未在仿真中找到 SimpleDistributedTorque 实例")

        self.timestepper = PositionVerlet()

        # 预热：让系统达到稳定状态
        print("正在初始化环境稳定性...")
        for _ in range(1000):
            self.current_time = self.timestepper.do_step(
                self.timestepper,
                self.timestepper.steps_and_prefactors,
                self.simulation,
                self.current_time,
                self.dt,
            )

    def set_action(self, driving_params):
        """原地更新当前驱动扭矩配置，不重置仿真状态。"""
        n_elements = self.torque_force.torque_profile.shape[1]
        new_profile = np.zeros((3, n_elements))
        new_profile[0, :] = driving_params[0]
        new_profile[1, :] = driving_params[1]
        self.torque_force.torque_profile[:] = new_profile

    def step(self, steps=1):
        """推进物理积分器若干步。"""
        for _ in range(steps):
            self.current_time = self.timestepper.do_step(
                self.timestepper,
                self.timestepper.steps_and_prefactors,
                self.simulation,
                self.current_time,
                self.dt,
            )
            self.step_count += 1

    def get_observation(self):
        """获取当前二值图像观测与驱动扭矩。

        Returns:
            (binary_img, current_action)
        """
        soft_arm = self.simulation[0]
        current_pos = soft_arm.position_collection.copy()
        current_radius = soft_arm.radius.copy()

        binary_img = render_to_binary(current_pos, current_radius)

        current_torques = self.torque_force.torque_profile[:, 0]
        current_action = np.array([current_torques[0], current_torques[1]])

        return binary_img, current_action

    def get_observation_3d(self):
        """获取当前二值图像 + 3D 节点坐标 + 半径。

        Returns:
            (binary_img, current_action, positions, radii)
            - binary_img: 二值渲染图像。
            - current_action: 驱动扭矩 (2,)。
            - positions: 节点坐标 (3, N_nodes)，N_nodes=N_ELEMENTS+1。
            - radii: 节点半径 (N_nodes,)。
        """
        soft_arm = self.simulation[0]
        positions = soft_arm.position_collection.copy()
        radii = soft_arm.radius.copy()

        binary_img = render_to_binary(positions, radii)

        current_torques = self.torque_force.torque_profile[:, 0]
        current_action = np.array([current_torques[0], current_torques[1]])

        return binary_img, current_action, positions, radii


if __name__ == "__main__":
    test_params = np.array([0.1, 0.0])
    get_simulation_data_pair(test_params, verbose=True, visualize=True)
