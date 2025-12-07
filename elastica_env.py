import numpy as np
import pyvista as pv
import cv2
from collections import defaultdict
from tqdm import tqdm

# --- 1. PyVista 全局设置 ---
pv.set_plot_theme("document")
pv.OFF_SCREEN = True 

# --- 2. PyElastica 模块导入 ---
from elastica.modules import BaseSystemCollection, Constraints, Forcing, CallBacks, Damping
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import NoForces, MuscleTorques
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
# [关键] 导入 SymplecticStepperMixin 以便我们查看其签名或调用它
from elastica.timestepper.symplectic_steppers import PositionVerlet, SymplecticStepperMixin
from elastica.timestepper import integrate

# --- 3. 自定义组件 ---

class SimpleDistributedTorque(NoForces):
    """
    自定义力类，用于沿杆施加分布扭矩。
    """
    # 类属性：用于在 finalize 后“偷”出实例引用
    latest_instance = None 

    def __init__(self, torque_profile, ramp_up_time=0.0):
        super().__init__()
        self.torque_profile = torque_profile
        self.ramp_up_time = ramp_up_time
        # 记录自身实例
        SimpleDistributedTorque.latest_instance = self

    def apply_torques(self, system, time: np.float64 = 0.0):
        factor = 1.0
        if self.ramp_up_time > 0:
            factor = min(1.0, time / self.ramp_up_time)
        system.external_torques += self.torque_profile * factor

class StoreRodDataCallback(CallBackBaseClass):
    def __init__(self, step_skip, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if current_step % self.every == 0:
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())

class ProgressBarCallback(CallBackBaseClass):
    def __init__(self, step_skip, total_steps):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.pbar = tqdm(total=total_steps, desc="Simulating")
        self.last_step = 0

    def make_callback(self, system, time, current_step):
        if current_step % self.every == 0:
            update_val = current_step - self.last_step
            self.pbar.update(update_val)
            self.last_step = current_step
            
    def __del__(self):
        if hasattr(self, 'pbar'):
            self.pbar.close()

# --- 4. 仿真器定义 ---
class SoftArmSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks, Damping):
    pass

# --- 5. 核心功能函数 ---

def create_simulation(driving_params, total_steps=0, verbose=False):
    simulation = SoftArmSimulator()
    
    # 物理参数
    n_elements = 30
    start_pos = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 0.5
    base_radius = 0.015
    density = 1000
    youngs_modulus = 1e6
    shear_modulus = youngs_modulus / (4.0 * (1.0 + 0.5))
    
    soft_arm = CosseratRod.straight_rod(
        n_elements, start_pos, direction, normal, base_length, base_radius,
        density, youngs_modulus=youngs_modulus, shear_modulus=shear_modulus
    )
    
    simulation.append(soft_arm)
    
    simulation.constrain(soft_arm).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    
    torque_profile = np.zeros((3, n_elements))
    torque_profile[0, :] = driving_params[0]
    torque_profile[1, :] = driving_params[1]

    simulation.add_forcing_to(soft_arm).using(
        SimpleDistributedTorque,
        torque_profile=torque_profile,
        ramp_up_time=0.5
    )
    
    simulation.dampen(soft_arm).using(
        AnalyticalLinearDamper, damping_constant=0.1, time_step=1e-4
    )

    callback_data = defaultdict(list)
    simulation.collect_diagnostics(soft_arm).using(
        StoreRodDataCallback, step_skip=5000, callback_params=callback_data
    )

    if verbose:
        simulation.collect_diagnostics(soft_arm).using(
            ProgressBarCallback, step_skip=100, total_steps=total_steps
        )

    # 初始化系统 (触发所有组件的 __init__)
    simulation.finalize()
    
    # [挂载力对象] 必须在 finalize 之后做，因为 using 只是注册，finalize 才是实例化
    if SimpleDistributedTorque.latest_instance is not None:
        simulation.user_torque_ref = SimpleDistributedTorque.latest_instance
    else:
        print("警告：SimpleDistributedTorque 实例未捕获到！")
        
    return simulation, callback_data

def render_rod_as_image(position_data, radius_data, image_size=(100, 100), show_window=False):
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
    
    # 调整为标准侧视图，方便观察 X/Y 方向弯曲
    plotter.camera_position = [
        (1.5, 0.0, 0.5), # Eye: X轴正向远处，稍微抬高
        (0.0, 0.0, 0.25), # Focus: 杆子中心
        (0.0, 0.0, 1.0)   # Up: Z轴向上
    ]
    
    if show_window:
        plotter.show()
        plotter = pv.Plotter(window_size=image_size, off_screen=True)
        plotter.set_background("black")
        plotter.add_mesh(tube, color="white", lighting=False)
        plotter.camera_position = [(1.5, 0.0, 0.5), (0.0, 0.0, 0.25), (0.0, 0.0, 1.0)]

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img

def get_simulation_data_pair(driving_params, verbose=True, visualize=False):
    """
    单次仿真接口 (非连续)
    """
    final_time = 1.0
    dt = 1e-4
    total_steps = int(final_time / dt)
    
    simulation, callback_data = create_simulation(driving_params, total_steps, verbose)
    timestepper = PositionVerlet()
    
    if verbose:
        print(f"开始仿真... 参数: {driving_params}")

    integrate(timestepper, simulation, final_time, total_steps)
    
    if not callback_data["position"]:
        return driving_params, np.zeros((100, 100), dtype=np.uint8)

    final_pos = callback_data["position"][-1]
    final_rad = callback_data["radius"][-1]
    
    rgb_img = render_rod_as_image(final_pos, final_rad, show_window=visualize)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)
    
    return driving_params, binary_img

# ==========================================
#   连续时间仿真环境类 (修复版)
# ==========================================

class ContinuousSoftArmEnv:
    def __init__(self, dt=1e-4):
        self.dt = dt
        self.current_time = 0.0
        self.step_count = 0
        
        # 1. 创建仿真器
        initial_params = np.array([0.0, 0.0])
        self.simulation, _ = create_simulation(initial_params)
        
        # 2. 获取 Force 对象引用
        if hasattr(self.simulation, 'user_torque_ref'):
            self.torque_force_cls = self.simulation.user_torque_ref
        else:
            raise RuntimeError("未在仿真中找到 SimpleDistributedTorque 实例！")

        # 3. 初始化时间步进器
        self.timestepper = PositionVerlet()
        
        # [关键修复] 预热循环: 使用显式参数调用 do_step
        # 你的 PyElastica 版本中 do_step 是 staticmethod，需要 5 个参数
        print("正在初始化环境稳定性...")
        for _ in range(1000):
            self.current_time = self.timestepper.do_step(
                self.timestepper,                       # time_stepper 实例
                self.timestepper.steps_and_prefactors,  # 预计算系数
                self.simulation,                        # 系统集合
                self.current_time,                      # 当前时间
                self.dt                                 # 步长
            )

    def set_action(self, driving_params):
        """
        更新当前的驱动参数 (不重置仿真，直接修改力)
        """
        n_elements = self.torque_force_cls.torque_profile.shape[1]
        new_profile = np.zeros((3, n_elements))
        new_profile[0, :] = driving_params[0]
        new_profile[1, :] = driving_params[1]
        # 原地修改数组
        self.torque_force_cls.torque_profile[:] = new_profile

    def step(self, steps=1):
        """
        向前推进物理仿真 n 步
        """
        for _ in range(steps):
            # [关键修复] 使用显式参数调用 do_step
            self.current_time = self.timestepper.do_step(
                self.timestepper,
                self.timestepper.steps_and_prefactors,
                self.simulation,
                self.current_time,
                self.dt
            )
            self.step_count += 1

    def get_observation(self):
        """
        获取当前的图像观察 (二值化) 和 真实驱动参数
        """
        # rod 是 simulation._systems 中的第一个对象
        soft_arm = self.simulation[0]
        
        current_pos = soft_arm.position_collection.copy()
        current_radius = soft_arm.radius.copy()
        
        # 渲染图像
        rgb_img = render_rod_as_image(current_pos, current_radius, show_window=False)
        
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)
        
        current_torques = self.torque_force_cls.torque_profile[:, 0]
        current_action = np.array([current_torques[0], current_torques[1]])
        
        return binary_img, current_action

if __name__ == "__main__":
    test_params = np.array([0.1, 0.0])
    get_simulation_data_pair(test_params, verbose=True, visualize=True)