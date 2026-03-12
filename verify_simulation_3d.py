import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

# 导入 PyElastica 和 环境配置
from elastica_env import create_simulation, render_rod_as_image
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

def get_camera_matrix(eye, center, up):
    """
    计算相机的 View Matrix (World -> Camera)
    """
    # 1. 计算视线方向 (Z_cam): 从 eye 指向 center (OpenGL习惯是反的，但在投影计算中我们通常取 look_dir)
    # 这里我们定义: Z_cam 指向相机背后 (符合标准图形学 View Matrix 推导)
    f = center - eye 
    f = f / np.linalg.norm(f) # 前向向量 (Forward)
    
    # 2. 计算右向量 (X_cam)
    # Right = Forward x Up
    s = np.cross(f, up) 
    s = s / np.linalg.norm(s) # 右向量 (Right/Side)
    
    # 3. 重新计算上向量 (Y_cam) 保证正交
    # Up = Right x Forward
    u = np.cross(s, f) # 修正后的上向量 (True Up)
    
    # 4. 构造 View Matrix (World -> Camera)
    # 这是一个刚体变换矩阵
    view_matrix = np.eye(4)
    view_matrix[0, :3] = s  # Row 0: Right
    view_matrix[1, :3] = u  # Row 1: Up
    view_matrix[2, :3] = -f # Row 2: Back (摄像机看向 -Z)
    
    # 平移部分
    view_matrix[0, 3] = -np.dot(s, eye)
    view_matrix[1, 3] = -np.dot(u, eye)
    view_matrix[2, 3] = np.dot(f, eye) # 注意这里的符号
    # 这是一个刚体变换矩阵 [R|t]
    # 注意：这是将点转换到相机坐标系，相机看向 -Z 方向（或者根据定义不同）
    # 在这个推导中，Z轴指向相机背后，X向右，Y向上
    return view_matrix

def project_points(points_3d, view_matrix, focal_length=1.0):
    """
    将 3D 点投影到 2D 平面 (简单的透视投影)
    points_3d: (3, N)
    """
    # 转换为齐次坐标 (4, N)
    ones = np.ones((1, points_3d.shape[1]))
    points_hom = np.vstack((points_3d, ones))
    
    # 转换到相机坐标系
    points_cam = np.dot(view_matrix, points_hom) # (4, N)
    
    # 透视除法
    # 假设相机看向 -Z (如果使用上面的 lookAt，Z 是正的指向眼睛)
    # 这里的 view_matrix z 轴是指向 eye 的，物体在 center，所以 z 是负值或正值取决于坐标系
    # 我们取 Z 的深度
    z_depth = -points_cam[2, :] # 距离相机的距离
    
    # 防止除零
    z_depth[z_depth < 0.1] = 0.1
    
    # 简单的针孔投影模型: u = f * x / z, v = f * y / z
    u = focal_length * points_cam[0, :] / z_depth
    v = focal_length * points_cam[1, :] / z_depth
    
    return u, v, points_cam

def run_verification(driving_params):
    """运行一次 3D 投影一致性验证并输出对比图。

    参数:
        driving_params: 仿真驱动参数。
    """
    print(f">>> 开始验证仿真，驱动参数: {driving_params}")
    
    # 1. 运行仿真获取数据
    # 我们这里手动运行以便拿到 callback 数据
    final_time = 1.0
    dt = 1e-4
    total_steps = int(final_time / dt)
    
    simulation, callback_data = create_simulation(driving_params, total_steps, verbose=True)
    timestepper = PositionVerlet()
    integrate(timestepper, simulation, final_time, total_steps)
    
    # 提取最后一帧的 3D 数据
    # position shape: (3, N_nodes)
    pos_3d = callback_data["position"][-1]
    radius_data = callback_data["radius"][-1]
    
    # 2. 生成实际渲染图 (用于对比)
    # 参数需与 elastica_env.py 中一致
    cam_eye = np.array([1.0, 0.0, 0.5])
    cam_center = np.array([0.0, 0.0, 0.25])
    cam_up = np.array([0.0, 0.0, 1.0])
    
    rgb_img = render_rod_as_image(pos_3d, radius_data, image_size=(300, 300), show_window=False)
    
    # 3. 计算理论投影
    view_mat = get_camera_matrix(cam_eye, cam_center, cam_up)
    u, v, pts_cam = project_points(pos_3d, view_mat, focal_length=2.0)
    
    # --- 开始绘图 ---
    print(">>> 正在生成验证图表...")
    fig = plt.figure(figsize=(15, 5))
    
    # 子图 1: 真实 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("3D Shape (World Coords)")
    ax1.plot(pos_3d[0], pos_3d[1], pos_3d[2], 'b-', linewidth=2, label='Soft Arm')
    ax1.scatter(pos_3d[0], pos_3d[1], pos_3d[2], c='b', s=10)
    
    # 画坐标轴 (原点)
    origin = np.zeros(3)
    length = 0.2
    ax1.quiver(origin[0], origin[1], origin[2], length, 0, 0, color='r', label='X')
    ax1.quiver(origin[0], origin[1], origin[2], 0, length, 0, color='g', label='Y')
    ax1.quiver(origin[0], origin[1], origin[2], 0, 0, length, color='b', label='Z')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-0.1, 0.6); ax1.set_ylim(-0.1, 0.6); ax1.set_zlim(0, 0.6)
    ax1.legend()
    
    # 子图 2: 2D 投影点 (计算值)
    ax2 = fig.add_subplot(132)
    ax2.set_title("Calculated 2D Projection (Points)")
    ax2.plot(u, v, 'r-o', label='Projected Nodes')
    ax2.set_xlabel('u (Image Horizontal)')
    ax2.set_ylabel('v (Image Vertical)')
    ax2.grid(True)
    ax2.axis('equal')
    
    # 在投影图上也画出坐标轴方向 (投影后的基向量)
    # 将世界坐标系的基向量投影到 2D
    axes_pts = np.array([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).T
    au, av, _ = project_points(axes_pts, view_mat, focal_length=2.0)
    origin_uv = (au[0], av[0])
    ax2.arrow(origin_uv[0], origin_uv[1], au[1]-au[0], av[1]-av[0], color='r', width=0.005, label='X_dir')
    ax2.arrow(origin_uv[0], origin_uv[1], au[2]-au[0], av[2]-av[0], color='g', width=0.005, label='Y_dir')
    ax2.arrow(origin_uv[0], origin_uv[1], au[3]-au[0], av[3]-av[0], color='b', width=0.005, label='Z_dir')
    ax2.legend()

    # 子图 3: 实际 PyVista 渲染图
    ax3 = fig.add_subplot(133)
    ax3.set_title("Actual Rendered Image (Binary)")
    # 将 RGB 转灰度显示
    ax3.imshow(rgb_img)
    ax3.axis('off')
    
    # 保存结果
    save_path = "verification_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✅ 验证图表已保存至: {os.path.abspath(save_path)}")
    print("   (包含 3D 轨迹、2D 投影点分布、以及实际渲染图像)")

if __name__ == "__main__":
    # 使用一个较小的测试参数，确保弯曲明显但不穿模
    test_params = np.array([-0.005, 0.005])
    run_verification(test_params)