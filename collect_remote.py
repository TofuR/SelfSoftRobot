import numpy as np
import cv2
import os
import time
from elastica_env import get_simulation_data_pair  # 导入我们刚才写好的仿真环境

def run_debug_check():
    """
    调试模式：运行一次仿真，保存图片和数据，供您检查
    """
    print(">>> [调试模式] 正在运行单次仿真...")
    
    # 1. 定义一个测试动作 (X轴弯曲)
    test_params = np.array([0.0, 0.01])    
    # 2. 运行仿真 (关闭弹窗 visualize=False)
    # verbose=True 会显示进度条
    params, binary_img = get_simulation_data_pair(test_params, verbose=True, visualize=False)
    
    # 3. 保存可视化图片 (这是给您看的)
    # 将二值化图像 (0/1) 扩展为 0/255 以便查看
    vis_img = binary_img * 255
    save_img_path = "debug_result.png"
    cv2.imwrite(save_img_path, vis_img)
    print(f"✅ 可视化图片已保存至: {os.path.abspath(save_img_path)}")
    print("   (请在 VS Code 文件列表中点击该图片查看效果)")

    # 4. 保存数据文件 (这是给模型训练用的)
    save_data_path = "debug_data.npz"
    np.savez(save_data_path, images=np.array([binary_img]), angles=np.array([params]))
    print(f"✅ 训练数据已保存至: {os.path.abspath(save_data_path)}")


def run_batch_collection(sample_count=100):
    """
    批量采集模式：采集大量数据用于训练
    """
    print(f"\n>>> [采集模式] 开始采集 {sample_count} 组数据...")
    
    data_save_dir = "data/real_data"
    os.makedirs(data_save_dir, exist_ok=True)
    
    images_list = []
    angles_list = []
    
    for i in range(sample_count):
        # 1. 随机生成驱动参数 (例如 -5.0 到 5.0 之间)
        # 假设我们有两个驱动自由度
        rand_params = (np.random.rand(2) * 10.0) - 5.0
        
        # 2. 运行仿真
        # 只有第一组显示进度条，后面为了刷屏整洁可以关掉 verbose，或者保留
        params, img = get_simulation_data_pair(rand_params, verbose=False, visualize=False)
        
        # 3. 收集数据
        images_list.append(img)
        angles_list.append(params)
        
        # 打印简略进度
        if (i + 1) % 10 == 0:
            print(f"   已采集: {i + 1}/{sample_count}")

    # 4. 打包保存
    timestamp = int(time.time())
    file_name = f"soft_arm_data_{sample_count}_{timestamp}.npz"
    full_path = os.path.join(data_save_dir, file_name)
    
    np.savez(full_path, 
             images=np.array(images_list), 
             angles=np.array(angles_list),
             focal=1.0) # 这里的 focal 只是占位，如果不涉及真实相机标定
             
    print(f"✅ 批量采集完成！文件保存至: {full_path}")
    print(f"   数据形状: Images {np.array(images_list).shape}, Angles {np.array(angles_list).shape}")

if __name__ == "__main__":
    # --- 选择您要运行的模式 ---
    
    # 步骤 1: 先运行调试，确保生成的图片 debug_result.png 是正常的软体臂样子
    run_debug_check()
    
    # 步骤 2: 确认无误后，注释掉上面一行，取消下面一行的注释，开始大规模采集
    # run_batch_collection(sample_count=1000)