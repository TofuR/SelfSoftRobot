import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import os

def view_dataset(file_path):
    # 1. 加载数据
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return

    print(f"正在加载: {file_path} ...")
    data = np.load(file_path)
    
    # 检查键名
    print(f"包含的键: {list(data.keys())}")
    
    if 'images' not in data or 'actions' not in data:
        print("错误：数据文件格式不对，必须包含 'images' 和 'actions'")
        return

    images = data['images']
    actions = data['actions']
    
    # 获取维度信息
    num_frames = images.shape[0]
    action_dim = actions.shape[1]
    
    print(f"数据统计:")
    print(f"  - 总帧数: {num_frames}")
    print(f"  - 图像尺寸: {images.shape[1:]}")
    print(f"  - 动作维度: {action_dim}")
    if 'dt' in data:
        print(f"  - 采样间隔(dt): {data['dt']} s")

    # 2. 创建交互式绘图窗口
    fig = plt.figure(figsize=(12, 6))
    
    # --- 左侧：图像显示 ---
    ax_img = plt.subplot(1, 2, 1)
    plt.subplots_adjust(bottom=0.25) # 为滑块留出空间
    
    # 显示第一帧
    img_plot = ax_img.imshow(images[0], cmap='gray', vmin=0, vmax=1)
    ax_img.set_title(f"Frame 0/{num_frames}")
    ax_img.axis('off')

    # --- 右侧：动作曲线 ---
    ax_act = plt.subplot(1, 2, 2)
    time_steps = np.arange(num_frames)
    
    lines = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(action_dim):
        line, = ax_act.plot(time_steps, actions[:, i], label=f'Action {i}', color=colors[i % len(colors)])
        lines.append(line)
        
    # 添加一个指示当前时间的垂直线
    vline = ax_act.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    ax_act.set_title("Control Actions over Time")
    ax_act.set_xlabel("Time Step")
    ax_act.set_ylabel("Value")
    ax_act.legend()
    ax_act.grid(True)

    # --- 底部：滑块控件 ---
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )

    # --- 更新函数 ---
    def update(val):
        idx = int(slider.val)
        
        # 1. 更新图像
        img_plot.set_data(images[idx])
        ax_img.set_title(f"Frame {idx}/{num_frames}")
        
        # 2. 更新指示线
        vline.set_xdata([idx, idx])
        
        # 3. 更新标题显示当前数值
        current_acts = actions[idx]
        act_str = ", ".join([f"{x:.2f}" for x in current_acts])
        fig.suptitle(f"Current Actions: [{act_str}]", fontsize=14, color='blue')
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    # 初始化一次标题
    update(0)

    print(">>> 窗口已打开。拖动滑块查看数据序列。")
    plt.show()

if __name__ == "__main__":
    # 使用方法：可以直接在代码里改路径，或者通过命令行传参
    # 默认路径 (请修改为您实际生成的 .npz 文件名)
    
    # 自动查找 data/sequence_data 下最新的文件
    target_dir = "data/sequence_data"
    default_file = None
    
    if os.path.exists(target_dir):
        files = [f for f in os.listdir(target_dir) if f.endswith('.npz')]
        if files:
            # 按时间排序找最新的
            files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
            default_file = os.path.join(target_dir, files[0])
    
    # 获取命令行参数，如果没有则使用默认找到的文件
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    elif default_file:
        file_path = default_file
    else:
        print("未找到数据文件，请指定路径或先运行采集脚本。")
        sys.exit(1)

    view_dataset(file_path)
    