# save_gif.py (在服务器运行)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def save_as_gif(npz_path, output_gif="preview.gif"):
    data = np.load(npz_path)
    images = data['images']
    actions = data['actions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im = ax1.imshow(images[0], cmap='gray')
    line, = ax2.plot([], [], 'r-')
    ax2.set_xlim(0, len(actions))
    ax2.set_ylim(np.min(actions), np.max(actions))
    
    # 红色竖线指示当前进度
    vline = ax2.axvline(x=0, color='k', linestyle='--')
    
    # 绘制完整的动作背景
    for i in range(actions.shape[1]):
        ax2.plot(actions[:, i], alpha=0.3)

    def update(frame):
        im.set_data(images[frame])
        vline.set_xdata([frame, frame])
        ax1.set_title(f"Frame {frame}")
        return im, vline

    # 降采样，防止GIF太大
    frames = range(0, len(images), 5) 
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    
    print(f"正在生成 GIF: {output_gif} ...")
    ani.save(output_gif, writer='pillow', fps=10)
    print("完成！请在 VS Code 中打开 GIF 查看。")

# 自动运行最新的数据
target_dir = "data/sequence_data"
files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npz')], key=os.path.getmtime)
if files:
    save_as_gif(files[-1])