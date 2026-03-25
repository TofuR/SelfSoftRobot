import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

def generate_validation_gif(model, val_data, seq_len, action_dim, H, W, render_fn, save_path, device='cpu'):
    """生成验证序列的 GT vs Pred GIF。

    Args:
        model: 模型
        val_data: 验证数据 {'images': (T, H, W), 'actions': (T, D)}
        seq_len: 序列长度
        action_dim: 动作维度
        H, W: 图像尺寸
        render_fn: 渲染函数
        save_path: 保存路径
        device: 设备
    """
    model.eval()
    seq_length = val_data['length']
    
    gt_imgs = []
    pred_imgs = []
    
    with torch.no_grad():
        for t in range(seq_length):
            start = max(0, t - seq_len + 1)
            actions_seq = val_data['actions'][start:t+1]
            if len(actions_seq) < seq_len:
                pad = np.zeros((seq_len - len(actions_seq), action_dim))
                actions_seq = np.concatenate([pad, actions_seq], axis=0)
            
            input_seq = torch.from_numpy(actions_seq).float().unsqueeze(0).to(device)
            target_img_flat = torch.from_numpy(val_data['images'][t]).float().reshape(-1).unsqueeze(0).to(device)
            
            pred_flat, _, _ = render_fn(input_seq, is_train=False)
            pred_img = pred_flat[0].reshape(H, W).cpu().numpy()
            gt_img = target_img_flat[0].reshape(H, W).cpu().numpy()
            
            gt_imgs.append(gt_img)
            pred_imgs.append(pred_img)
    
    # 生成 GIF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    ax1.axis('off'); ax2.axis('off')
    
    im1 = ax1.imshow(gt_imgs[0], cmap='gray', vmin=0, vmax=1)
    im2 = ax2.imshow(pred_imgs[0], cmap='gray', vmin=0, vmax=1)
    
    def update(frame):
        im1.set_data(gt_imgs[frame])
        im2.set_data(pred_imgs[frame])
        ax1.set_title(f"GT (Frame {frame})")
        ax2.set_title(f"Pred (Frame {frame})")
        return im1, im2
    
    ani = animation.FuncAnimation(fig, update, frames=len(gt_imgs), blit=True)
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()