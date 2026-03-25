import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class SoftSequenceDataset(Dataset):
    """通用序列数据集：从 .npz 文件加载动作-图像序列。

    支持不同模式：图像序列或单帧。
    """
    def __init__(self, data_dir, seq_len=10, file_list=None, norm_factor=None, target_size=None):
        self.seq_len = seq_len
        self.target_size = target_size  # 如果需要 resize 图像
        self.samples = []
        
        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # 归一化计算
        if norm_factor is None:
            all_acts = []
            for f in file_list:
                try:
                    d = np.load(f)
                    all_acts.append(d['actions'])
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            if all_acts:
                all_acts = np.concatenate(all_acts, axis=0)
                self.norm_factor = np.max(np.abs(all_acts))
            else:
                self.norm_factor = 1.0
            if self.norm_factor == 0:
                self.norm_factor = 1.0
            print(f"Norm Factor: {self.norm_factor}")
        else:
            self.norm_factor = norm_factor

        self.data_cache = []
        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor
            
            # 图像处理
            imgs_raw = raw['images']
            if self.target_size is not None:
                imgs_processed = []
                for img in imgs_raw:
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                    img_resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                    imgs_processed.append(img_resized)
                images = np.stack(imgs_processed, axis=0)[:, np.newaxis, :, :]  # (T, 1, H, W)
            else:
                images = imgs_raw  # 保持原始
            
            self.data_cache.append({
                'images': images, 
                'actions': actions, 
                'length': len(images)
            })
            
        for seq_id, item in enumerate(self.data_cache):
            for t in range(item['length']):
                self.samples.append((seq_id, t))
        
        self.H, self.W = self.data_cache[0]['images'].shape[1:3] if self.target_size else self.data_cache[0]['images'].shape[1:3]
        self.action_dim = self.data_cache[0]['actions'].shape[1]
        self.focal = float(raw.get('focal', 130.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]
        start, end = t - self.seq_len + 1, t + 1
        if start >= 0:
            seq = data['actions'][start:end]
        else:
            pad = np.zeros((self.seq_len - (end), self.action_dim))
            seq = np.concatenate([pad, data['actions'][0:end]], axis=0)
        
        if self.target_size:
            # 返回图像序列
            image_seq = data['images'][start:end]  # (Seq, 1, H, W)
            return torch.from_numpy(image_seq).float(), torch.from_numpy(seq).float()
        else:
            # 返回单帧展平
            return torch.from_numpy(seq).float(), torch.from_numpy(data['images'][t]).float().reshape(-1)
    
    def get_raw_actions(self, seq_id=0):
        return self.data_cache[seq_id]['actions'] * self.norm_factor