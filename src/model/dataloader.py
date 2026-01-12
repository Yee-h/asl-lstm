import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg


def preprocess_keypoints(data: np.ndarray, max_frames: int = cfg.MAX_FRAMES) -> torch.Tensor:
    """
    对关键点序列进行预处理：展平、转换为 Tensor。
    (V2 数据已经是固定长度 90, 4, 135，只需展平)
    """
    # 展平特征：(Frames, 4, 135) -> (Frames, 540)
    data = data.reshape(data.shape[0], -1)
    
    # 转换为 PyTorch 张量
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    return data_tensor


class CSLDataset(Dataset):
    """
    手语识别数据集类，负责加载 HDF5 格式的特征数据和相应的标签。
    """
    def __init__(self, hdf5_path, label_map_path, max_frames=cfg.MAX_FRAMES, augment=False):
        self.max_frames = max_frames
        self.augment = augment
        
        # --- 加载标签映射表 ---
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            
        if 'id_to_label' in label_map:
            self.label_to_id = label_map['id_to_label']
        elif 'label_to_id' in label_map:
            sample_key = next(iter(label_map['label_to_id'].keys()))
            if sample_key.isdigit():
                self.label_to_id = {v: int(k) for k, v in label_map['label_to_id'].items()}
            else:
                self.label_to_id = label_map['label_to_id']
        else:
            self.label_to_id = label_map
        
        # --- 将数据预加载到内存中 ---
        self.data_cache = []
        print(f"正在加载 {hdf5_path} 数据到内存...")
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                item = f[key]
                if not isinstance(item, h5py.Group): continue
                group = item
                
                if 'data' not in group.keys() or 'label' not in group.keys():
                    continue
                    
                data_item = group['data']
                feature = np.array(data_item)
                # 读取有效长度，如果不存在则使用 max_frames
                length = int(group['length'][()]) if 'length' in group.keys() else self.max_frames
                
                label_dataset = group['label']
                if isinstance(label_dataset, h5py.Dataset):
                    label_raw = label_dataset[()]
                else:
                    continue
                if isinstance(label_raw, bytes):
                    label_str = label_raw.decode('utf-8')
                else:
                    label_str = str(label_raw)
                
                if label_str in self.label_to_id:
                    label_id = int(self.label_to_id[label_str])
                    self.data_cache.append((feature, label_id, length))
        
        print(f"成功加载 {len(self.data_cache)} 条样本。")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        # Unpack data, label, length
        data, label_id, valid_len = self.data_cache[idx]
        
        if self.augment:
            data = self._apply_augmentation(data)
        
        data_tensor = preprocess_keypoints(data, self.max_frames)
        
        return data_tensor, label_id, valid_len
    
    def _apply_augmentation(self, data):
        """应用随机旋转和缩放 (适配 4 通道: x, y, dx, dy)"""
        angle = np.random.uniform(-cfg.AUG_ROTATION_RANGE, cfg.AUG_ROTATION_RANGE)
        theta = np.radians(angle)
        scale = np.random.uniform(cfg.AUG_SCALE_MIN, cfg.AUG_SCALE_MAX)
        tx = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)
        ty = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        transformed_data = data.copy()
        
        # 1. 分离位置 (..., 0:2, ...) 和 速度 (..., 2:4, ...)
        # data shape: (T, 4, 135)
        pos = data[:, 0:2, :]  # (T, 2, 135)
        vel = data[:, 2:4, :]  # (T, 2, 135)

        for t in range(data.shape[0]):
            # --- 处理位置 ---
            # (2, 135).T -> (135, 2)
            pos_t = pos[t].T
            # Rotate & Scale
            pos_t = (pos_t @ rotation_matrix.T) * scale
            # Translate
            pos_t = pos_t + np.array([tx, ty])
            transformed_data[t, 0:2, :] = pos_t.T
            
            # --- 处理速度 ---
            # 速度只旋转和缩放，不平移
            vel_t = vel[t].T
            vel_t = (vel_t @ rotation_matrix.T) * scale
            transformed_data[t, 2:4, :] = vel_t.T

        # 2. 高斯噪声
        noise = np.random.normal(loc=0.0, scale=cfg.AUG_NOISE_STD, size=transformed_data.shape)
        transformed_data = transformed_data + noise

        # 3. 水平翻转
        if np.random.random() < cfg.AUG_HFLIP_PROB:
            # 翻转 X 坐标 (pos_x 和 vel_x)
            transformed_data[:, 0, :] = 1.0 - transformed_data[:, 0, :]
            transformed_data[:, 2, :] = -transformed_data[:, 2, :] # 速度反向
        
        return transformed_data

def get_dataloaders():
    """
    创建并返回训练、验证和测试数据加载器。
    """
    train_dataset = CSLDataset(cfg.TRAIN_DATA_PATH, cfg.LABEL_MAP_PATH, augment=True)
    val_dataset = CSLDataset(cfg.VAL_DATA_PATH, cfg.LABEL_MAP_PATH, augment=False)
    test_dataset = CSLDataset(cfg.TEST_DATA_PATH, cfg.LABEL_MAP_PATH, augment=False)
    
    # 在 Windows 系统上，num_workers 设置为 0 通常更稳定
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, _, _ = get_dataloaders()
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data Shape {data.shape}, Label Shape {label.shape}")
        if batch_idx == 0:
            break
