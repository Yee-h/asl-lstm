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

class CSLDataset(Dataset):
    """
    手语识别数据集类，负责加载 HDF5 格式的特征数据和相应的标签。
    """
    def __init__(self, hdf5_path, label_map_path, max_frames=cfg.MAX_FRAMES, augment=False):
        """
        初始化数据集。
        
        Args:
            hdf5_path (str): HDF5 数据文件的路径。
            label_map_path (str): 标签映射 JSON 文件的路径。
            max_frames (int): 序列的最大帧数，用于统一输入长度。
            augment (bool): 是否启用数据增强（随机旋转和缩放）。
        """
        self.max_frames = max_frames
        self.augment = augment
        
        # --- 加载标签映射表 ---
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            
        # 兼容不同格式的标签映射文件
        if 'id_to_label' in label_map:
            self.label_to_id = label_map['id_to_label']
        elif 'label_to_id' in label_map:
            # 检查键是否为数字，如果是，则可能是 ID->Label 映射，需要翻转
            sample_key = next(iter(label_map['label_to_id'].keys()))
            if sample_key.isdigit():
                self.label_to_id = {v: int(k) for k, v in label_map['label_to_id'].items()}
            else:
                self.label_to_id = label_map['label_to_id']
        else:
            # 默认情况
            self.label_to_id = label_map
        
        # --- 将数据预加载到内存中 ---
        self.data_cache = []
        print(f"正在加载 {hdf5_path} 数据到内存...")
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                item = f[key]
                
                # 确保 item 是 Group 类型
                if not isinstance(item, h5py.Group):
                    continue
                
                group = item
                
                # 检查是否存在必要的数据键
                if 'data' not in group.keys() or 'label' not in group.keys():
                    continue
                    
                # 读取特征数据 (Frames, 2, 135)
                data_item = group['data']
                feature = np.array(data_item)
                
                # 读取并处理标签字符串
                label_dataset = group['label']
                if isinstance(label_dataset, h5py.Dataset):
                    label_raw = label_dataset[()]
                else:
                    continue
                if isinstance(label_raw, bytes):
                    label_str = label_raw.decode('utf-8')
                else:
                    label_str = str(label_raw)
                
                # 如果标签在映射表中，则保存特征和对应的 ID
                if label_str in self.label_to_id:
                    label_id = int(self.label_to_id[label_str])
                    self.data_cache.append((feature, label_id))
        
        print(f"成功加载 {len(self.data_cache)} 条样本。")

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.data_cache)

    def __getitem__(self, idx):
        """
        获取指定索引的样本，并进行预处理。
        
        Returns:
            data_tensor (torch.Tensor): 形状为 (max_frames, input_size) 的特征张量。
            label_id (int): 类别标签。
        """
        # 从内存缓存中读取
        data, label_id = self.data_cache[idx]
        
        # --- 数据增强：随机旋转和缩放 ---
        if self.augment:
            data = self._apply_augmentation(data)
        
        # --- 数据预处理 ---
        # 展平特征：(Frames, 2, 135) -> (Frames, 270)
        data = data.reshape(data.shape[0], -1)
        
        # --- 长度统一处理 (截断或填充) ---
        # --- 长度统一处理 (截断或填充) ---
        seq_len, dim = data.shape
        if seq_len > self.max_frames:
            # 超过最大长度则截断
            data = data[:self.max_frames]
            valid_len = self.max_frames
        elif seq_len < self.max_frames:
            # 不足最大长度则在末尾补零
            padding = np.zeros((self.max_frames - seq_len, dim), dtype=data.dtype)
            data = np.concatenate((data, padding), axis=0)
            valid_len = seq_len
        else:
            valid_len = seq_len
            
        # 转换为 PyTorch 张量，并使用 float32 精度
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        return data_tensor, label_id, valid_len
    
    def _apply_augmentation(self, data):
        """
        应用随机旋转和缩放的坐标变换。
        
        Args:
            data (np.ndarray): 形状为 (Frames, 2, 135) 的特征数据。
                              第二维的 0 是 X 坐标，1 是 Y 坐标。
        
        Returns:
            np.ndarray: 变换后的特征数据，形状不变。
        """
        # 随机旋转角度：可配置，默认约 ±20°，保持适度
        angle = np.random.uniform(-cfg.AUG_ROTATION_RANGE, cfg.AUG_ROTATION_RANGE)
        theta = np.radians(angle)
        
        # 随机缩放因子：适度 0.85~1.15
        scale = np.random.uniform(cfg.AUG_SCALE_MIN, cfg.AUG_SCALE_MAX)

        # 随机平移：可配置，默认 ±0.12
        tx = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)
        ty = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)
        
        # 构建旋转矩阵
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # 应用变换
        # data 形状: (Frames, 2, 135)
        transformed_data = data.copy()
        
        for frame_idx in range(data.shape[0]):
            # 提取当前帧的所有关键点坐标 (2, 135)
            frame_coords = data[frame_idx]  # shape: (2, 135)
            
            # 1. 旋转
            # 转置为 (135, 2) 以便进行矩阵乘法
            coords_transposed = frame_coords.T  # shape: (135, 2)
            # (135, 2) @ (2, 2)^T = (135, 2)
            coords_rotated = coords_transposed @ rotation_matrix.T
            
            # 2. 缩放
            coords_scaled = coords_rotated * scale

            # 3. 平移 (广播加法)
            coords_translated = coords_scaled + np.array([tx, ty])
            
            # 转置回 (2, 135)
            transformed_data[frame_idx] = coords_translated.T

        # 4. 高斯噪声 (Gaussian Noise)
        # 对每个时间步的每个关键点添加独立噪声
        noise = np.random.normal(loc=0.0, scale=cfg.AUG_NOISE_STD, size=transformed_data.shape)
        transformed_data = transformed_data + noise

        # 5. 水平翻转（适度，默认 30% 概率）
        if np.random.random() < cfg.AUG_HFLIP_PROB:
            # 翻转 X 坐标，假设已归一化到 0~1
            transformed_data[:, 0, :] = 1.0 - transformed_data[:, 0, :]
            # 这里不交换左右关键点索引，简化处理
        
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
