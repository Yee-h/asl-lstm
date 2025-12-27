#这个文件负责将 .npy 文件加载为 PyTorch Tensor，并处理变长序列的零填充。

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json
import sys

# 动态添加路径以支持从项目根目录运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.utils import load_npy_feature

class CSLDataset(Dataset):
    def __init__(self, split, label_map=None):
        """
        Args:
            split: 'train', 'test', 'dev'
            label_map: 词汇表字典 {'词语': 0, ...}。训练集需要构建，验证/测试集传入。
        """
        self.split = split
        self.data_list = self._load_data_info()
        
        # 处理标签映射 (Vocabulary)
        if label_map is None:
            self.label_map, self.idx_to_label = self._build_vocab()
        else:
            self.label_map = label_map
            self.idx_to_label = {v: k for k, v in label_map.items()}

    def _load_data_info(self):
        json_path = os.path.join(cfg.PROCESSED_LABELS_PATH, f'{self.split}_labels.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到标签文件: {json_path}. 请先运行预处理。")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_vocab(self):
        # 统计所有唯一的中文词汇，分配 ID
        unique_words = sorted(list(set([item['chinese'] for item in self.data_list])))
        label_map = {word: idx for idx, word in enumerate(unique_words)}
        idx_to_label = {idx: word for word, idx in label_map.items()}
        print(f"[{self.split}] 构建词汇表完成: 共 {len(label_map)} 个类别")
        return label_map, idx_to_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        npy_path = item['npy_path']
        chinese_label = item['chinese']
        
        # 1. 加载特征 (Frames, 225)
        # 注意：这里我们信任预处理生成的路径，如果是跨平台运行可能需要处理路径分隔符
        if not os.path.exists(npy_path):
             # 尝试修复路径 (应对从 linux 拷贝到 windows 或反之的情况)
             rel_path = npy_path.split('dataset')[-1].strip(os.sep)
             npy_path = os.path.join(cfg.DATASET_PATH, rel_path)

        features = load_npy_feature(npy_path)
        
        # 2. 转换为 FloatTensor
        features_tensor = torch.FloatTensor(features)
        
        # 3. 获取标签 ID
        label_id = self.label_map.get(chinese_label, -1)
        
        if label_id == -1:
            raise ValueError(f"未知标签: {chinese_label}")
        
        return features_tensor, label_id

def collate_fn(batch):
    """
    处理变长序列的 Collate Function
    """
    # batch: list of (features_tensor, label_id)
    features, labels = zip(*batch)
    
    # 1. 记录原始长度 (CPU Tensor 用于 pack_padded_sequence)
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    
    # 2. 填充特征 (Batch, Max_Len, Dim)
    # padding_value=0
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    
    # 3. 转换标签
    labels_tensor = torch.LongTensor(labels)
    
    return features_padded, labels_tensor, lengths