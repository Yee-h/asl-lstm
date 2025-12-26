import os
import numpy as np
import config as cfg

# MediaPipe feature extraction logic has been moved to:
# src/data_preprocessing/extract_keypoints.py
# src/data_preprocessing/normalize.py



def load_processed_labels(split: str) -> list:
    """
    加载处理后的标签文件
    
    Args:
        split: 数据集划分 ('train', 'test', 'dev')
        
    Returns:
        list: 标签信息列表
    """
    import json
    import os
    
    label_file = os.path.join(cfg.PROCESSED_LABELS_PATH, f'{split}_labels.json')
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"处理后的标签文件不存在: {label_file}")
    
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_npy_feature(npy_path: str) -> np.ndarray:
    """
    加载单个 .npy 特征文件
    
    Args:
        npy_path: .npy 文件路径
        
    Returns:
        np.ndarray: 特征数组
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"特征文件不存在: {npy_path}")
    
    return np.load(npy_path)


def get_dataset_stats(split: str | None = None) -> dict:
    """
    获取数据集统计信息
    
    Args:
        split: 数据集划分，None 表示所有划分
        
    Returns:
        dict: 统计信息
    """
    
    splits = [split] if split else cfg.SPLITS
    stats = {}
    
    for s in splits:
        processed_path = os.path.join(cfg.PROCESSED_DATA_PATH, s)
        if not os.path.exists(processed_path):
            stats[s] = {'count': 0, 'translators': {}}
            continue
        
        translator_stats = {}
        total_count = 0
        
        for translator in cfg.TRANSLATORS:
            translator_path = os.path.join(processed_path, translator)
            if os.path.exists(translator_path):
                count = len([f for f in os.listdir(translator_path) if f.endswith('.npy')])
                translator_stats[translator] = count
                total_count += count
        
        stats[s] = {
            'count': total_count,
            'translators': translator_stats
        }
    
    return stats