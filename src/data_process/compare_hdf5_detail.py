"""
详细对比官方数据和预处理数据的脚本

此脚本会逐项对比两个 HDF5 文件的所有内容，包括：
1. 标签映射文件格式和顺序
2. HDF5 文件结构
3. 每个样本的具体数据（关键点坐标、标签、视频名等）
4. 数据的统计特征（均值、标准差、范围等）

使用方法：
    uv run python src/data_process/compare_hdf5_detail.py
"""
import h5py
import json
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg


def decode_bytes(value):
    """解码 bytes 类型的值"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def compare_maplabels():
    """对比标签映射文件"""
    subset = cfg.DATASET_SCALE
    
    official_path = os.path.join(cfg.PROJECT_ROOT, f"WLASL{subset}", f"wlasl_{subset}_maplabels.json")
    processed_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"wlasl_{subset}_maplabels.json")
    
    print("=" * 80)
    print("1. 标签映射文件对比")
    print("=" * 80)
    print(f"官方文件: {official_path}")
    print(f"预处理文件: {processed_path}")
    
    with open(official_path, 'r', encoding='utf-8') as f:
        official = json.load(f)
    with open(processed_path, 'r', encoding='utf-8') as f:
        processed = json.load(f)
    
    # 对比结构
    print(f"\n官方文件的键: {list(official.keys())}")
    print(f"预处理文件的键: {list(processed.keys())}")
    
    # 对比 id_to_label
    o_id_to_label = official.get('id_to_label', {})
    p_id_to_label = processed.get('id_to_label', {})
    
    print(f"\n官方 id_to_label 数量: {len(o_id_to_label)}")
    print(f"预处理 id_to_label 数量: {len(p_id_to_label)}")
    
    # 显示前10个映射
    print("\n前10个 id_to_label 映射对比:")
    print(f"{'标签':<15} {'官方ID':<8} {'预处理ID':<8} {'结果'}")
    print("-" * 50)
    
    all_labels = sorted(set(o_id_to_label.keys()) | set(p_id_to_label.keys()))
    diff_count = 0
    for label in all_labels[:10]:
        o_id = o_id_to_label.get(label, 'N/A')
        p_id = p_id_to_label.get(label, 'N/A')
        match = "[OK]" if o_id == p_id else "[FAIL]"
        if o_id != p_id:
            diff_count += 1
        print(f"{label:<15} {str(o_id):<8} {str(p_id):<8} {match}")
    
    # 统计差异
    total_diff = sum(1 for l in all_labels if o_id_to_label.get(l) != p_id_to_label.get(l))
    print(f"\n总差异数: {total_diff} / {len(all_labels)}")
    
    return total_diff == 0


def compare_hdf5_structure(split: str):
    """对比 HDF5 文件结构"""
    subset = cfg.DATASET_SCALE
    
    official_path = os.path.join(cfg.PROJECT_ROOT, f"WLASL{subset}", f"WLASL{subset}_135-{split}.hdf5")
    processed_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"WLASL{subset}_135-{split}.hdf5")
    
    print("\n" + "=" * 80)
    print(f"2. {split} HDF5 文件结构对比")
    print("=" * 80)
    print(f"官方文件: {official_path}")
    print(f"预处理文件: {processed_path}")
    
    with h5py.File(official_path, 'r') as f_o, h5py.File(processed_path, 'r') as f_p:
        # 基本信息
        o_keys = sorted(f_o.keys(), key=lambda x: int(x) if x.isdigit() else x)
        p_keys = sorted(f_p.keys(), key=lambda x: int(x) if x.isdigit() else x)
        
        print(f"\n官方样本数: {len(o_keys)}")
        print(f"预处理样本数: {len(p_keys)}")
        print(f"前5个 Group 名: 官方 {o_keys[:5]} vs 预处理 {p_keys[:5]}")
        
        # 构建 video_id 到 group 的映射
        o_video_to_group = {}
        p_video_to_group = {}
        
        for key in o_keys:
            video_name = decode_bytes(f_o[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            o_video_to_group[video_id] = key
        
        for key in p_keys:
            video_name = decode_bytes(f_p[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            p_video_to_group[video_id] = key
        
        # 检查视频集合是否一致
        o_videos = set(o_video_to_group.keys())
        p_videos = set(p_video_to_group.keys())
        
        common_videos = o_videos & p_videos
        missing_in_processed = o_videos - p_videos
        extra_in_processed = p_videos - o_videos
        
        print(f"\n共同视频数: {len(common_videos)}")
        print(f"预处理缺少: {len(missing_in_processed)}")
        print(f"预处理多出: {len(extra_in_processed)}")
        
        if missing_in_processed:
            print(f"  缺少的视频: {list(missing_in_processed)[:5]}...")
        if extra_in_processed:
            print(f"  多出的视频: {list(extra_in_processed)[:5]}...")
        
        return o_video_to_group, p_video_to_group, f_o, f_p


def compare_sample_data(split: str, num_samples: int = 5):
    """详细对比几个样本的具体数据"""
    subset = cfg.DATASET_SCALE
    
    official_path = os.path.join(cfg.PROJECT_ROOT, f"WLASL{subset}", f"WLASL{subset}_135-{split}.hdf5")
    processed_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"WLASL{subset}_135-{split}.hdf5")
    
    print("\n" + "=" * 80)
    print(f"3. {split} 样本数据详细对比 (前 {num_samples} 个共同视频)")
    print("=" * 80)
    
    with h5py.File(official_path, 'r') as f_o, h5py.File(processed_path, 'r') as f_p:
        # 构建 video_id 映射
        o_video_to_group = {}
        p_video_to_group = {}
        
        for key in f_o.keys():
            video_name = decode_bytes(f_o[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            o_video_to_group[video_id] = key
        
        for key in f_p.keys():
            video_name = decode_bytes(f_p[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            p_video_to_group[video_id] = key
        
        # 找出共同视频
        common_videos = sorted(set(o_video_to_group.keys()) & set(p_video_to_group.keys()))
        
        for i, video_id in enumerate(common_videos[:num_samples]):
            o_key = o_video_to_group[video_id]
            p_key = p_video_to_group[video_id]
            
            print(f"\n--- 视频 {video_id} ---")
            print(f"官方 Group: {o_key}, 预处理 Group: {p_key}")
            
            o_grp = f_o[o_key]
            p_grp = f_p[p_key]
            
            # 对比每个字段
            for field in ['label', 'video_name', 'width', 'height']:
                o_val = decode_bytes(o_grp[field][()])
                p_val = decode_bytes(p_grp[field][()])
                match = "[OK]" if o_val == p_val else "[FAIL]"
                print(f"  {field}: 官方={o_val}, 预处理={p_val} {match}")
            
            # 对比数据形状和统计
            o_data = np.array(o_grp['data'])
            p_data = np.array(p_grp['data'])
            
            shape_match = "[OK]" if o_data.shape == p_data.shape else "[FAIL]"
            print(f"  data shape: 官方={o_data.shape}, 预处理={p_data.shape} {shape_match}")
            print(f"  data dtype: 官方={o_data.dtype}, 预处理={p_data.dtype}")
            
            # 数据统计
            print(f"  data mean:  官方={o_data.mean():.6f}, 预处理={p_data.mean():.6f}")
            print(f"  data std:   官方={o_data.std():.6f}, 预处理={p_data.std():.6f}")
            print(f"  data min:   官方={o_data.min():.6f}, 预处理={p_data.min():.6f}")
            print(f"  data max:   官方={o_data.max():.6f}, 预处理={p_data.max():.6f}")
            
            # 如果形状相同，计算差异
            if o_data.shape == p_data.shape:
                diff = np.abs(o_data - p_data)
                print(f"  data diff mean: {diff.mean():.6f}")
                print(f"  data diff max:  {diff.max():.6f}")
                is_same = "[OK]" if np.allclose(o_data, p_data, rtol=1e-5) else "[FAIL]"
                print(f"  data 完全相同: {is_same}")


def compare_data_statistics(split: str):
    """对比整体数据统计"""
    subset = cfg.DATASET_SCALE
    
    official_path = os.path.join(cfg.PROJECT_ROOT, f"WLASL{subset}", f"WLASL{subset}_135-{split}.hdf5")
    processed_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"WLASL{subset}_135-{split}.hdf5")
    
    print("\n" + "=" * 80)
    print(f"4. {split} 整体数据统计对比")
    print("=" * 80)
    
    with h5py.File(official_path, 'r') as f_o, h5py.File(processed_path, 'r') as f_p:
        # 收集所有数据
        o_all_data = []
        p_all_data = []
        o_frame_counts = []
        p_frame_counts = []
        
        for key in f_o.keys():
            data = np.array(f_o[key]['data'])
            o_all_data.append(data)
            o_frame_counts.append(data.shape[0])
        
        for key in f_p.keys():
            data = np.array(f_p[key]['data'])
            p_all_data.append(data)
            p_frame_counts.append(data.shape[0])
        
        # 帧数统计
        print(f"\n帧数统计:")
        print(f"  官方 - 总帧数: {sum(o_frame_counts)}, 平均: {np.mean(o_frame_counts):.1f}, 最小: {min(o_frame_counts)}, 最大: {max(o_frame_counts)}")
        print(f"  预处理 - 总帧数: {sum(p_frame_counts)}, 平均: {np.mean(p_frame_counts):.1f}, 最小: {min(p_frame_counts)}, 最大: {max(p_frame_counts)}")
        
        # 数据值统计
        o_concat = np.concatenate([d.flatten() for d in o_all_data])
        p_concat = np.concatenate([d.flatten() for d in p_all_data])
        
        print(f"\n数据值统计:")
        print(f"  官方 - mean: {o_concat.mean():.6f}, std: {o_concat.std():.6f}, min: {o_concat.min():.6f}, max: {o_concat.max():.6f}")
        print(f"  预处理 - mean: {p_concat.mean():.6f}, std: {p_concat.std():.6f}, min: {p_concat.min():.6f}, max: {p_concat.max():.6f}")
        
        # 零值统计（可能反映关键点检测失败）
        o_zero_ratio = (o_concat == 0).sum() / len(o_concat)
        p_zero_ratio = (p_concat == 0).sum() / len(p_concat)
        print(f"\n零值比例:")
        print(f"  官方: {o_zero_ratio:.4%}")
        print(f"  预处理: {p_zero_ratio:.4%}")


def compare_first_frame_keypoints(split: str, video_idx: int = 0):
    """详细对比第一帧的关键点"""
    subset = cfg.DATASET_SCALE
    
    official_path = os.path.join(cfg.PROJECT_ROOT, f"WLASL{subset}", f"WLASL{subset}_135-{split}.hdf5")
    processed_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"WLASL{subset}_135-{split}.hdf5")
    
    print("\n" + "=" * 80)
    print(f"5. 关键点数据详细对比 (第一个共同视频的第一帧)")
    print("=" * 80)
    
    with h5py.File(official_path, 'r') as f_o, h5py.File(processed_path, 'r') as f_p:
        # 找一个共同视频
        o_video_to_group = {}
        p_video_to_group = {}
        
        for key in f_o.keys():
            video_name = decode_bytes(f_o[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            o_video_to_group[video_id] = key
        
        for key in f_p.keys():
            video_name = decode_bytes(f_p[key]['video_name'][()])
            video_id = video_name.split('/')[-1].replace('.mp4', '')
            p_video_to_group[video_id] = key
        
        common_videos = sorted(set(o_video_to_group.keys()) & set(p_video_to_group.keys()))
        video_id = common_videos[video_idx]
        
        o_key = o_video_to_group[video_id]
        p_key = p_video_to_group[video_id]
        
        o_data = np.array(f_o[o_key]['data'])
        p_data = np.array(f_p[p_key]['data'])
        
        print(f"视频: {video_id}")
        print(f"数据形状: 官方 {o_data.shape}, 预处理 {p_data.shape}")
        
        # 取第一帧对比
        if o_data.shape[0] > 0 and p_data.shape[0] > 0:
            o_frame = o_data[0]  # (2, 135)
            p_frame = p_data[0]  # (2, 135)
            
            print(f"\n第一帧关键点对比 (X 坐标, 前10个关键点):")
            print(f"{'关键点':<8} {'官方X':<12} {'预处理X':<12} {'差值':<12}")
            print("-" * 44)
            for i in range(10):
                o_x = o_frame[0, i]
                p_x = p_frame[0, i] if i < p_frame.shape[1] else 0
                diff = abs(o_x - p_x)
                print(f"{i:<8} {o_x:<12.6f} {p_x:<12.6f} {diff:<12.6f}")
            
            print(f"\n第一帧关键点对比 (Y 坐标, 前10个关键点):")
            print(f"{'关键点':<8} {'官方Y':<12} {'预处理Y':<12} {'差值':<12}")
            print("-" * 44)
            for i in range(10):
                o_y = o_frame[1, i]
                p_y = p_frame[1, i] if i < p_frame.shape[1] else 0
                diff = abs(o_y - p_y)
                print(f"{i:<8} {o_y:<12.6f} {p_y:<12.6f} {diff:<12.6f}")


def main():
    """主函数"""
    print("=" * 80)
    print("WLASL 数据详细对比工具")
    print("=" * 80)
    
    # 1. 对比标签映射
    compare_maplabels()
    
    # 2. 对比 Train 数据
    compare_hdf5_structure("Train")
    compare_sample_data("Train", num_samples=3)
    compare_data_statistics("Train")
    compare_first_frame_keypoints("Train")
    
    print("\n" + "=" * 80)
    print("对比完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
