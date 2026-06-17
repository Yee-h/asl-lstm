"""
验证视频文件与标签的对应关系是否正确

此脚本会：
1. 对比 nslt_xxx.json 中的 action 索引与 wlasl_class_list.txt 的映射
2. 与原始 WLASL_v0.3.json 中的 video_id -> gloss 映射进行交叉验证
3. 验证 pose_action_dataset 中 HDF5 文件的标签与词汇表的一致性
"""

import json
import os
import random
import argparse

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("警告: 未安装 h5py，将跳过 HDF5 文件验证")


def load_class_list(path: str) -> dict:
    """加载 wlasl_class_list.txt"""
    class_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                class_map[int(parts[0])] = parts[1]
    return class_map


def load_wlasl_v3(path: str) -> dict:
    """加载 WLASL_v0.3.json 并构建 video_id -> gloss 映射"""
    with open(path, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)
    
    video_to_gloss = {}
    gloss_index = {}  # gloss -> 在列表中的位置（即类别索引）
    
    for idx, entry in enumerate(wlasl_data):
        gloss = entry['gloss']
        gloss_index[gloss] = idx
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_to_gloss[video_id] = gloss
    
    return video_to_gloss, gloss_index


def verify_nslt_mapping(nslt_path: str, class_map: dict, video_to_gloss: dict, sample_size: int = None):
    """验证 nslt JSON 中的映射"""
    with open(nslt_path, 'r', encoding='utf-8') as f:
        nslt_data = json.load(f)
    
    video_ids = list(nslt_data.keys())
    
    if sample_size:
        random.seed(42)
        video_ids = random.sample(video_ids, min(sample_size, len(video_ids)))
    
    matches = 0
    mismatches = []
    
    for vid in video_ids:
        action_idx = nslt_data[vid]['action'][0]
        word_from_class_list = class_map.get(action_idx, 'NOT_FOUND')
        word_from_wlasl = video_to_gloss.get(vid, 'NOT_FOUND')
        
        if word_from_class_list == word_from_wlasl:
            matches += 1
        else:
            mismatches.append({
                'video_id': vid,
                'action_idx': action_idx,
                'class_list_word': word_from_class_list,
                'wlasl_word': word_from_wlasl
            })
    
    return matches, mismatches, len(video_ids)


def verify_hdf5_labels(hdf5_path: str, maplabels_path: str, nslt_path: str, class_map: dict):
    """验证 HDF5 文件中的标签与 nslt JSON 是否一致"""
    if not HAS_H5PY:
        return None, None, None
    
    with open(maplabels_path, 'r', encoding='utf-8') as f:
        maplabels = json.load(f)
    
    with open(nslt_path, 'r', encoding='utf-8') as f:
        nslt_data = json.load(f)
    
    label_to_id = maplabels.get('label_to_id', {})
    
    matches = 0
    mismatches = []
    total = 0
    
    with h5py.File(hdf5_path, 'r') as h5f:
        for key in h5f.keys():
            if key.startswith('label_'):
                video_id = key.replace('label_', '')
                hdf5_label = int(h5f[key][()])
                
                # 从 HDF5 标签获取词汇
                hdf5_word = label_to_id.get(str(hdf5_label), 'NOT_FOUND')
                
                # 从 nslt 获取原始类别索引
                if video_id in nslt_data:
                    action_idx = nslt_data[video_id]['action'][0]
                    nslt_word = class_map.get(action_idx, 'NOT_FOUND')
                    
                    if hdf5_word == nslt_word:
                        matches += 1
                    else:
                        mismatches.append({
                            'video_id': video_id,
                            'hdf5_label': hdf5_label,
                            'hdf5_word': hdf5_word,
                            'nslt_action_idx': action_idx,
                            'nslt_word': nslt_word
                        })
                    total += 1
    
    return matches, mismatches, total


def main():
    parser = argparse.ArgumentParser(description='验证视频标签映射的正确性')
    parser.add_argument('--scale', type=int, choices=[100, 300, 2000], default=100)
    parser.add_argument('--sample', type=int, default=None, help='随机抽样数量，不指定则验证全部')
    parser.add_argument('--show-mismatches', action='store_true', help='显示不匹配的详细信息')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    class_list_path = os.path.join(base_dir, 'split_video', 'wlasl_class_list.txt')
    wlasl_v3_path = os.path.join(base_dir, 'WLASL_v0.3.json')
    nslt_path = os.path.join(base_dir, f'nslt_{args.scale}.json')
    
    pose_dataset_dir = os.path.join(base_dir, 'pose_action_dataset', f'WLASL{args.scale}')
    maplabels_path = os.path.join(pose_dataset_dir, f'wlasl_{args.scale}_maplabels.json')
    
    print("=" * 70)
    print(f"WLASL{args.scale} 视频-标签映射验证")
    print("=" * 70)
    
    # 加载数据
    print("\n[1] 加载映射数据...")
    class_map = load_class_list(class_list_path)
    print(f"    - wlasl_class_list.txt: {len(class_map)} 个词汇")
    
    video_to_gloss, gloss_index = load_wlasl_v3(wlasl_v3_path)
    print(f"    - WLASL_v0.3.json: {len(video_to_gloss)} 个视频, {len(gloss_index)} 个词汇")
    
    # 验证 nslt 映射
    print(f"\n[2] 验证 nslt_{args.scale}.json 与 WLASL_v0.3.json 的一致性...")
    matches, mismatches, total = verify_nslt_mapping(nslt_path, class_map, video_to_gloss, args.sample)
    
    print(f"    - 验证视频数: {total}")
    print(f"    - 匹配: {matches} ({100*matches/total:.2f}%)")
    print(f"    - 不匹配: {len(mismatches)}")
    
    if mismatches and args.show_mismatches:
        print("\n    不匹配详情:")
        for m in mismatches[:10]:
            print(f"      - {m['video_id']}: idx={m['action_idx']} -> "
                  f"class_list=\"{m['class_list_word']}\" vs WLASL=\"{m['wlasl_word']}\"")
        if len(mismatches) > 10:
            print(f"      ... 省略 {len(mismatches) - 10} 条")
    
    # 验证 HDF5 标签
    if HAS_H5PY and os.path.exists(maplabels_path):
        print(f"\n[3] 验证 pose_action_dataset HDF5 标签一致性...")
        
        for subset in ['Train', 'Val', 'Test']:
            hdf5_path = os.path.join(pose_dataset_dir, f'WLASL{args.scale}_135-{subset}.hdf5')
            if os.path.exists(hdf5_path):
                h5_matches, h5_mismatches, h5_total = verify_hdf5_labels(
                    hdf5_path, maplabels_path, nslt_path, class_map
                )
                if h5_total:
                    print(f"    - {subset}: {h5_matches}/{h5_total} 匹配 ({100*h5_matches/h5_total:.2f}%)")
                    if h5_mismatches and args.show_mismatches:
                        for m in h5_mismatches[:3]:
                            print(f"        不匹配: {m['video_id']}: HDF5={m['hdf5_word']} vs NSLT={m['nslt_word']}")
    
    # 验证前100个词汇的顺序
    print(f"\n[4] 验证 wlasl_class_list.txt 与 WLASL_v0.3.json 的词汇顺序...")
    order_matches = 0
    order_mismatches = []
    for idx in range(min(100, len(class_map))):
        if idx in class_map:
            word = class_map[idx]
            if word in gloss_index:
                wlasl_idx = gloss_index[word]
                if wlasl_idx == idx:
                    order_matches += 1
                else:
                    order_mismatches.append((idx, word, wlasl_idx))
    
    print(f"    - 前100个词汇顺序匹配: {order_matches}/100")
    if order_mismatches and args.show_mismatches:
        for idx, word, wlasl_idx in order_mismatches[:5]:
            print(f"      - idx={idx}: \"{word}\" 在 WLASL 中的索引为 {wlasl_idx}")
    
    print("\n" + "=" * 70)
    if len(mismatches) == 0 and order_matches == 100:
        print("✓ 验证通过！视频和标签映射完全正确。")
    else:
        print("⚠ 存在一些不匹配，请检查上述详情。")
    print("=" * 70)


if __name__ == '__main__':
    main()
