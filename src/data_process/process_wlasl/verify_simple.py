"""简单验证脚本 - 验证视频标签映射"""
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# 加载 class_list
class_map = {}
with open(os.path.join(base_dir, 'split_video', 'wlasl_class_list.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            class_map[int(parts[0])] = parts[1]

# 加载 nslt_100.json
with open(os.path.join(base_dir, 'nslt_100.json'), 'r', encoding='utf-8') as f:
    nslt_data = json.load(f)

# 加载 WLASL_v0.3.json
with open(os.path.join(base_dir, 'WLASL_v0.3.json'), 'r', encoding='utf-8') as f:
    wlasl_data = json.load(f)

# 构建 video_id -> gloss 映射
video_to_gloss = {}
gloss_to_idx = {}
for idx, entry in enumerate(wlasl_data):
    gloss = entry['gloss']
    gloss_to_idx[gloss] = idx
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_to_gloss[video_id] = gloss

# 验证所有 nslt_100 中的视频
print("=" * 70)
print("WLASL100 Video-Label Mapping Verification")
print("=" * 70)

matches = 0
mismatches = 0
mismatch_details = []

for vid, info in nslt_data.items():
    action_idx = info['action'][0]
    word_from_class_list = class_map.get(action_idx, 'NOT_FOUND')
    word_from_wlasl = video_to_gloss.get(vid, 'NOT_FOUND')
    
    if word_from_class_list == word_from_wlasl:
        matches += 1
    else:
        mismatches += 1
        mismatch_details.append(f"  {vid}: idx={action_idx} -> class_list='{word_from_class_list}' vs WLASL='{word_from_wlasl}'")

print(f"\n[1] nslt_100.json vs WLASL_v0.3.json:")
print(f"    Total videos: {len(nslt_data)}")
print(f"    Matches: {matches}")
print(f"    Mismatches: {mismatches}")

if mismatch_details:
    print("\n    Mismatch details (first 10):")
    for detail in mismatch_details[:10]:
        print(detail)

# 验证词汇顺序
print(f"\n[2] Vocabulary order (first 100 words):")
order_matches = 0
for idx in range(100):
    if idx in class_map:
        word = class_map[idx]
        if word in gloss_to_idx and gloss_to_idx[word] == idx:
            order_matches += 1

print(f"    Order matches: {order_matches}/100")

# 验证 HDF5 标签
print(f"\n[3] HDF5 Label Verification:")
try:
    import h5py
    
    maplabels_path = os.path.join(base_dir, 'pose_action_dataset', 'WLASL100', 'wlasl_100_maplabels.json')
    with open(maplabels_path, 'r', encoding='utf-8') as f:
        maplabels = json.load(f)
    
    label_to_word = maplabels['label_to_id']  # number -> word
    
    for subset in ['Train', 'Val', 'Test']:
        hdf5_path = os.path.join(base_dir, 'pose_action_dataset', 'WLASL100', f'WLASL100_135-{subset}.hdf5')
        if os.path.exists(hdf5_path):
            h5_matches = 0
            h5_total = 0
            with h5py.File(hdf5_path, 'r') as h5f:
                for key in h5f.keys():
                    if key.startswith('label_'):
                        video_id = key.replace('label_', '')
                        hdf5_label = int(h5f[key][()])
                        hdf5_word = label_to_word.get(str(hdf5_label), 'NOT_FOUND')
                        
                        if video_id in nslt_data:
                            action_idx = nslt_data[video_id]['action'][0]
                            nslt_word = class_map.get(action_idx, 'NOT_FOUND')
                            h5_total += 1
                            if hdf5_word == nslt_word:
                                h5_matches += 1
            
            if h5_total > 0:
                print(f"    {subset}: {h5_matches}/{h5_total} match ({100*h5_matches/h5_total:.1f}%)")
except ImportError:
    print("    h5py not installed, skipping HDF5 verification")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 70)
if mismatches == 0 and order_matches == 100:
    print("PASSED: All video-label mappings are correct!")
else:
    print("WARNING: Some mismatches found, please check above.")
print("=" * 70)
