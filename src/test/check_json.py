import json
import os

# 标签映射 JSON 文件的路径
path = "d:/Document/Code/csl-lstm/dataset/processed/WLASL100/wlasl_100_maplabels.json"

# 读取 JSON 文件
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打印 JSON 对象顶层的所有键
print("JSON 中的键:", list(data.keys()))

# 检查是否存在 'label_to_id' 映射 (标签 -> ID)
if 'label_to_id' in data:
    d = data['label_to_id']
    print("label_to_id 示例:", list(d.items())[:5])
    k, v = list(d.items())[0]
    # 打印第一个键值对的类型，用于调试
    print(f"l2i 键: {k} (类型 {type(k)}), 值: {v} (类型 {type(v)})")

# 检查是否存在 'id_to_label' 映射 (ID -> 标签)
if 'id_to_label' in data:
    d = data['id_to_label']
    print("id_to_label 示例:", list(d.items())[:5])
    k, v = list(d.items())[0]
    # 打印第一个键值对的类型，用于调试
    print(f"i2l 键: {k} (类型 {type(k)}), 值: {v} (类型 {type(v)})")
