"""
验证预处理数据与官方数据一致性的脚本

此脚本验证：
1. 从 WLASL_v0.3.json 生成的标签映射是否与官方一致
2. 预处理后的 HDF5 文件结构是否与官方一致

使用方法：
    uv run python src/data_process/verify_consistency.py
"""

import h5py
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


def verify_maplabels_from_raw(subset: int) -> bool:
    """
    验证从 WLASL_v0.3.json 生成的标签映射是否与官方一致

    Args:
        subset: 数据集规模 (100/300/1000/2000)

    Returns:
        True 如果一致，False 如果不一致
    """
    print("\n=== 验证标签映射生成逻辑 ===")

    # 从 WLASL_v0.3.json 读取 gloss 顺序
    wlasl_json_path = os.path.join(cfg.PATHS.raw_data_dir, "WLASL_v0.3.json")
    if not os.path.exists(wlasl_json_path):
        print(f"警告: 未找到 {wlasl_json_path}")
        return False

    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    # 构建标签映射
    generated_mapping = {}
    for idx in range(min(subset, len(wlasl_data))):
        gloss = wlasl_data[idx]["gloss"]
        generated_mapping[gloss] = idx

    print(f"  从 WLASL_v0.3.json 生成了 {len(generated_mapping)} 个标签映射")

    # 读取官方映射
    official_path = os.path.join(
        cfg.PATHS.project_root, f"WLASL{subset}", f"wlasl_{subset}_maplabels.json"
    )
    if not os.path.exists(official_path):
        print(f"  警告: 未找到官方映射文件 {official_path}，跳过对比")
        return True  # 没有官方文件可对比，视为通过

    with open(official_path, "r", encoding="utf-8") as f:
        official = json.load(f)

    official_mapping = official.get("id_to_label", {})

    # 对比
    matches = 0
    for gloss, idx in official_mapping.items():
        if generated_mapping.get(gloss) == idx:
            matches += 1

    if matches == len(official_mapping):
        print(f"  ✓ 所有 {matches} 个标签映射与官方完全一致")
        return True
    else:
        print(f"  ✗ 只有 {matches}/{len(official_mapping)} 个标签映射与官方一致")
        # 显示前5个不一致的
        diff_count = 0
        for gloss, idx in official_mapping.items():
            gen_idx = generated_mapping.get(gloss)
            if gen_idx != idx:
                print(f"    差异: {gloss} - 生成: {gen_idx}, 官方: {idx}")
                diff_count += 1
                if diff_count >= 5:
                    print("    ... (更多差异省略)")
                    break
        return False


def compare_hdf5_structure(official_path: str, processed_path: str) -> bool:
    """
    对比 HDF5 文件结构
    """
    is_consistent = True

    with (
        h5py.File(official_path, "r") as f_official,
        h5py.File(processed_path, "r") as f_processed,
    ):
        # 样本数量
        official_count = len(f_official.keys())
        processed_count = len(f_processed.keys())

        if official_count == processed_count:
            print(f"  ✓ 样本数量一致: {official_count}")
        else:
            print(
                f"  ✗ 样本数量不一致: 官方 {official_count}, 预处理 {processed_count}"
            )
            is_consistent = False

        # 标签分布
        official_labels = {}
        for key in f_official.keys():
            label = f_official[key]["label"][()]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            official_labels[label] = official_labels.get(label, 0) + 1

        processed_labels = {}
        for key in f_processed.keys():
            label = f_processed[key]["label"][()]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            processed_labels[label] = processed_labels.get(label, 0) + 1

        if official_labels == processed_labels:
            print(f"  ✓ 标签分布一致: {len(official_labels)} 个类别")
        else:
            print(f"  ✗ 标签分布不一致")
            is_consistent = False

        # 视频集合
        official_videos = set()
        for key in f_official.keys():
            video_name = f_official[key]["video_name"][()]
            if isinstance(video_name, bytes):
                video_name = video_name.decode("utf-8")
            video_id = video_name.split("/")[-1].replace(".mp4", "")
            official_videos.add(video_id)

        processed_videos = set()
        for key in f_processed.keys():
            video_name = f_processed[key]["video_name"][()]
            if isinstance(video_name, bytes):
                video_name = video_name.decode("utf-8")
            video_id = video_name.split("/")[-1].replace(".mp4", "")
            processed_videos.add(video_id)

        if official_videos == processed_videos:
            print(f"  ✓ 视频集合一致: {len(official_videos)} 个视频")
        else:
            print("  ✗ 视频集合不一致")
            is_consistent = False

    return is_consistent


def main():
    """主函数"""
    subset = cfg.PATHS.dataset_scale

    print("=" * 60)
    print(f"验证 WLASL{subset} 数据一致性")
    print("=" * 60)

    all_consistent = True

    # 1. 验证标签映射生成逻辑（从原始数据生成）
    if not verify_maplabels_from_raw(subset):
        all_consistent = False

    # 2. 如果预处理数据存在，对比 HDF5 文件(可选)
    official_dir = os.path.join(cfg.PATHS.project_root, f"WLASL{subset}")
    processed_dir = cfg.PATHS.processed_data_dir

    if os.path.exists(processed_dir):
        print(f"\n=== 对比预处理数据 ===")
        print(f"预处理数据目录: {processed_dir}")

        # 对比标签映射文件
        official_maplabels = os.path.join(
            official_dir, f"wlasl_{subset}_maplabels.json"
        )
        processed_maplabels = os.path.join(
            processed_dir, f"wlasl_{subset}_maplabels.json"
        )

        if os.path.exists(official_maplabels) and os.path.exists(processed_maplabels):
            with open(official_maplabels, "r", encoding="utf-8") as f:
                official = json.load(f)
            with open(processed_maplabels, "r", encoding="utf-8") as f:
                processed = json.load(f)

            if official.get("id_to_label") == processed.get("id_to_label"):
                print("  ✓ 预处理的标签映射与官方一致")
            else:
                print("  ✗ 预处理的标签映射与官方不一致")
                all_consistent = False

        # 对比 HDF5 文件
        splits = ["Train", "Val", "Test"]
        for split in splits:
            official_hdf5 = os.path.join(
                official_dir, f"WLASL{subset}_135-{split}.hdf5"
            )
            processed_hdf5 = os.path.join(
                processed_dir, f"WLASL{subset}_135-{split}.hdf5"
            )

            if os.path.exists(official_hdf5) and os.path.exists(processed_hdf5):
                print(f"\n  {split} 数据集:")
                if not compare_hdf5_structure(official_hdf5, processed_hdf5):
                    all_consistent = False
    else:
        print(f"\n注意: 预处理数据目录 {processed_dir} 不存在，跳过 HDF5 对比")

    # 输出结论
    print("\n" + "=" * 60)
    if all_consistent:
        print("✓ 验证通过")
    else:
        print("✗ 验证失败")
    print("=" * 60)

    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
