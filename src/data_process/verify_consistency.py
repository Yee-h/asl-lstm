"""
验证预处理数据与官方数据一致性的脚本。

验证项：
1. 标签映射文件是否满足统一契约。
2. 预处理 HDF5 与官方 HDF5 在样本数、标签分布、视频集合上的一致性。
3. HDF5 样本是否满足必填字段约束。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg
from src.core.hdf5_schema import validate_required_fields
from src.core.labels import load_label_to_id_map
from src.core.paths import dataset_subset_dir, label_map_file_path, project_root


def _build_generated_label_to_id(subset: int) -> dict[str, int]:
    raw_json_path = Path(cfg.PATHS.raw_data_dir) / "WLASL_v0.3.json"
    if not raw_json_path.exists():
        raise FileNotFoundError(f"未找到原始标签文件: {raw_json_path}")

    with open(raw_json_path, "r", encoding="utf-8") as file_obj:
        wlasl_data = json.load(file_obj)

    generated: dict[str, int] = {}
    for idx in range(min(subset, len(wlasl_data))):
        generated[wlasl_data[idx]["gloss"]] = idx
    return generated


def verify_maplabels_from_raw(subset: int, official_map_path: Path) -> bool:
    print("\n=== 验证标签映射生成逻辑 ===")
    try:
        generated_map = _build_generated_label_to_id(subset)
    except Exception as exc:
        print(f"✗ 构建标签映射失败: {exc}")
        return False

    print(f"  从原始数据构建标签映射: {len(generated_map)} 项")

    if not official_map_path.exists():
        print(f"  警告: 未找到官方标签映射文件 {official_map_path}，跳过对比")
        return True

    try:
        official_map = load_label_to_id_map(official_map_path)
    except Exception as exc:
        print(f"  ✗ 官方标签映射不符合统一契约: {exc}")
        return False

    if generated_map == official_map:
        print(f"  ✓ 标签映射与官方一致: {len(official_map)} 项")
        return True

    print("  ✗ 标签映射与官方不一致")
    diff_count = 0
    for label, idx in generated_map.items():
        official_idx = official_map.get(label)
        if official_idx != idx:
            print(f"    差异: {label} -> 生成={idx}, 官方={official_idx}")
            diff_count += 1
            if diff_count >= 5:
                print("    ... (更多差异省略)")
                break
    return False


def _collect_label_distribution(h5_file: h5py.File) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for key in h5_file.keys():
        sample = h5_file[key]
        if not isinstance(sample, h5py.Group):
            continue
        validate_required_fields(sample.keys())

        label_data = sample["label"][()]
        if isinstance(label_data, bytes):
            label = label_data.decode("utf-8")
        else:
            label = str(label_data)
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def _collect_video_ids(h5_file: h5py.File) -> set[str]:
    video_ids: set[str] = set()
    for key in h5_file.keys():
        sample = h5_file[key]
        if not isinstance(sample, h5py.Group):
            continue
        validate_required_fields(sample.keys())

        video_name_data = sample["video_name"][()]
        if isinstance(video_name_data, bytes):
            video_name = video_name_data.decode("utf-8")
        else:
            video_name = str(video_name_data)
        video_ids.add(Path(video_name).name.replace(".mp4", ""))
    return video_ids


def compare_hdf5_structure(official_path: Path, processed_path: Path) -> bool:
    is_consistent = True
    with (
        h5py.File(official_path, "r") as official_file,
        h5py.File(processed_path, "r") as processed_file,
    ):
        official_count = len(official_file.keys())
        processed_count = len(processed_file.keys())
        if official_count == processed_count:
            print(f"  ✓ 样本数量一致: {official_count}")
        else:
            print(
                f"  ✗ 样本数量不一致: 官方 {official_count}, 预处理 {processed_count}"
            )
            is_consistent = False

        try:
            official_labels = _collect_label_distribution(official_file)
            processed_labels = _collect_label_distribution(processed_file)
            if official_labels == processed_labels:
                print(f"  ✓ 标签分布一致: {len(official_labels)} 个类别")
            else:
                print("  ✗ 标签分布不一致")
                is_consistent = False

            official_videos = _collect_video_ids(official_file)
            processed_videos = _collect_video_ids(processed_file)
            if official_videos == processed_videos:
                print(f"  ✓ 视频集合一致: {len(official_videos)} 个视频")
            else:
                print("  ✗ 视频集合不一致")
                is_consistent = False
        except ValueError as exc:
            print(f"  ✗ HDF5 必填字段校验失败: {exc}")
            is_consistent = False

    return is_consistent


def verify_processed_maplabels(official_map: Path, processed_map: Path) -> bool:
    if not official_map.exists() or not processed_map.exists():
        print("  警告: 官方或预处理标签映射文件缺失，跳过映射对比")
        return True

    try:
        official_label_to_id = load_label_to_id_map(official_map)
        processed_label_to_id = load_label_to_id_map(processed_map)
    except Exception as exc:
        print(f"  ✗ 标签映射契约校验失败: {exc}")
        return False

    if official_label_to_id == processed_label_to_id:
        print("  ✓ 预处理标签映射与官方一致")
        return True

    print("  ✗ 预处理标签映射与官方不一致")
    return False


def main() -> int:
    subset = int(cfg.PATHS.dataset_scale)
    root = project_root()
    official_dir = root / f"WLASL{subset}"
    processed_dir = dataset_subset_dir(subset)

    official_map = official_dir / f"wlasl_{subset}_maplabels.json"
    processed_map = label_map_file_path(subset)

    print("=" * 60)
    print(f"验证 WLASL{subset} 数据一致性")
    print("=" * 60)

    all_consistent = True
    if not verify_maplabels_from_raw(subset, official_map):
        all_consistent = False

    if processed_dir.exists():
        print("\n=== 对比预处理数据 ===")
        print(f"预处理目录: {processed_dir}")

        if not verify_processed_maplabels(official_map, processed_map):
            all_consistent = False

        for split in ("Train", "Val", "Test"):
            official_hdf5 = official_dir / f"WLASL{subset}_135-{split}.hdf5"
            processed_hdf5 = processed_dir / f"WLASL{subset}_135-{split}.hdf5"
            if official_hdf5.exists() and processed_hdf5.exists():
                print(f"\n  {split} 数据集:")
                if not compare_hdf5_structure(official_hdf5, processed_hdf5):
                    all_consistent = False
    else:
        print(f"\n注意: 预处理目录不存在，跳过 HDF5 对比: {processed_dir}")

    print("\n" + "=" * 60)
    if all_consistent:
        print("✓ 验证通过")
    else:
        print("✗ 验证失败")
    print("=" * 60)
    return 0 if all_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
