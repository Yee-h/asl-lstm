"""
数据结构检测工具
用于检测 dataset/processed 下的 HDF5 数据集结构信息。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


def default_data_root() -> Path:
    return Path(cfg.PATHS.project_root) / "dataset" / "processed"


def build_dataset_path(dataset_scale: str, data_root: Path) -> Path:
    return data_root / f"WLASL{dataset_scale}"


def list_target_hdf5_files(dataset_path: Path, split: str) -> list[Path]:
    files = sorted(dataset_path.glob("*.hdf5"))
    if split != "all":
        suffix = f"-{split}.hdf5"
        files = [file_path for file_path in files if file_path.name.endswith(suffix)]
    return files


def print_hdf5_summary(file_path: Path, sample_limit: int | None) -> dict | None:
    print(f"\n{'=' * 70}")
    print(f"文件: {file_path.name}")
    print(f"路径: {file_path.resolve()}")
    print(f"{'=' * 70}")

    try:
        with h5py.File(file_path, "r") as h5_obj:
            sample_keys = list(h5_obj.keys())
            sample_count = len(sample_keys)

            print(f"样本总数: {sample_count}")
            print(f"文件大小: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
            if sample_count == 0:
                return {
                    "sample_count": 0,
                    "label_counts": {},
                }

            first_key = sample_keys[0]
            first_group = h5_obj[first_key]
            print(f"示例样本键: {first_key}")

            if isinstance(first_group, h5py.Group):
                print("示例样本字段:")
                for field_name in sorted(first_group.keys()):
                    field_obj = first_group[field_name]
                    if isinstance(field_obj, h5py.Dataset):
                        shape_text = (
                            str(field_obj.shape) if field_obj.shape != () else "scalar"
                        )
                        print(
                            f"  - {field_name}: dtype={field_obj.dtype}, shape={shape_text}"
                        )

            check_count = (
                sample_count
                if sample_limit is None
                else min(sample_limit, sample_count)
            )
            frame_counts: list[int] = []
            feature_shapes: set[tuple[int, ...]] = set()
            label_counts: dict[str, int] = {}

            for key in sample_keys[:check_count]:
                group_obj = h5_obj[key]
                if not isinstance(group_obj, h5py.Group):
                    continue

                if "data" in group_obj:
                    data_obj = group_obj["data"]
                    if isinstance(data_obj, h5py.Dataset) and len(data_obj.shape) >= 1:
                        frame_counts.append(int(data_obj.shape[0]))
                        feature_shapes.add(
                            tuple(int(dim) for dim in data_obj.shape[1:])
                        )

                if "label" in group_obj:
                    label_obj = group_obj["label"]
                    if isinstance(label_obj, h5py.Dataset):
                        label_val = label_obj[()]
                        if isinstance(label_val, bytes):
                            label_text = label_val.decode("utf-8")
                        else:
                            label_text = str(label_val)
                        label_counts[label_text] = label_counts.get(label_text, 0) + 1

            print(f"检测样本数: {check_count}/{sample_count}")
            if frame_counts:
                print(
                    "帧数统计: "
                    f"mean={np.mean(frame_counts):.2f}, "
                    f"min={np.min(frame_counts)}, "
                    f"max={np.max(frame_counts)}"
                )

            if feature_shapes:
                print("特征形状:")
                for shape in sorted(feature_shapes):
                    print(f"  - {shape}")

            if label_counts:
                print(f"标签类别数: {len(label_counts)}")
                top_items = sorted(
                    label_counts.items(), key=lambda item: item[1], reverse=True
                )[:10]
                print("标签频次 Top10:")
                for label, count in top_items:
                    print(f"  - {label}: {count}")

            return {
                "sample_count": sample_count,
                "label_counts": label_counts,
            }

    except Exception as exc:
        print(f"读取文件失败: {exc}")
        return None


def print_label_map_summary(dataset_path: Path) -> None:
    json_files = sorted(dataset_path.glob("*.json"))
    if not json_files:
        print("未发现标签映射 JSON 文件。")
        return

    print("\n标签映射文件:")
    for json_path in json_files:
        print(f"- {json_path.name}")
        try:
            with open(json_path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                print(f"  顶层键: {list(payload.keys())}")
        except Exception as exc:
            print(f"  读取失败: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测 WLASL 处理后数据结构")
    parser.add_argument(
        "--dataset-scale",
        default=str(cfg.PATHS.dataset_scale),
        choices=["100", "300", "1000", "2000"],
        help="数据集规模，默认读取 config.py 配置",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "Train", "Val", "Test"],
        help="仅检查指定划分，默认检查全部",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="已处理数据根目录，默认 <project>/dataset/processed",
    )
    parser.add_argument(
        "--sample-limit",
        "-n",
        type=int,
        default=None,
        help="每个 HDF5 文件最多检查的样本数，默认检查全部",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = build_dataset_path(args.dataset_scale, args.data_root)

    print("\n" + "#" * 70)
    print("# 数据结构检测工具")
    print("#" * 70)
    print(f"解析后的数据规模: WLASL{args.dataset_scale}")
    print(f"解析后的划分参数: {args.split}")
    print(f"解析后的实际目录: {dataset_path.resolve()}")

    if not dataset_path.exists() or not dataset_path.is_dir():
        print("错误: 目标目录不存在或不是目录")
        return 1

    print_label_map_summary(dataset_path)

    hdf5_files = list_target_hdf5_files(dataset_path, args.split)
    if not hdf5_files:
        print("未找到匹配条件的 HDF5 文件")
        return 1

    total_samples = 0
    split_counter = {"Train": 0, "Val": 0, "Test": 0}

    for file_path in hdf5_files:
        stats = print_hdf5_summary(file_path, args.sample_limit)
        if stats is None:
            continue

        sample_count = int(stats.get("sample_count", 0))
        total_samples += sample_count
        for split_name in split_counter:
            if file_path.name.endswith(f"-{split_name}.hdf5"):
                split_counter[split_name] = sample_count

    print(f"\n{'=' * 70}")
    print("汇总统计")
    print(f"{'=' * 70}")
    print(f"数据集目录: {dataset_path.name}")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {split_counter['Train']}")
    print(f"验证集: {split_counter['Val']}")
    print(f"测试集: {split_counter['Test']}")

    if total_samples > 0:
        train_ratio = split_counter["Train"] / total_samples * 100
        val_ratio = split_counter["Val"] / total_samples * 100
        test_ratio = split_counter["Test"] / total_samples * 100
        print(
            f"数据划分比例: 训练 {train_ratio:.1f}% / 验证 {val_ratio:.1f}% / 测试 {test_ratio:.1f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
