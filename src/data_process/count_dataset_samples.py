import argparse
import os
import sys
from pathlib import Path

import h5py


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


def default_data_root() -> Path:
    return Path(cfg.PATHS.project_root) / "dataset" / "processed"


def build_dataset_dir(dataset_scale: str, data_root: Path) -> Path:
    return data_root / f"WLASL{dataset_scale}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计数据集目录下 HDF5 样本数量")
    parser.add_argument(
        "--dataset-scale",
        default=str(cfg.PATHS.dataset_scale),
        choices=["100", "300", "1000", "2000"],
        help="数据集规模，默认读取 config.py 中配置",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "Train", "Val", "Test"],
        help="仅统计指定划分，默认统计全部",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="已处理数据根目录，默认 <project>/dataset/processed",
    )
    return parser.parse_args()


def count_samples(dataset_path: Path, split: str = "all") -> int:
    if not dataset_path.exists():
        print(f"错误: 目录不存在 - {dataset_path}")
        return 1

    print("=" * 60)
    print("数据集样本统计")
    print(f"数据目录: {dataset_path.resolve()}")
    print(f"划分筛选: {split}")
    print("=" * 60)

    hdf5_files = sorted(dataset_path.glob("*.hdf5"))
    if split != "all":
        suffix = f"-{split}.hdf5"
        hdf5_files = [path for path in hdf5_files if path.name.endswith(suffix)]

    if not hdf5_files:
        print("该目录下未找到匹配条件的 .hdf5 文件。")
        return 1

    total_all = 0
    print(f"\n{'文件名':<35} | {'样本数量':>10}")
    print(f"{'-' * 35}-|-{'-' * 10}")

    for hdf5_file in hdf5_files:
        try:
            with h5py.File(hdf5_file, "r") as h5_obj:
                count = len(h5_obj.keys())
                print(f"{hdf5_file.name:<35} | {count:>10}")
                total_all += count
        except Exception as exc:
            print(f"{hdf5_file.name:<35} | 错误: {exc}")

    print(f"{'-' * 35}-|-{'-' * 10}")
    print(f"{'总计':<35} | {total_all:>10}")
    print("=" * 60)
    return 0


def main() -> int:
    args = parse_args()
    dataset_dir = build_dataset_dir(args.dataset_scale, args.data_root)
    print(f"解析后的实际目录: {dataset_dir.resolve()}")
    return count_samples(dataset_dir, split=args.split)


if __name__ == "__main__":
    raise SystemExit(main())
