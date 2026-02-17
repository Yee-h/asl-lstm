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


def build_hdf5_path(dataset_scale: str, split: str, data_root: Path) -> Path:
    file_name = f"WLASL{dataset_scale}_135-{split}.hdf5"
    return data_root / f"WLASL{dataset_scale}" / file_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 HDF5 坐标样本结构")
    parser.add_argument(
        "--dataset-scale",
        default=str(cfg.PATHS.dataset_scale),
        choices=["100", "300", "1000", "2000"],
        help="数据集规模，默认读取 config.py 配置",
    )
    parser.add_argument(
        "--split",
        default="Test",
        choices=["Train", "Val", "Test"],
        help="数据划分，默认 Test",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="已处理数据根目录，默认 <project>/dataset/processed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_path = build_hdf5_path(args.dataset_scale, args.split, args.data_root)

    print(f"解析后的数据规模: WLASL{args.dataset_scale}")
    print(f"解析后的划分参数: {args.split}")
    print(f"解析后的实际文件路径: {file_path.resolve()}")

    if not file_path.exists():
        print("错误: HDF5 文件不存在")
        return 1

    try:
        with h5py.File(file_path, "r") as h5_obj:
            if not h5_obj.keys():
                print("错误: 文件中没有样本数据")
                return 1

            sample_id = list(h5_obj.keys())[0]
            data = h5_obj[sample_id]["data"][:]

            print(f"\n{'=' * 50}")
            print(f"检查样本 ID: {sample_id}")
            print(f"数据总形状: {data.shape}")
            print(f"  - 帧数: {data.shape[0]}")
            print(f"  - 坐标维度: {data.shape[1]}")
            print(f"  - 关键点数量: {data.shape[2]}")
            print(f"{'=' * 50}\n")

            frame_idx = 0
            frame_data = data[frame_idx]
            print(f"第 {frame_idx + 1} 帧前 5 个关键点:")
            print("-" * 40)
            for idx in range(5):
                x_val = frame_data[0, idx]
                y_val = frame_data[1, idx]
                print(f"关键点 {idx:>2}: x = {x_val:.6f}, y = {y_val:.6f}")
            print("-" * 40)

    except Exception as exc:
        print(f"程序运行发生错误: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
