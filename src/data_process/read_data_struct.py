from pathlib import Path
import argparse
import sys
import os
import h5py
import numpy as np


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


def _enable_utf8_console_output() -> None:
    """确保 Windows 终端能够正确显示 UTF-8 字符。"""
    if os.name != "nt":
        return

    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        # 如果无法修改代码页，我们依旧尝试调整 stdout 编码
        pass

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _default_data_root() -> Path:
    return Path(cfg.PATHS.project_root) / "dataset" / "processed"


def _build_file_path(dataset: str, split: str, data_root: Path | None = None) -> Path:
    if data_root is None:
        data_root = _default_data_root()
    subset_dir = data_root / f"WLASL{dataset}"
    filename = f"WLASL{dataset}_135-{split}.hdf5"
    return subset_dir / filename


def inspect_hdf5(file_path: Path) -> None:
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            print("---- 文件内的顶层键 (Keys) ----")
            print(f"样本数量: {len(keys)}")
            print(f"前10个键: {keys[:10]}")

            if not keys:
                print("\n文件为空，没有可用样本。")
                return

            print("\n---- 数据详细信息 ----")
            preview_count = min(20, len(keys))
            group_count = 0
            dataset_count = 0
            for key in keys:
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    dataset_count += 1
                elif isinstance(data, h5py.Group):
                    group_count += 1

            print(f"顶层 Group 数: {group_count}")
            print(f"顶层 Dataset 数: {dataset_count}")
            print(f"键预览 (前{preview_count}个):")
            for key in keys[:preview_count]:
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    print(f"  Dataset: {key}, shape={data.shape}, dtype={data.dtype}")
                elif isinstance(data, h5py.Group):
                    print(f"  Group: {key}")

            # Inspect the first group to see deep structure
            first_key = keys[0]
            print(f"\n---- 深入检查第一个组 ('{first_key}') ----")
            first_group = f[first_key]
            if isinstance(first_group, h5py.Group):
                required_fields = [
                    "data",
                    "length",
                    "label",
                    "video_name",
                    "width",
                    "height",
                    "mask",
                    "quality",
                ]
                missing = [
                    field for field in required_fields if field not in first_group
                ]
                if missing:
                    print(f"缺少字段: {missing}")
                else:
                    print(
                        "字段检查: data/length/label/video_name/width/height/mask/quality 全部存在"
                    )

                def print_group_structure(name, obj):
                    indent = "  " * (name.count("/") + 1)
                    if isinstance(obj, h5py.Dataset):
                        print(f"{indent}Dataset: {name.split('/')[-1]}")
                        print(f"{indent}  Shape: {obj.shape}")
                        print(f"{indent}  Dtype: {obj.dtype}")

                        # Print values for scalars or small strings
                        if obj.shape == ():
                            val = obj[()]
                            if isinstance(val, bytes):
                                print(
                                    f"{indent}  Value: {val.decode('utf-8', errors='ignore')}"
                                )
                            else:
                                print(f"{indent}  Value: {val}")
                        elif name.endswith("data"):
                            print(
                                f"{indent}  First Frame Data (partial): {obj[0, 0, :5]}"
                            )

                    elif isinstance(obj, h5py.Group):
                        print(f"{indent}Group: {name.split('/')[-1]}")

                first_group.visititems(print_group_structure)

            print("\n---- 验证不同样本的形状 (前5个) ----")
            quality_values = []
            for i in range(min(5, len(keys))):
                key = keys[i]
                item = f[key]
                if not isinstance(item, h5py.Group):
                    continue

                if "data" in item:
                    d = item["data"]
                    if isinstance(d, h5py.Dataset):
                        print(f"Sample ID: {key}, Data Shape: {d.shape}")

                if "quality" in item:
                    quality_ds = item["quality"]
                    if isinstance(quality_ds, h5py.Dataset):
                        quality_values.append(float(np.asarray(quality_ds)[()]))

            if quality_values:
                print(
                    "\n前5个样本 quality 统计: "
                    f"min={np.min(quality_values):.4f}, "
                    f"mean={np.mean(quality_values):.4f}, "
                    f"max={np.max(quality_values):.4f}"
                )

    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
    except OSError as err:
        print(f"无法打开 HDF5 文件: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 WLASL 数据集的 HDF5 结构")
    parser.add_argument(
        "--dataset",
        choices=["100", "300", "1000", "2000"],
        default="100",
        help="选择数据子集：100 / 300 / 2000",
    )
    parser.add_argument(
        "--split",
        choices=["Train", "Val", "Test"],
        default="Train",
        help="选择数据集划分",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_default_data_root(),
        help="已处理数据根目录（默认: <project>/dataset/processed）",
    )
    return parser.parse_args()


def main() -> None:
    _enable_utf8_console_output()

    args = parse_args()
    file_path = _build_file_path(args.dataset, args.split, data_root=args.data_root)

    print(f"读取文件: {file_path}")
    inspect_hdf5(file_path)


if __name__ == "__main__":
    main()
