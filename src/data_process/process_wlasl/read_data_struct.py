from pathlib import Path
import argparse
import sys
import os
import h5py
import numpy as np


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


def _build_file_path(dataset: str, split: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "pose_action_dataset"
    subset_dir = data_root / f"WLASL{dataset}"
    filename = f"WLASL{dataset}_135-{split}.hdf5"
    return subset_dir / filename


def inspect_hdf5(file_path: Path) -> None:
    try:
        with h5py.File(file_path, "r") as f:
            print("---- 文件内的顶层键 (Keys) ----")
            print(list(f.keys()))

            print("\n---- 数据详细信息 ----")
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    print(f"变量名: {key}")
                    print(f"  - 形状 (Shape): {data.shape}")
                    print(f"  - 类型 (Dtype): {data.dtype}")
                else:
                    print(f"组 (Group): {key}")

            # Inspect the first group to see deep structure
            first_key = list(f.keys())[0]
            print(f"\n---- 深入检查第一个组 ('{first_key}') ----")
            first_group = f[first_key]
            if isinstance(first_group, h5py.Group):
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
                                print(f"{indent}  Value: {val.decode('utf-8', errors='ignore')}")
                            else:
                                print(f"{indent}  Value: {val}")
                        elif name.endswith("data"):
                             print(f"{indent}  First Frame Data (partial): {obj[0, 0, :5]}")

                    elif isinstance(obj, h5py.Group):
                        print(f"{indent}Group: {name.split('/')[-1]}")

                first_group.visititems(print_group_structure)

            print("\n---- 验证不同样本的形状 (前5个) ----")
            keys = list(f.keys())
            for i in range(min(5, len(keys))):
                key = keys[i]
                if "data" in f[key]:
                    d = f[key]["data"]
                    print(f"Sample ID: {key}, Data Shape: {d.shape}")



    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
    except OSError as err:
        print(f"无法打开 HDF5 文件: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 WLASL 数据集的 HDF5 结构")
    parser.add_argument(
        "--dataset",
        choices=["100", "300", "2000"],
        default="100",
        help="选择数据子集：100 / 300 / 2000"
    )
    parser.add_argument(
        "--split",
        choices=["Train", "Val", "Test"],
        default="Train",
        help="选择数据集划分"
    )
    return parser.parse_args()


def main() -> None:
    _enable_utf8_console_output()

    args = parse_args()
    file_path = _build_file_path(args.dataset, args.split)

    print(f"读取文件: {file_path}")
    inspect_hdf5(file_path)


if __name__ == "__main__":
    main()