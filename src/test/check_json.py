import argparse
import json
import os
import sys
from pathlib import Path


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


def default_data_root() -> Path:
    return Path(cfg.PATHS.project_root) / "dataset" / "processed"


def build_label_map_path(dataset_scale: str, data_root: Path) -> Path:
    return data_root / f"WLASL{dataset_scale}" / f"wlasl_{dataset_scale}_maplabels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查标签映射 JSON 的键值结构")
    parser.add_argument(
        "--dataset-scale",
        default=str(cfg.PATHS.dataset_scale),
        choices=["100", "300", "1000", "2000"],
        help="数据集规模，默认读取 config.py 配置",
    )
    parser.add_argument(
        "--split",
        default="Train",
        choices=["Train", "Val", "Test"],
        help="保留统一参数接口，当前仅用于输出上下文",
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
    label_map_path = build_label_map_path(args.dataset_scale, args.data_root)

    print(f"解析后的数据规模: WLASL{args.dataset_scale}")
    print(f"解析后的划分参数: {args.split}")
    print(f"解析后的实际文件路径: {label_map_path.resolve()}")

    if not label_map_path.exists():
        print("错误: 标签映射文件不存在")
        return 1

    with open(label_map_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    print("JSON 顶层键:", list(data.keys()))

    if "label_to_id" in data:
        label_to_id = data["label_to_id"]
        print("label_to_id 示例:", list(label_to_id.items())[:5])

    if "id_to_label" in data:
        id_to_label = data["id_to_label"]
        print("id_to_label 示例:", list(id_to_label.items())[:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
