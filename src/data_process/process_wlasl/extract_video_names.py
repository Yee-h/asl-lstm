"""从 WLASL 元数据 JSON 提取视频名称及其划分 (train/val/test)。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# ============================================================
# 配置区域 - 修改这里选择不同规模的数据集
# 可选值: "100", "300", "1000", "2000"
# ============================================================
DATASET_SCALE = "2000"
# ============================================================

DATASET_JSON_MAP: dict[str, str] = {
    "100": "nslt_100.json",
    "300": "nslt_300.json",
    "1000": "nslt_1000.json",
    "2000": "nslt_2000.json",
}


def _normalize_subset(name: str) -> str:
    mapping = {
        "validation": "val",
        "dev": "val",
    }
    return mapping.get(name.lower(), name.lower())


def load_video_entries(json_path: Path) -> list[tuple[str, str]]:
    """返回 (video_id, subset) 列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: list[tuple[str, str]] = []

    # nslt_*.json: dict -> {video_id: {subset: str, ...}}
    if isinstance(data, dict):
        for video_id, info in data.items():
            subset = _normalize_subset(info.get("subset", "train"))
            entries.append((video_id, subset))
        return entries

    # WLASL_v0.3.json: list -> [{gloss, instances: [...] }]
    if isinstance(data, list):
        for entry in data:
            for instance in entry.get("instances", []):
                video_id = instance.get("video_id")
                subset = _normalize_subset(instance.get("split", "train"))
                if video_id:
                    entries.append((video_id, subset))
        return entries

    raise ValueError(f"不支持的 JSON 结构: {json_path}")


def write_lists(output_dir: Path, dataset_label: str, entries: list[tuple[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    master_file = output_dir / f"{dataset_label.lower()}_videos.txt"

    lines = [f"{dataset_label}/{subset}/{vid}.mp4" for vid, subset in entries]
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 同时输出 train/val/test 单独列表，便于调试
    subsets = {subset for _, subset in entries}
    for subset in sorted(subsets):
        subset_file = output_dir / f"{dataset_label.lower()}_{subset}.txt"
        subset_lines = [line for line in lines if f"/{subset}/" in line]
        with open(subset_file, "w", encoding="utf-8") as f:
            f.write("\n".join(subset_lines))

    counter = Counter(subset for _, subset in entries)
    print("统计: " + ", ".join(f"{k}={v}" for k, v in counter.items()))
    print(f"已写入: {master_file}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "dataset" / "wlasl-complete"

    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    json_name = DATASET_JSON_MAP.get(DATASET_SCALE)
    if not json_name:
        raise ValueError(f"不支持的数据集规模: {DATASET_SCALE}")

    json_path = dataset_root / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件: {json_path}")

    print(f"使用 JSON: {json_path}")
    entries = load_video_entries(json_path)
    print(f"共提取 {len(entries)} 条记录")

    dataset_label = f"WLASL{DATASET_SCALE}"
    output_dir = dataset_root / "lists" / f"wlasl{DATASET_SCALE}"
    write_lists(output_dir, dataset_label, entries)


if __name__ == "__main__":
    main()
