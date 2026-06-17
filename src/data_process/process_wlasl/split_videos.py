"""根据视频列表文件，将原始视频复制/移动到分好子集的输出目录。"""

from __future__ import annotations

import shutil
from pathlib import Path

# ============================================================
# 配置区域 - 修改这里选择不同规模的数据集
# 可选值: "100", "300", "1000", "2000"
# ============================================================
DATASET_SCALE = "2000"
# ============================================================

# 是否复制文件（True）还是移动文件（False）
# 建议先使用 True 测试，确认无误后改为 False 移动文件
COPY_MODE = True


def load_video_names(video_list_file: Path) -> list[str]:
    """从文本文件中加载视频名称列表"""
    with open(video_list_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_videos(dataset_root: Path, video_names: list[str]) -> dict:
    """按照列表将视频复制/移动到 dataset_root/split_videos 下的对应子目录。"""

    video_dir = dataset_root / "videos"
    output_dir = dataset_root / "split_videos"

    stats = {"success": 0, "not_found": 0, "error": 0, "not_found_list": []}

    for video_path in video_names:
        # video_path 形如: WLASL100/train/58363.mp4
        rel_path = Path(video_path)
        video_filename = rel_path.name

        src_file = video_dir / video_filename
        dst_file = output_dir / rel_path

        if not src_file.exists():
            stats["not_found"] += 1
            stats["not_found_list"].append(video_filename)
            continue

        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            if COPY_MODE:
                shutil.copy2(src_file, dst_file)
            else:
                shutil.move(src_file, dst_file)

            stats["success"] += 1
        except Exception as e:  # pragma: no cover - 仅用于日志
            stats["error"] += 1
            print(f"处理文件 {video_filename} 时出错: {e}")

    return stats


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "dataset" / "wlasl-complete"

    video_list_file = dataset_root / "lists"  / f"wlasl{DATASET_SCALE}" / f"wlasl{DATASET_SCALE}_videos.txt"

    if not video_list_file.exists():
        print(f"错误: 视频列表文件不存在: {video_list_file}")
        print("请先运行 extract_video_names.py 生成列表")
        return

    if not (dataset_root / "videos").exists():
        print(f"错误: 原始视频目录不存在: {dataset_root / 'videos'}")
        return

    print(f"数据集规模: WLASL{DATASET_SCALE}")
    print(f"模式: {'复制' if COPY_MODE else '移动'}")
    print(f"读取视频列表: {video_list_file}")

    video_names = load_video_names(video_list_file)
    print(f"共有 {len(video_names)} 个视频需要处理")

    print("\n开始处理视频...")
    stats = split_videos(dataset_root, video_names)

    print("\n========== 处理完成 ==========")
    print(f"成功: {stats['success']} 个")
    print(f"未找到: {stats['not_found']} 个")
    print(f"错误: {stats['error']} 个")

    if stats["not_found_list"]:
        log_file = dataset_root / "lists" / f"wlasl{DATASET_SCALE}_missing_videos.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            for name in stats["not_found_list"]:
                f.write(name + "\n")
        print(f"\n未找到的视频列表已保存到: {log_file}")
        print(f"前10个未找到的视频: {stats['not_found_list'][:10]}")


if __name__ == "__main__":
    main()
