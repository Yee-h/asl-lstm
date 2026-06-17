"""
WLASL视频重命名脚本

功能：
1. 遍历指定规模的数据集和对应的JSON词汇表(用户可指定)，比如：WLASL100
2. 对指定数据集下的所有视频重命名(train、val、test均要重命名)为原视频名称_对应的词汇
   比如：00623_book.mp4

使用方法：
    直接修改下方的配置参数，然后运行：
    uv run python rename_videos.py
"""

import os
import json
import shutil
from pathlib import Path


# ==================== 配置参数（在此修改） ====================

# 数据集规模: 100, 300, 1000或 2000
DATASET_SCALE = 2000

# 运行模式:
#   - False: 实际执行重命名
#   - True:  预览模式，只显示将要进行的重命名操作，不实际执行
DRY_RUN = False

# 撤销模式:
#   - False: 正常重命名 (69241.mp4 -> 69241_book.mp4)
#   - True:  撤销重命名，恢复原始文件名 (69241_book.mp4 -> 69241.mp4)
UNDO_MODE = False

# 数据根目录（硬编码）
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset" / "wlasl-complete"
SPLIT_VIDEO_ROOT = DATASET_ROOT / "split_videos"
CLASS_LIST_PATH = DATASET_ROOT / "wlasl_class_list.txt"
NSLT_JSON_TEMPLATE = DATASET_ROOT / "nslt_{scale}.json"
OUTPUT_ROOT = DATASET_ROOT / "videoId_word"

# ============================================================


def load_class_list(class_list_path: Path) -> dict:
    """
    加载词汇表映射文件
    
    Args:
        class_list_path: wlasl_class_list.txt的路径
        
    Returns:
        dict: {类别索引: 词汇名称} 的映射字典
    """
    class_map = {}
    with open(class_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    word = parts[1]
                    class_map[idx] = word
    return class_map


def load_nslt_json(json_path: Path) -> dict:
    """
    加载nslt JSON文件
    
    Args:
        json_path: nslt_xxx.json的路径
        
    Returns:
        dict: {视频ID: {"subset": "train/val/test", "action": [类别索引, ...]}} 的映射
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_video_label_map(nslt_data: dict, class_map: dict) -> dict:
    """
    构建视频ID到词汇的映射
    
    Args:
        nslt_data: nslt JSON数据
        class_map: 类别索引到词汇的映射
        
    Returns:
        dict: {视频ID: 词汇名称} 的映射
    """
    video_label_map = {}
    for video_id, info in nslt_data.items():
        action = info.get('action', [])
        if action:
            class_idx = action[0]  # action数组的第一个元素是类别索引
            if class_idx in class_map:
                video_label_map[video_id] = class_map[class_idx]
    return video_label_map


def rename_videos(dataset_path: Path, video_label_map: dict, dry_run: bool = False) -> tuple:
    """
    将重命名后的文件复制到 OUTPUT_ROOT/WLASL{scale}/子集 下，不在原地修改。
    """
    subsets = ['train', 'val', 'test']
    success_count = 0
    skip_count = 0
    fail_count = 0
    details = []
    
    for subset in subsets:
        subset_path = dataset_path / subset
        if not subset_path.exists():
            print(f"警告: 子集目录不存在: {subset_path}")
            continue
            
        print(f"\n处理 {subset} 子集...")
        
        for filename in os.listdir(subset_path):
            if not filename.endswith('.mp4'):
                continue
                
            # 提取视频ID (不带扩展名)
            video_id = filename.replace('.mp4', '')
            # 如果已含下划线，取下划线前作为 ID，重新生成标准命名
            if '_' in video_id:
                video_id = video_id.split('_')[0]
            
            # 查找对应的词汇
            if video_id not in video_label_map:
                fail_count += 1
                details.append({
                    'subset': subset,
                    'original': filename,
                    'new': None,
                    'status': 'failed',
                    'reason': f'在词汇表中找不到视频ID: {video_id}'
                })
                continue
                
            word = video_label_map[video_id]
            # 清理词汇名称，移除不适合作为文件名的字符
            safe_word = word.replace(' ', '_').replace('/', '_').replace('\\', '_')
            new_filename = f"{video_id}_{safe_word}.mp4"

            old_path = subset_path / filename
            new_path = OUTPUT_ROOT / f"WLASL{DATASET_SCALE}" / subset / new_filename

            if dry_run:
                print(f"  [预览] {old_path} -> {new_path}")
                success_count += 1
                details.append({
                    'subset': subset,
                    'original': str(old_path),
                    'new': str(new_path),
                    'status': 'preview',
                    'reason': None
                })
            else:
                try:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    if new_path.exists():
                        new_path.unlink()
                    shutil.copy2(old_path, new_path)
                    print(f"  [成功] {old_path.name} -> {new_path}")
                    success_count += 1
                    details.append({
                        'subset': subset,
                        'original': str(old_path),
                        'new': str(new_path),
                        'status': 'success',
                        'reason': None
                    })
                except Exception as e:
                    print(f"  [失败] {filename}: {e}")
                    fail_count += 1
                    details.append({
                        'subset': subset,
                        'original': str(old_path),
                        'new': str(new_path),
                        'status': 'failed',
                        'reason': str(e)
                    })
    
    return success_count, skip_count, fail_count, details


def undo_rename(dataset_path: Path, dry_run: bool = False) -> tuple:
    """撤销：删除 OUTPUT_ROOT/子集 下已复制的重命名文件（不改动源文件）。"""
    subsets = ['train', 'val', 'test']
    success_count = 0
    skip_count = 0
    fail_count = 0
    details = []

    for subset in subsets:
        target_subset_dir = OUTPUT_ROOT / f"WLASL{DATASET_SCALE}" / subset
        if not target_subset_dir.exists():
            print(f"警告: 目标子集目录不存在: {target_subset_dir}")
            continue

        print(f"\n处理 {subset} 子集 (删除已复制文件)...")

        for filename in os.listdir(target_subset_dir):
            if not filename.endswith('.mp4'):
                continue

            file_path = target_subset_dir / filename

            if dry_run:
                print(f"  [预览删除] {file_path}")
                success_count += 1
                details.append({
                    'subset': subset,
                    'original': str(file_path),
                    'new': None,
                    'status': 'preview-delete',
                    'reason': None
                })
            else:
                try:
                    file_path.unlink(missing_ok=False)
                    print(f"  [删除] {file_path}")
                    success_count += 1
                    details.append({
                        'subset': subset,
                        'original': str(file_path),
                        'new': None,
                        'status': 'deleted',
                        'reason': None
                    })
                except Exception as e:
                    print(f"  [失败] 删除 {file_path}: {e}")
                    fail_count += 1
                    details.append({
                        'subset': subset,
                        'original': str(file_path),
                        'new': None,
                        'status': 'failed',
                        'reason': str(e)
                    })

    return success_count, skip_count, fail_count, details


def main():
    # 使用硬编码配置
    scale = DATASET_SCALE
    dry_run = DRY_RUN
    undo = UNDO_MODE

    dataset_name = f"WLASL{scale}"
    dataset_path = SPLIT_VIDEO_ROOT / dataset_name
    class_list_path = CLASS_LIST_PATH
    nslt_json_path = Path(str(NSLT_JSON_TEMPLATE).format(scale=scale))

    # 验证路径
    if not dataset_path.exists():
        print(f"错误: 数据集目录不存在: {dataset_path}")
        return

    if not class_list_path.exists():
        print(f"错误: 词汇表文件不存在: {class_list_path}")
        return

    if not nslt_json_path.exists():
        print(f"错误: nslt JSON文件不存在: {nslt_json_path}")
        return

    (OUTPUT_ROOT / f"WLASL{scale}").mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"WLASL 视频重命名工具")
    print("=" * 60)
    print(f"数据集规模: {dataset_name}")
    print(f"数据集路径: {dataset_path}")
    print(f"词汇表路径: {class_list_path}")
    print(f"nslt JSON路径: {nslt_json_path}")
    print(f"模式: {'撤销重命名' if undo else ('预览' if dry_run else '执行重命名')}")
    print("=" * 60)
    
    if undo:
        # 撤销重命名
        success, skip, fail, details = undo_rename(dataset_path, dry_run)
    else:
        # 加载映射数据
        print("\n加载词汇表...")
        class_map = load_class_list(class_list_path)
        print(f"  加载了 {len(class_map)} 个词汇")
        
        print("\n加载nslt JSON...")
        nslt_data = load_nslt_json(nslt_json_path)
        print(f"  加载了 {len(nslt_data)} 条视频记录")
        
        print("\n构建视频-词汇映射...")
        video_label_map = get_video_label_map(nslt_data, class_map)
        print(f"  构建了 {len(video_label_map)} 条映射")
        
        # 执行重命名
        success, skip, fail, details = rename_videos(dataset_path, video_label_map, dry_run)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    print(f"  {'预览' if dry_run else '成功'}: {success}")
    print(f"  跳过: {skip}")
    print(f"  失败: {fail}")
    print(f"  总计: {success + skip + fail}")
    
    if fail > 0:
        print("\n失败的文件:")
        for d in details:
            if d['status'] == 'failed':
                print(f"  - [{d['subset']}] {d['original']}: {d['reason']}")
    
    print("\n完成!")


if __name__ == '__main__':
    main()

