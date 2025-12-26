"""
数据预处理模块
功能：
1. 遍历 dataset/video/{split}/{translator}/ 下的所有视频文件
2. 使用 MediaPipe 提取手部关键点特征
3. 将特征保存为 .npy 文件到 dataset/processed/{split}/{translator}/
4. 关联标签文件并生成处理后的标签映射
5. 支持多进程加速和断点续传
"""

import sys
import os

# 将 src 目录添加到 path 中以支持导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import numpy as np
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial
from tqdm import tqdm
import config as cfg

# Windows 多进程 (spawn) 模式下，子进程会重新导入模块
# 使用 try/except 处理不同的导入场景
try:
    from data_preprocessing import extract_keypoints
except ImportError:
    # 作为包内模块直接运行时的 fallback
    import extract_keypoints

# 修复 Windows 控制台中文乱码问题
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')  # type: ignore


def load_labels(split: str) -> dict:
    """
    加载指定划分的标签文件
    
    Args:
        split: 数据集划分 ('train', 'test', 'dev')
    
    Returns:
        dict: {video_id: {'translator': str, 'chinese': str, 'gloss': str, 'note': str}}
    """
    label_file = os.path.join(cfg.RAW_LABELS_PATH, f'{split}.csv')
    labels = {}
    
    if not os.path.exists(label_file):
        print(f"[WARN] 标签文件不存在: {label_file}")
        return labels
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过前两行标题
            next(reader)  # Column1, Column2, ...
            next(reader)  # Number, Translator, ...
            
            for row in reader:
                if len(row) >= 4:
                    video_id = row[0].strip()  # e.g., 'train-00001'
                    labels[video_id] = {
                        'translator': row[1].strip(),
                        'chinese': row[2].strip(),
                        'gloss': row[3].strip(),
                        'note': row[4].strip() if len(row) > 4 else ''
                    }
    except Exception as e:
        print(f"[ERROR] 加载标签文件失败: {label_file}, 错误: {e}")
    
    return labels


def get_video_id_from_filename(filename: str) -> str:
    """
    从文件名中提取视频ID
    例如: 'train-00001.mp4' -> 'train-00001'
    """
    return os.path.splitext(filename)[0]


def process_single_video(args: tuple) -> dict:
    """
    处理单个视频文件的包装函数 (供多进程调用)
    
    Args:
        args: (video_path, save_path, video_id, label_info)
    
    Returns:
        dict: 处理结果信息
    """
    video_path, save_path, video_id, label_info = args
    result = {
        'video_id': video_id,
        'status': 'Unknown',
        'video_path': video_path,
        'save_path': save_path,
        'error': None,
        'frame_count': 0
    }
    
    try:
        # 检查是否已经处理过 (断点续传)
        if cfg.ENABLE_RESUME and os.path.exists(save_path):
            result['status'] = 'Skipped'
            return result
        
        # 检查源视频是否存在
        if not os.path.exists(video_path):
            result['status'] = 'Error'
            result['error'] = f"视频文件不存在: {video_path}"
            return result
        
        # 核心特征提取
        npy_data = extract_keypoints.extract_features(video_path)
        
        if npy_data is None or len(npy_data) == 0:
            result['status'] = 'Error'
            result['error'] = "特征提取失败或视频为空"
            return result
        
        # 创建输出目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存特征为 .npy 文件
        np.save(save_path, npy_data)
        
        result['status'] = 'Success'
        result['frame_count'] = len(npy_data)
        
    except Exception as e:
        result['status'] = 'Error'
        result['error'] = str(e)
    
    return result


def collect_video_tasks(split: str, labels: dict) -> list:
    """
    收集指定划分下的所有视频处理任务
    
    Args:
        split: 数据集划分 ('train', 'test', 'dev')
        labels: 标签字典
    
    Returns:
        list: [(video_path, save_path, video_id, label_info), ...]
    """
    tasks = []
    split_video_path = os.path.join(cfg.RAW_VIDEOS_PATH, split)
    split_output_path = os.path.join(cfg.PROCESSED_DATA_PATH, split)
    
    if not os.path.exists(split_video_path):
        print(f"[WARN] 视频目录不存在: {split_video_path}")
        return tasks
    
    # 遍历所有翻译者目录
    for translator in cfg.TRANSLATORS:
        translator_path = os.path.join(split_video_path, translator)
        
        if not os.path.exists(translator_path):
            continue
        
        # 遍历该翻译者目录下的所有视频
        for filename in os.listdir(translator_path):
            # 检查是否为支持的视频格式
            ext = os.path.splitext(filename)[1].lower()
            if ext not in cfg.VIDEO_EXTENSIONS:
                continue
            
            video_path = os.path.join(translator_path, filename)
            video_id = get_video_id_from_filename(filename)
            
            # 构造输出路径
            output_filename = video_id + '.npy'
            save_path = os.path.join(split_output_path, translator, output_filename)
            
            # 获取标签信息
            label_info = labels.get(video_id, None)
            
            tasks.append((video_path, save_path, video_id, label_info))
    
    return tasks


def save_processed_labels(split: str, labels: dict, results: list):
    """
    保存处理后的标签映射文件
    
    Args:
        split: 数据集划分
        labels: 原始标签字典
        results: 处理结果列表
    """
    os.makedirs(cfg.PROCESSED_LABELS_PATH, exist_ok=True)
    
    # 筛选成功处理的视频
    successful_ids = {r['video_id'] for r in results if r['status'] == 'Success'}
    # 也包括已跳过的 (之前已处理的)
    skipped_ids = {r['video_id'] for r in results if r['status'] == 'Skipped'}
    valid_ids = successful_ids | skipped_ids
    
    # 构建处理后的标签列表
    processed_labels = []
    for video_id in sorted(valid_ids):
        label_info = labels.get(video_id, {})
        processed_labels.append({
            'video_id': video_id,
            'npy_path': os.path.join(cfg.PROCESSED_DATA_PATH, split, 
                                     label_info.get('translator', 'unknown'), 
                                     video_id + '.npy'),
            'translator': label_info.get('translator', ''),
            'chinese': label_info.get('chinese', ''),
            'gloss': label_info.get('gloss', ''),
            'note': label_info.get('note', '')
        })
    
    # 保存为 JSON 文件
    output_file = os.path.join(cfg.PROCESSED_LABELS_PATH, f'{split}_labels.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_labels, f, ensure_ascii=False, indent=2)
    
    print(f"  [INFO] 标签文件已保存: {output_file} ({len(processed_labels)} 条)")


def save_error_log(all_errors: list):
    """保存错误日志"""
    if not cfg.SAVE_ERROR_LOG or not all_errors:
        return
    
    with open(cfg.ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"预处理错误日志\n")
        f.write(f"{'='*50}\n\n")
        for err in all_errors:
            f.write(f"视频ID: {err['video_id']}\n")
            f.write(f"视频路径: {err['video_path']}\n")
            f.write(f"错误信息: {err['error']}\n")
            f.write(f"{'-'*30}\n")
    
    print(f"[INFO] 错误日志已保存: {cfg.ERROR_LOG_PATH}")


def process_split(split: str) -> dict:
    """
    处理单个数据集划分
    
    Args:
        split: 数据集划分 ('train', 'test', 'dev')
    
    Returns:
        dict: 处理统计信息
    """
    print(f"\n{'='*50}")
    print(f"[TASK] 处理 {split.upper()} 数据集...")
    print(f"{'='*50}")
    
    # 1. 加载标签
    labels = load_labels(split)
    print(f"  标签数量: {len(labels)}")
    
    # 2. 收集任务
    tasks = collect_video_tasks(split, labels)
    print(f"  视频数量: {len(tasks)}")
    
    # 应用数据量限制
    if cfg.DATA_LIMIT is not None and cfg.DATA_LIMIT != 'all':
        try:
            limit = int(cfg.DATA_LIMIT)
            if limit > 0:
                print(f"  [TEST] 启用测试模式: 仅处理前 {limit} 个视频")
                tasks = tasks[:limit]
        except ValueError:
            print(f"  [WARN] 配置错误: DATA_LIMIT 应为整数或 'all'，当前为 {cfg.DATA_LIMIT}")

    if not tasks:
        print(f"  [WARN] 未找到视频文件，跳过...")
        return {'split': split, 'total': 0, 'success': 0, 'skipped': 0, 'errors': 0}
    
    # 3. 多进程处理
    results = []
    
    if cfg.USE_MULTIPROCESSING and cfg.NUM_WORKERS > 1:
        print(f"  [MP] 使用 {cfg.NUM_WORKERS} 个进程并行处理...")
        
        with Pool(processes=cfg.NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(process_single_video, tasks, chunksize=cfg.CHUNK_SIZE),
                total=len(tasks),
                desc=f"  处理 {split}",
                unit="video"
            ))
    else:
        print(f"  [SP] 使用单进程处理...")
        for task in tqdm(tasks, desc=f"  处理 {split}", unit="video"):
            results.append(process_single_video(task))
    
    # 4. 统计结果
    success_count = sum(1 for r in results if r['status'] == 'Success')
    skip_count = sum(1 for r in results if r['status'] == 'Skipped')
    error_results = [r for r in results if r['status'] == 'Error']
    
    print(f"\n  [OK] 成功: {success_count}")
    print(f"  [SKIP] 跳过: {skip_count}")
    print(f"  [FAIL] 失败: {len(error_results)}")
    
    # 5. 保存处理后的标签
    save_processed_labels(split, labels, results)
    
    return {
        'split': split,
        'total': len(tasks),
        'success': success_count,
        'skipped': skip_count,
        'errors': len(error_results),
        'error_details': error_results
    }


def preprocess_all(splits: list | None = None):
    """
    预处理所有数据集划分
    
    Args:
        splits: 要处理的划分列表，默认为所有划分
    """
    if splits is None:
        splits = cfg.SPLITS
    
    print("="*60)
    print("CSL-LSTM 数据预处理")
    print("="*60)
    print(f"[PATH] 项目根目录: {cfg.PROJECT_ROOT}")
    print(f"[PATH] 视频目录: {cfg.RAW_VIDEOS_PATH}")
    print(f"[PATH] 标签目录: {cfg.RAW_LABELS_PATH}")
    print(f"[PATH] 输出目录: {cfg.PROCESSED_DATA_PATH}")
    print(f"[SYS] CPU核心: {cfg.NUM_WORKERS} / {cpu_count()}")
    print(f"[CFG] 断点续传: {'启用' if cfg.ENABLE_RESUME else '禁用'}")
    print(f"[CFG] 多进程: {'启用' if cfg.USE_MULTIPROCESSING else '禁用'}")
    
    all_stats = []
    all_errors = []
    
    for split in splits:
        stats = process_split(split)
        all_stats.append(stats)
        all_errors.extend(stats.get('error_details', []))
    
    # 汇总统计
    print("\n" + "="*60)
    print("处理汇总")
    print("="*60)
    
    total_all = sum(s['total'] for s in all_stats)
    success_all = sum(s['success'] for s in all_stats)
    skip_all = sum(s['skipped'] for s in all_stats)
    error_all = sum(s['errors'] for s in all_stats)
    
    for stats in all_stats:
        print(f"  {stats['split']:8s}: 总计 {stats['total']:5d}, "
              f"成功 {stats['success']:5d}, "
              f"跳过 {stats['skipped']:5d}, "
              f"失败 {stats['errors']:5d}")
    
    print(f"  {'合计':8s}: 总计 {total_all:5d}, "
          f"成功 {success_all:5d}, "
          f"跳过 {skip_all:5d}, "
          f"失败 {error_all:5d}")
    
    # 保存错误日志
    if all_errors:
        save_error_log(all_errors)
        print("\n[WARN] 部分视频处理失败，请查看错误日志。")
    
    print("\n[DONE] 数据预处理完成!")
    
    return all_stats


if __name__ == '__main__':
    # Windows 下多进程必须放在 if __name__ == '__main__' 下
    freeze_support()
    
    # 可以指定只处理某些划分
    # preprocess_all(splits=['train'])
    
    # 处理所有划分
    preprocess_all()