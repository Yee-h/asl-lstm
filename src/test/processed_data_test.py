"""
数据结构检测工具
用于检测 dataset/processed 下的 HDF5 数据集结构信息

使用方法:
    方法1 (推荐): 直接修改下方 CONFIG 中的 DATASET_PATH，然后运行:
        python src/test/processed_data_test.py
    
    方法2: 通过命令行参数指定路径:
        python src/test/processed_data_test.py dataset/processed/WLASL100
        python src/test/processed_data_test.py d:/Document/Code/csl-lstm/dataset/processed/WLASL100
"""

import os
import sys
import argparse
import h5py
import json
import numpy as np


# ============================================================================
# 用户配置区域 - 在此处修改数据集路径
# ============================================================================
CONFIG = {
    # 待检测的数据集路径 - 修改此处以切换不同规模的数据集进行分析
    # 支持绝对路径或相对于项目根目录的相对路径
    "DATASET_PATH": "d:/Document/Code/csl-lstm/dataset/processed/WLASL100",
    
    # 采样限制 - 指定每个 HDF5 文件检查的样本数量 (用于快速预览结构)
    # 设置为 None 表示检查全部样本 (进行完整的统计分析)
    "SAMPLE_LIMIT": 10,
}
# ============================================================================


def print_hdf5_structure(hdf5_obj, prefix="", max_items=5, show_all=False):
    """
    递归提取并打印 HDF5 组或文件的树状结构信息。
    
    Args:
        hdf5_obj: HDF5 对象，可以是 File (文件), Group (组) 或 Dataset (数据集)。
        prefix: 用于缩进的前缀字符串，维持树状展示的美观。
        max_items: 当样本过多时，每级目录最多显示的项目数。
        show_all: 布尔值，如果为 True，则忽略 max_items 限制，全文输出。
    """
    if isinstance(hdf5_obj, h5py.File):
        print(f"{prefix}HDF5 File: {os.path.basename(hdf5_obj.filename)}")
        print(f"{prefix}├── 文件属性:")
        for attr_name, attr_value in hdf5_obj.attrs.items():
            print(f"{prefix}│   └── {attr_name}: {attr_value}")
        
        keys = list(hdf5_obj.keys())
        total_keys = len(keys)
        display_keys = keys if show_all else keys[:max_items]
        
        print(f"{prefix}├── 顶级组 (共 {total_keys} 个样本):")
        for i, key in enumerate(display_keys):
            is_last = (i == len(display_keys) - 1) and (len(display_keys) == total_keys)
            branch = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
            print(f"{prefix}│   {branch}[{key}]")
            print_hdf5_structure(hdf5_obj[key], prefix + "│   " + child_prefix, max_items, show_all)
        
        if not show_all and total_keys > max_items:
            print(f"{prefix}│   └── ... 还有 {total_keys - max_items} 个样本未显示")
    
    elif isinstance(hdf5_obj, h5py.Group):
        for i, key in enumerate(sorted(hdf5_obj.keys())):
            item = hdf5_obj[key]
            is_last = (i == len(hdf5_obj.keys()) - 1)
            branch = "└── " if is_last else "├── "
            
            if isinstance(item, h5py.Dataset):
                # 数据集
                shape_str = str(item.shape) if item.shape != () else "scalar"
                dtype_str = str(item.dtype)
                
                # 尝试显示值 (对于标量或小型数据)
                value_str = ""
                try:
                    if item.shape == ():
                        value = item[()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        value_str = f" = {value}"
                    elif item.shape[0] <= 3 and len(item.shape) == 1:
                        value_str = f" = {item[:]}"
                except:
                    pass
                
                print(f"{prefix}{branch}{key}: Dataset[{shape_str}] dtype={dtype_str}{value_str}")
            
            elif isinstance(item, h5py.Group):
                # 子组
                print(f"{prefix}{branch}{key}/ (Group)")
                child_prefix = "    " if is_last else "│   "
                print_hdf5_structure(item, prefix + child_prefix, max_items, show_all)
    
    elif isinstance(hdf5_obj, h5py.Dataset):
        shape_str = str(hdf5_obj.shape)
        dtype_str = str(hdf5_obj.dtype)
        print(f"{prefix}Dataset: shape={shape_str}, dtype={dtype_str}")


def inspect_hdf5_complete(file_path, sample_limit=None):
    """
    完整检查单个 HDF5 文件的结构和数据形状
    
    Args:
        file_path: HDF5 文件路径
        sample_limit: 采样数量限制，None 表示检查全部
    """
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"路径: {file_path}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            video_ids = list(f.keys())
            num_samples = len(video_ids)
            print(f"\n[基本信息]")
            print(f"  样本总数: {num_samples}")
            print(f"  文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
            if num_samples == 0:
                print("  警告: 文件为空!")
                return None
            
            # 输出完整的 HDF5 层级结构
            print(f"\n[完整 HDF5 层级结构] (显示前5个样本)")
            print("-" * 50)
            print_hdf5_structure(f, prefix="  ", max_items=5, show_all=False)
            print("-" * 50)

            # 取第一个样本查看结构
            sample_id = video_ids[0]
            group = f[sample_id]
            
            print(f"\n[数据结构] (示例 ID: {sample_id})")
            for key in sorted(group.keys()):
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # 对于标量数据，尝试显示值
                    if item.shape == ():
                        try:
                            value = item[()]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            print(f"  - {key}: dtype={item.dtype}, value={value}")
                        except:
                            print(f"  - {key}: dtype={item.dtype}, shape={item.shape}")
                    else:
                        print(f"  - {key}: dtype={item.dtype}, shape={item.shape}")
                else:
                    print(f"  - {key}: {type(item).__name__}")
            
            # 统计所有样本
            check_limit = sample_limit if sample_limit else num_samples
            check_limit = min(check_limit, num_samples)
            
            frame_counts = []
            features_dims = set()
            labels = {}  # label -> count
            video_names = []
            
            print(f"\n[正在分析 {check_limit} 个样本...]")
            
            for i in range(check_limit):
                vid = video_ids[i]
                grp = f[vid]
                
                # 帧数和特征维度
                if 'data' in grp:
                    data = grp['data']
                    frame_counts.append(data.shape[0])
                    features_dims.add(data.shape[1:])
                
                # 标签统计 (可能是整数或字符串gloss)
                if 'label' in grp:
                    label_raw = grp['label'][()]
                    if isinstance(label_raw, bytes):
                        label = label_raw.decode('utf-8')
                    elif isinstance(label_raw, np.integer) or isinstance(label_raw, int):
                        label = int(label_raw)
                    else:
                        label = str(label_raw)
                    labels[label] = labels.get(label, 0) + 1
                
                # 视频名称
                if 'video_name' in grp:
                    try:
                        name = grp['video_name'][()]
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        video_names.append(name)
                    except:
                        pass
            
            # 输出统计结果
            print(f"\n[数据统计] (基于 {check_limit}/{num_samples} 个样本)")
            
            if frame_counts:
                print(f"\n  帧数分布:")
                print(f"    - 平均帧数: {np.mean(frame_counts):.2f}")
                print(f"    - 最小帧数: {np.min(frame_counts)}")
                print(f"    - 最大帧数: {np.max(frame_counts)}")
                print(f"    - 标准差: {np.std(frame_counts):.2f}")
                
                # 帧数分布直方图 - 动态生成bins确保单调递增
                max_frames = max(frame_counts)
                # 预定义边界，然后过滤掉大于最大帧数的边界
                predefined_bins = [0, 30, 60, 90, 120, 150, 200, 300, 500]
                bins = [b for b in predefined_bins if b <= max_frames]
                bins.append(max_frames + 1)  # 添加最后一个边界
                # 去重并排序确保单调递增
                bins = sorted(set(bins))
                
                try:
                    hist, _ = np.histogram(frame_counts, bins=bins)
                    print(f"    - 帧数区间分布:")
                    for i in range(len(hist)):
                        if hist[i] > 0:
                            print(f"        [{bins[i]}-{bins[i+1]}): {hist[i]} 个样本")
                except Exception as e:
                    print(f"    - 帧数区间分布: 计算失败 ({e})")
            
            if features_dims:
                print(f"\n  特征维度:")
                for dim in features_dims:
                    print(f"    - shape: {dim}")
                    if len(dim) == 2 and dim[0] == 2:
                        print(f"      解释: (坐标轴=2 [x,y], 关键点数={dim[1]})")
                    elif len(dim) == 2 and dim[1] == 2:
                        print(f"      解释: (关键点数={dim[0]}, 坐标轴=2 [x,y])")
            
            if labels:
                print(f"\n  标签分布:")
                # 判断标签类型
                sample_label = list(labels.keys())[0]
                if isinstance(sample_label, str):
                    print(f"    - 标签类型: 字符串 (gloss名称)")
                    print(f"    - 不同标签数 (类别数): {len(labels)}")
                    print(f"    - 每类平均样本数: {np.mean(list(labels.values())):.2f}")
                    print(f"    - 每类最少样本数: {min(labels.values())}")
                    print(f"    - 每类最多样本数: {max(labels.values())}")
                    # 显示部分标签
                    print(f"    - 标签示例 (前10个):")
                    for i, (lbl, cnt) in enumerate(list(labels.items())[:10]):
                        print(f"        {lbl}: {cnt} 个样本")
                else:
                    print(f"    - 标签类型: 整数")
                    print(f"    - 标签范围: {min(labels.keys())} 到 {max(labels.keys())}")
                    print(f"    - 不同标签数: {len(labels)}")
                    print(f"    - 每类平均样本数: {np.mean(list(labels.values())):.2f}")
                    print(f"    - 每类最少样本数: {min(labels.values())}")
                    print(f"    - 每类最多样本数: {max(labels.values())}")
            
            if video_names:
                print(f"\n  视频名称示例 (前5个):")
                for name in video_names[:5]:
                    print(f"    - {name}")
            
            return {
                'num_samples': num_samples,
                'frame_stats': {
                    'mean': np.mean(frame_counts) if frame_counts else 0,
                    'min': np.min(frame_counts) if frame_counts else 0,
                    'max': np.max(frame_counts) if frame_counts else 0,
                },
                'num_classes': len(labels),
                'feature_dims': list(features_dims)
            }

    except Exception as e:
        print(f"读取文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_dataset(dataset_path, sample_limit=None):
    """
    检测指定数据集目录的完整数据结构信息
    
    Args:
        dataset_path: 数据集目录路径 (如 dataset/processed/WLASL100)
        sample_limit: 每个文件的采样数量限制
    """
    if not os.path.exists(dataset_path):
        print(f"错误: 目录 {dataset_path} 不存在")
        return
    
    if not os.path.isdir(dataset_path):
        print(f"错误: {dataset_path} 不是一个目录")
        return
    
    dataset_name = os.path.basename(dataset_path)
    
    print("\n" + "#" * 70)
    print(f"#  数据结构检测工具 - {dataset_name}")
    print("#" * 70)
    print(f"\n目标目录: {os.path.abspath(dataset_path)}")
    
    # 列出目录内容
    all_files = os.listdir(dataset_path)
    hdf5_files = sorted([f for f in all_files if f.endswith('.hdf5')])
    json_files = [f for f in all_files if f.endswith('.json')]
    other_files = [f for f in all_files if f not in hdf5_files and f not in json_files]
    
    print(f"\n[目录内容]")
    print(f"  HDF5 文件: {len(hdf5_files)} 个")
    print(f"  JSON 文件: {len(json_files)} 个")
    if other_files:
        print(f"  其他文件: {len(other_files)} 个")
    
    # 检查 JSON 标签映射文件
    if json_files:
        print(f"\n[标签映射文件]")
        for jf in json_files:
            jf_path = os.path.join(dataset_path, jf)
            print(f"\n  文件: {jf}")
            try:
                with open(jf_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                    print(f"  词汇表大小: {len(label_map)}")
                    
                    # 显示部分词汇
                    if isinstance(label_map, dict):
                        items = list(label_map.items())[:10]
                        print(f"  示例词汇 (前10个):")
                        for key, value in items:
                            print(f"    {key}: {value}")
                        if len(label_map) > 10:
                            print(f"    ... 共 {len(label_map)} 个词汇")
            except Exception as e:
                print(f"  读取失败: {e}")
    
    # 检查 HDF5 文件
    total_stats = {
        'total_samples': 0,
        'train_samples': 0,
        'val_samples': 0,
        'test_samples': 0,
    }
    
    for hf in hdf5_files:
        hf_path = os.path.join(dataset_path, hf)
        stats = inspect_hdf5_complete(hf_path, sample_limit)
        
        if stats:
            total_stats['total_samples'] += stats['num_samples']
            if 'Train' in hf:
                total_stats['train_samples'] = stats['num_samples']
            elif 'Val' in hf:
                total_stats['val_samples'] = stats['num_samples']
            elif 'Test' in hf:
                total_stats['test_samples'] = stats['num_samples']
    
    # 汇总统计
    print(f"\n{'='*70}")
    print(f"[汇总统计]")
    print(f"{'='*70}")
    print(f"  数据集: {dataset_name}")
    print(f"  总样本数: {total_stats['total_samples']}")
    print(f"  训练集: {total_stats['train_samples']}")
    print(f"  验证集: {total_stats['val_samples']}")
    print(f"  测试集: {total_stats['test_samples']}")
    
    if total_stats['total_samples'] > 0:
        train_ratio = total_stats['train_samples'] / total_stats['total_samples'] * 100
        val_ratio = total_stats['val_samples'] / total_stats['total_samples'] * 100
        test_ratio = total_stats['test_samples'] / total_stats['total_samples'] * 100
        print(f"  数据划分比例: 训练 {train_ratio:.1f}% / 验证 {val_ratio:.1f}% / 测试 {test_ratio:.1f}%")


def main():
    """
    主函数 - 优先使用硬编码配置，否则使用命令行参数
    """
    # 检查是否使用硬编码配置
    use_hardcoded_config = CONFIG.get("DATASET_PATH") is not None
    
    if use_hardcoded_config:
        # 使用硬编码配置
        print("\n[配置模式] 使用硬编码配置 (CONFIG)")
        dataset_path = CONFIG["DATASET_PATH"]
        sample_limit = CONFIG.get("SAMPLE_LIMIT")
        
        print(f"  DATASET_PATH: {dataset_path}")
        print(f"  SAMPLE_LIMIT: {sample_limit if sample_limit else '全部样本'}")
    else:
        # 使用命令行参数
        parser = argparse.ArgumentParser(
            description='检测 WLASL 数据集的数据结构信息',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  方法1 (推荐): 修改文件顶部的 CONFIG["DATASET_PATH"]，然后直接运行:
    python src/test/processed_data_test.py
  
  方法2: 通过命令行参数:
    python src/test/processed_data_test.py dataset/processed/WLASL100
    python src/test/processed_data_test.py dataset/processed/WLASL100 --sample-limit 50
    python src/test/processed_data_test.py d:/Document/Code/csl-lstm/dataset/processed/WLASL100
            """
        )
        
        parser.add_argument(
            'dataset_path',
            type=str,
            nargs='?',  # 可选参数
            default=None,
            help='数据集目录路径 (如: dataset/processed/WLASL100)'
        )
        
        parser.add_argument(
            '--sample-limit', '-n',
            type=int,
            default=None,
            help='每个 HDF5 文件的采样数量限制 (默认: 检查全部)'
        )
        
        args = parser.parse_args()
        
        if args.dataset_path is None:
            print("\n错误: 未指定数据集路径!")
            print("请使用以下方式之一:")
            print("  1. 修改文件顶部的 CONFIG['DATASET_PATH'] 为有效路径")
            print("  2. 通过命令行参数传入路径: python src/test/processed_data_test.py <路径>")
            sys.exit(1)
        
        dataset_path = args.dataset_path
        sample_limit = args.sample_limit
        print("\n[配置模式] 使用命令行参数")
    
    # 处理相对路径
    if not os.path.isabs(dataset_path):
        # 尝试从当前目录解析
        if os.path.exists(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        else:
            # 尝试从脚本所在目录解析
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            dataset_path = os.path.join(project_root, dataset_path)
    
    inspect_dataset(dataset_path, sample_limit)


if __name__ == "__main__":
    main()
