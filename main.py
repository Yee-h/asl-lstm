"""
CSL-LSTM 主入口
中国手语识别 LSTM 模型训练项目
"""

import argparse
import sys


def main():
    """主函数 - 提供命令行接口"""
    parser = argparse.ArgumentParser(
        description='CSL-LSTM: 中国手语识别项目',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据预处理命令
    preprocess_parser = subparsers.add_parser(
        'preprocess', 
        help='运行数据预处理 - 从视频中提取特征'
    )
    preprocess_parser.add_argument(
        '--split', 
        type=str, 
        choices=['train', 'test', 'dev'],
        help='仅处理指定的数据集划分'
    )
    preprocess_parser.add_argument(
        '--no-multiprocessing',
        action='store_true',
        help='禁用多进程 (调试用)'
    )
    preprocess_parser.add_argument(
        '--limit',
        type=str,
        default=None,
        help='限制处理的视频数量 (例如: 10 或 all)，用于快速测试'
    )
    
    # 统计信息命令
    stats_parser = subparsers.add_parser(
        'stats',
        help='显示数据集统计信息'
    )
    stats_parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'dev'],
        help='仅显示指定划分的统计信息'
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        from data_preprocessing import preprocess_all
        import config as cfg
        
        splits = [args.split] if args.split else None
        
        # 如果命令行指定了禁用多进程，则覆盖配置
        if args.no_multiprocessing:
            cfg.USE_MULTIPROCESSING = False
            
        # 如果命令行指定了数量限制，则覆盖配置
        if args.limit:
            if args.limit.lower() == 'all':
                cfg.DATA_LIMIT = None
            else:
                try:
                    cfg.DATA_LIMIT = int(args.limit)
                except ValueError:
                    print(f"❌ 错误: --limit 参数必须是整数或 'all'，收到: {args.limit}")
                    return
        
        preprocess_all(splits=splits)
        
    elif args.command == 'stats':
        from utils import get_dataset_stats
        import config as cfg
        
        print("="*50)
        print("📊 数据集统计信息")
        print("="*50)
        
        stats = get_dataset_stats(args.split)
        
        for split_name, split_stats in stats.items():
            print(f"\n📂 {split_name.upper()}")
            print(f"   总计: {split_stats['count']} 个样本")
            if split_stats['translators']:
                print("   各翻译者分布:")
                for translator, count in sorted(split_stats['translators'].items()):
                    print(f"     {translator}: {count}")
    
    else:
        parser.print_help()
        print("\n示例用法:")
        print("  python main.py preprocess          # 运行完整数据预处理")
        print("  python main.py preprocess --split train  # 仅处理训练集")
        print("  python main.py stats               # 查看数据集统计")


if __name__ == "__main__":
    main()
