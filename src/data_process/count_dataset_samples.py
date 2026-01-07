import h5py
import os
from pathlib import Path

# ================= 配置区域 =================
# 数据集目录路径 (硬编码，用户可修改)
# 例如: 
# D:\Document\Code\process_wlasl\pose_action_dataset\WLASL100
# D:\Document\Code\process_wlasl\pose_action_dataset\WLASL300
DATASET_DIR = r'D:\Document\\Code\\asl-lstm\dataset\\processed\WLASL100'
# ===========================================

def count_samples():
    dataset_path = Path(DATASET_DIR)
    
    if not dataset_path.exists():
        print(f"错误: 目录不存在 - {dataset_path}")
        return

    print(f"{'='*60}")
    print(f"数据集样本统计")
    print(f"目标目录: {dataset_path}")
    print(f"{'='*60}")

    # 查找所有 .hdf5 文件
    hdf5_files = sorted(dataset_path.glob('*.hdf5'))
    
    if not hdf5_files:
        print("该目录下未找到任何 .hdf5 文件。")
        return

    total_all = 0
    
    # 打印表头
    print(f"\n{'文件名':<35} | {'样本数量':>10}")
    print(f"{'-'*35}-|-{'-'*10}")
    
    for h5_file in hdf5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                # 统计样本数 (Keys 的数量)
                count = len(f.keys())
                print(f"{h5_file.name:<35} | {count:>10}")
                total_all += count
        except Exception as e:
            print(f"{h5_file.name:<35} | 错误: {e}")

    print(f"{'-'*35}-|-{'-'*10}")
    print(f"{'总计 (Total)':<35} | {total_all:>10}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    count_samples()
