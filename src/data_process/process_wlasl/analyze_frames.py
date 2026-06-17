import h5py
import numpy as np

def analyze_frame_distribution(file_path):
    lengths = []
    print(f"正在分析文件: {file_path} ...")
    
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            # 假设数据在 'data' 字段，且 shape[0] 是帧数
            if 'data' in f[key]:
                t = f[key]['data'].shape[0]
                lengths.append(t)
    
    lengths = np.array(lengths)
    
    print("\n==== 帧数统计报告 ====")
    print(f"样本总数: {len(lengths)}")
    print(f"最小帧数: {np.min(lengths)}")
    print(f"最大帧数: {np.max(lengths)}")
    print(f"平均帧数: {np.mean(lengths):.2f}")
    print(f"中位数帧数: {np.median(lengths)}")
    
    # 关键的分位点
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    print(f"90% 的样本长度小于: {p90:.0f}")
    print(f"95% 的样本长度小于: {p95:.0f}")
    print(f"99% 的样本长度小于: {p99:.0f}")
    
    return int(p95)

if __name__ == "__main__":
    file_path = r'D:\Document\\Code\\process_wlasl\\pose_action_dataset\WLASL100\WLASL100_135-Train.hdf5'
    analyze_frame_distribution(file_path)
    