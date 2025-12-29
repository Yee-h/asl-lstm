import h5py
import numpy as np

# --- 配置文件路径 ---
# 注意：在 Windows 下路径分隔符可以使用 '/' 或 '\\'
file_path = "d:/Document/Code/csl-lstm/dataset/processed/WLASL100/WLASL100_135-Test.hdf5"

try:
    # 以只读模式打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        # 1. 检查文件内容
        if not f.keys():
            print("错误：文件中没有样本数据！")
            exit()
            
        # 选择第一个样本进行检查
        sample_id = list(f.keys())[0]
        
        # 2. 读取样本数据
        # 数据的预期结构通常为: (总帧数, 坐标维度, 关键点数量)
        # 这里的坐标维度通常是 2 (x, y) 或 3 (x, y, z)
        data = f[sample_id]['data'][:]
        
        print(f"\n{'='*50}")
        print(f"检查样本 ID: {sample_id}")
        print(f"数据总形状 (Data Shape): {data.shape}")
        print(f"  - 帧数 (Frames): {data.shape[0]}")
        print(f"  - 坐标维度 (Dims): {data.shape[1]} (通常为 x, y)")
        print(f"  - 关键点数量 (Landmarks): {data.shape[2]}")
        print(f"{'='*50}\n")
        
        # 3. 打印具体数值以验证数据有效性
        # 取第一帧 (索引为 0) 进行展示
        frame_idx = 0
        frame_data = data[frame_idx]  # 形状为 (2, 135)
        
        print(f"【第 {frame_idx+1} 帧 - 前 5 个关键点的具体坐标值】:")
        print("-" * 40)
        
        for i in range(5):
            # 获取对应的 x 和 y 坐标
            x_val = frame_data[0, i]
            y_val = frame_data[1, i]
            print(f"  关键点 {i: >2}: x = {x_val: .6f},  y = {y_val: .6f}")
            
        print("-" * 40)
        print("\n数据检查提示：如果能够看到非零的浮点数值，说明特征提取成功。")

except Exception as e:
    # 异常处理，捕获读取过程中可能出现的错误
    print(f"程序运行发生错误: {e}")
    print("建议：请确认 HDF5 文件路径是否正确，且文件没有被其他进程占用。")