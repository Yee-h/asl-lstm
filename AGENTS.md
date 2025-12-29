所有配置参数信息均在config.py中,只能通过导入使用参数，不允许写死在函数中

check_coords.py执行后的输出：
[Running] set PYTHONIOENCODING=utf8 && d:\Document\Code\csl-lstm\.venv\Scripts\python.exe -u "d:\Document\Code\csl-lstm\src\test\check_coords.py"

========================================
样本ID: 0
数据总形状: (35, 2, 135)
  -> 35 帧
  -> 2 个坐标维度 (x, y)
  -> 135 个骨骼关键点
========================================

【第 1 帧 - 前 5 个关键点的坐标值】:
------------------------------
  关键点 0: x = 0.449527,  y = 0.297011
  关键点 1: x = 0.492519,  y = 0.262340
  关键点 2: x = 0.416243,  y = 0.267888
  关键点 3: x = 0.550766,  y = 0.288690
  关键点 4: x = 0.387120,  y = 0.303945
------------------------------

看到上面具体的 x, y 小数了吗？这就是要训练的数据！

[Done] exited with code=0 in 0.234 seconds



processed_data_test.py实行后的输出：
[Running] set PYTHONIOENCODING=utf8 && d:\Document\Code\csl-lstm\.venv\Scripts\python.exe -u "d:\Document\Code\csl-lstm\src\test\processed_data_test.py"

[配置模式] 使用硬编码配置 (CONFIG)
  DATASET_PATH: d:/Document/Code/csl-lstm/dataset/processed/WLASL100
  SAMPLE_LIMIT: 10

######################################################################
#  数据结构检测工具 - WLASL100
######################################################################

目标目录: d:\Document\Code\csl-lstm\dataset\processed\WLASL100

[目录内容]
  HDF5 文件: 3 个
  JSON 文件: 1 个

[标签映射文件]

  文件: wlasl_100_maplabels.json
  词汇表大小: 2
  示例词汇 (前10个):
    id_to_label: {'book': 0, 'drink': 1, 'computer': 2, 'before': 3, 'chair': 4, 'go': 5, 'clothes': 6, 'who': 7, 'candy': 8, 'cousin': 9, 'deaf': 10, 'fine': 11, 'help': 12, 'no': 13, 'thin': 14, 'walk': 15, 'year': 16, 'yes': 17, 'all': 18, 'black': 19, 'cool': 20, 'finish': 21, 'hot': 22, 'like': 23, 'many': 24, 'mother': 25, 'now': 26, 'orange': 27, 'table': 28, 'thanksgiving': 29, 'what': 30, 'woman': 31, 'bed': 32, 'blue': 33, 'bowling': 34, 'can': 35, 'dog': 36, 'family': 37, 'fish': 38, 'graduate': 39, 'hat': 40, 'hearing': 41, 'kiss': 42, 'language': 43, 'later': 44, 'man': 45, 'shirt': 46, 'study': 47, 'tall': 48, 'white': 49, 'wrong': 50, 'accident': 51, 'apple': 52, 'bird': 53, 'change': 54, 'color': 55, 'corn': 56, 'cow': 57, 'dance': 58, 'dark': 59, 'doctor': 60, 'eat': 61, 'enjoy': 62, 'forget': 63, 'give': 64, 'last': 65, 'meet': 66, 'pink': 67, 'pizza': 68, 'play': 69, 'school': 70, 'secretary': 71, 'short': 72, 'time': 73, 'want': 74, 'work': 75, 'africa': 76, 'basketball': 77, 'birthday': 78, 'brown': 79, 'but': 80, 'cheat': 81, 'city': 82, 'cook': 83, 'decide': 84, 'full': 85, 'how': 86, 'jacket': 87, 'letter': 88, 'medicine': 89, 'need': 90, 'paint': 91, 'paper': 92, 'pull': 93, 'purple': 94, 'right': 95, 'same': 96, 'son': 97, 'tell': 98, 'thursday': 99}
    label_to_id: {'0': 'book', '1': 'drink', '2': 'computer', '3': 'before', '4': 'chair', '5': 'go', '6': 'clothes', '7': 'who', '8': 'candy', '9': 'cousin', '10': 'deaf', '11': 'fine', '12': 'help', '13': 'no', '14': 'thin', '15': 'walk', '16': 'year', '17': 'yes', '18': 'all', '19': 'black', '20': 'cool', '21': 'finish', '22': 'hot', '23': 'like', '24': 'many', '25': 'mother', '26': 'now', '27': 'orange', '28': 'table', '29': 'thanksgiving', '30': 'what', '31': 'woman', '32': 'bed', '33': 'blue', '34': 'bowling', '35': 'can', '36': 'dog', '37': 'family', '38': 'fish', '39': 'graduate', '40': 'hat', '41': 'hearing', '42': 'kiss', '43': 'language', '44': 'later', '45': 'man', '46': 'shirt', '47': 'study', '48': 'tall', '49': 'white', '50': 'wrong', '51': 'accident', '52': 'apple', '53': 'bird', '54': 'change', '55': 'color', '56': 'corn', '57': 'cow', '58': 'dance', '59': 'dark', '60': 'doctor', '61': 'eat', '62': 'enjoy', '63': 'forget', '64': 'give', '65': 'last', '66': 'meet', '67': 'pink', '68': 'pizza', '69': 'play', '70': 'school', '71': 'secretary', '72': 'short', '73': 'time', '74': 'want', '75': 'work', '76': 'africa', '77': 'basketball', '78': 'birthday', '79': 'brown', '80': 'but', '81': 'cheat', '82': 'city', '83': 'cook', '84': 'decide', '85': 'full', '86': 'how', '87': 'jacket', '88': 'letter', '89': 'medicine', '90': 'need', '91': 'paint', '92': 'paper', '93': 'pull', '94': 'purple', '95': 'right', '96': 'same', '97': 'son', '98': 'tell', '99': 'thursday'}

============================================================
文件: WLASL100_135-Test.hdf5
路径: d:/Document/Code/csl-lstm/dataset/processed/WLASL100\WLASL100_135-Test.hdf5
============================================================

[基本信息]
  样本总数: 258
  文件大小: 32.26 MB

[完整 HDF5 层级结构] (显示前5个样本)
--------------------------------------------------
  HDF5 File: WLASL100_135-Test.hdf5
  ├── 文件属性:
  ├── 顶级组 (共 258 个样本):
  │   ├── [0]
  │   │   ├── data: Dataset[(35, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = accident
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/test/00625.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [1]
  │   │   ├── data: Dataset[(35, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = accident
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/test/00634.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [10]
  │   │   ├── data: Dataset[(46, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = basketball
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/test/05227.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [100]
  │   │   ├── data: Dataset[(97, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = last
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/test/32246.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [101]
  │   │   ├── data: Dataset[(67, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = now
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/test/38991.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   └── ... 还有 253 个样本未显示
--------------------------------------------------

[数据结构] (示例 ID: 0)
  - data: dtype=float64, shape=(35, 2, 135)
  - height: dtype=int64, value=256
  - label: dtype=object, value=accident
  - video_name: dtype=object, value=rgb/WLASL100/test/00625.mp4
  - width: dtype=int64, value=256

[正在分析 10 个样本...]

[数据统计] (基于 10/258 个样本)

  帧数分布:
    - 平均帧数: 49.80
    - 最小帧数: 27
    - 最大帧数: 97
    - 标准差: 19.52
    - 帧数区间分布:
        [0-30): 1 个样本
        [30-60): 7 个样本
        [60-90): 1 个样本
        [90-98): 1 个样本

  特征维度:
    - shape: (2, 135)
      解释: (坐标轴=2 [x,y], 关键点数=135)

  标签分布:
    - 标签类型: 字符串 (gloss名称)
    - 不同标签数 (类别数): 9
    - 每类平均样本数: 1.11
    - 每类最少样本数: 1
    - 每类最多样本数: 2
    - 标签示例 (前10个):
        accident: 2 个样本
        basketball: 1 个样本
        last: 1 个样本
        now: 1 个样本
        secretary: 1 个样本
        thin: 1 个样本
        work: 1 个样本
        computer: 1 个样本
        graduate: 1 个样本

  视频名称示例 (前5个):
    - rgb/WLASL100/test/00625.mp4
    - rgb/WLASL100/test/00634.mp4
    - rgb/WLASL100/test/05227.mp4
    - rgb/WLASL100/test/32246.mp4
    - rgb/WLASL100/test/38991.mp4

============================================================
文件: WLASL100_135-Train.hdf5
路径: d:/Document/Code/csl-lstm/dataset/processed/WLASL100\WLASL100_135-Train.hdf5
============================================================

[基本信息]
  样本总数: 1442
  文件大小: 188.89 MB

[完整 HDF5 层级结构] (显示前5个样本)
--------------------------------------------------
  HDF5 File: WLASL100_135-Train.hdf5
  ├── 文件属性:
  ├── 顶级组 (共 1442 个样本):
  │   ├── [0]
  │   │   ├── data: Dataset[(67, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = dark
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/train/14675.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [1]
  │   │   ├── data: Dataset[(82, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = many
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/train/34835.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [10]
  │   │   ├── data: Dataset[(23, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = accident
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/train/00633.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [100]
  │   │   ├── data: Dataset[(78, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = bird
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/train/06332.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [1000]
  │   │   ├── data: Dataset[(43, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = thin
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/train/57919.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   └── ... 还有 1437 个样本未显示
--------------------------------------------------

[数据结构] (示例 ID: 0)
  - data: dtype=float64, shape=(67, 2, 135)
  - height: dtype=int64, value=256
  - label: dtype=object, value=dark
  - video_name: dtype=object, value=rgb/WLASL100/train/14675.mp4
  - width: dtype=int64, value=256

[正在分析 10 个样本...]

[数据统计] (基于 10/1442 个样本)

  帧数分布:
    - 平均帧数: 53.20
    - 最小帧数: 23
    - 最大帧数: 82
    - 标准差: 23.76
    - 帧数区间分布:
        [0-30): 3 个样本
        [30-60): 2 个样本
        [60-83): 5 个样本

  特征维度:
    - shape: (2, 135)
      解释: (坐标轴=2 [x,y], 关键点数=135)

  标签分布:
    - 标签类型: 字符串 (gloss名称)
    - 不同标签数 (类别数): 5
    - 每类平均样本数: 2.00
    - 每类最少样本数: 1
    - 每类最多样本数: 6
    - 标签示例 (前10个):
        dark: 1 个样本
        many: 1 个样本
        accident: 1 个样本
        bird: 1 个样本
        thin: 6 个样本

  视频名称示例 (前5个):
    - rgb/WLASL100/train/14675.mp4
    - rgb/WLASL100/train/34835.mp4
    - rgb/WLASL100/train/00633.mp4
    - rgb/WLASL100/train/06332.mp4
    - rgb/WLASL100/train/57919.mp4

============================================================
文件: WLASL100_135-Val.hdf5
路径: d:/Document/Code/csl-lstm/dataset/processed/WLASL100\WLASL100_135-Val.hdf5
============================================================

[基本信息]
  样本总数: 338
  文件大小: 45.45 MB

[完整 HDF5 层级结构] (显示前5个样本)
--------------------------------------------------
  HDF5 File: WLASL100_135-Val.hdf5
  ├── 文件属性:
  ├── 顶级组 (共 338 个样本):
  │   ├── [0]
  │   │   ├── data: Dataset[(46, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = accident
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/val/00626.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [1]
  │   │   ├── data: Dataset[(28, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = accident
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/val/00627.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [10]
  │   │   ├── data: Dataset[(53, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = basketball
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/val/05236.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [100]
  │   │   ├── data: Dataset[(34, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = go
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/val/24946.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   ├── [101]
  │   │   ├── data: Dataset[(25, 2, 135)] dtype=float64
  │   │   ├── height: Dataset[scalar] dtype=int64 = 256
  │   │   ├── label: Dataset[scalar] dtype=object = last
  │   │   ├── video_name: Dataset[scalar] dtype=object = rgb/WLASL100/val/32255.mp4
  │   │   └── width: Dataset[scalar] dtype=int64 = 256
  │   └── ... 还有 333 个样本未显示
--------------------------------------------------

[数据结构] (示例 ID: 0)
  - data: dtype=float64, shape=(46, 2, 135)
  - height: dtype=int64, value=256
  - label: dtype=object, value=accident
  - video_name: dtype=object, value=rgb/WLASL100/val/00626.mp4
  - width: dtype=int64, value=256

[正在分析 10 个样本...]

[数据统计] (基于 10/338 个样本)

  帧数分布:
    - 平均帧数: 51.10
    - 最小帧数: 25
    - 最大帧数: 87
    - 标准差: 19.37
    - 帧数区间分布:
        [0-30): 2 个样本
        [30-60): 6 个样本
        [60-88): 2 个样本

  特征维度:
    - shape: (2, 135)
      解释: (坐标轴=2 [x,y], 关键点数=135)

  标签分布:
    - 标签类型: 字符串 (gloss名称)
    - 不同标签数 (类别数): 9
    - 每类平均样本数: 1.11
    - 每类最少样本数: 1
    - 每类最多样本数: 2
    - 标签示例 (前10个):
        accident: 2 个样本
        basketball: 1 个样本
        go: 1 个样本
        last: 1 个样本
        no: 1 个样本
        pull: 1 个样本
        short: 1 个样本
        thursday: 1 个样本
        wrong: 1 个样本

  视频名称示例 (前5个):
    - rgb/WLASL100/val/00626.mp4
    - rgb/WLASL100/val/00627.mp4
    - rgb/WLASL100/val/05236.mp4
    - rgb/WLASL100/val/24946.mp4
    - rgb/WLASL100/val/32255.mp4

======================================================================
[汇总统计]
======================================================================
  数据集: WLASL100
  总样本数: 2038
  训练集: 1442
  验证集: 338
  测试集: 258
  数据划分比例: 训练 70.8% / 验证 16.6% / 测试 12.7%

[Done] exited with code=0 in 0.285 seconds

