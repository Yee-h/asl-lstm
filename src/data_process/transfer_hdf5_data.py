#!/usr/bin/env python3
"""
WLASL 数据集 HDF5 文件数据迁移工具

功能：
1. 读取指定数据集文件夹中的 *.hdf5 和 *.json 文件
2. 将指定百分比的数据从源 HDF5 文件复制到目标 HDF5 文件
3. 自动修改复制数据的 video_name 字段中的子集划分标识（如 Test -> Train）

使用方法：
    uv run src\\data_process\\transfer_hdf5_data.py

配置参数在 CONFIG 部分进行硬编码修改
"""

from pathlib import Path
import sys
import os
import json
import random
from typing import Any, Optional
import h5py

# 添加项目根目录到 Python 路径，以便导入 src.config
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg
from src.core.hdf5_schema import missing_required_fields


# ==================== 硬编码配置区域（用户可自行修改） ====================
class CONFIG:
    """配置参数 - 用户可以根据需要修改这些值"""

    # 数据集规模：100、300 或 2000
    DATASET_SCALE: str = "100"

    # 数据迁移百分比（0.0 ~ 1.0）
    # 例如：0.5 表示将 50% 的数据从源文件复制到目标文件
    TRANSFER_PERCENTAGE: float = 0.6

    # 源文件的数据集划分：Train、Val 或 Test
    SOURCE_SPLIT: str = "Val"

    # 目标文件的数据集划分：Train、Val 或 Test
    TARGET_SPLIT: str = "Train"

    # 是否从源文件中删除已复制的数据（True: 移动模式，False: 复制模式）
    DELETE_FROM_SOURCE: bool = False

    # 随机种子（用于可复现的随机采样，设为 None 则每次运行结果不同）
    RANDOM_SEED: Optional[int] = None

    # 是否显示详细信息
    VERBOSE: bool = True


# ========================================================================


def _enable_utf8_console_output() -> None:
    """确保 Windows 终端能够正确显示 UTF-8 字符。"""
    if os.name != "nt":
        return

    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def get_data_dir(dataset_scale: str) -> Path:
    """获取数据集目录路径"""
    # 使用 src.config 中定义的项目根目录
    project_root = Path(cfg.PATHS.project_root)
    return project_root / "dataset" / "processed" / f"WLASL{dataset_scale}"


def build_hdf5_path(dataset_scale: str, split: str) -> Path:
    """构建 HDF5 文件路径"""
    data_dir = get_data_dir(dataset_scale)
    filename = f"WLASL{dataset_scale}_135-{split}.hdf5"
    return data_dir / filename


def list_files_in_dataset(dataset_scale: str) -> tuple[list[Path], list[Path]]:
    """列出数据集目录中的所有 HDF5 和 JSON 文件"""
    data_dir = get_data_dir(dataset_scale)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    json_files = sorted(data_dir.glob("*.json"))

    return hdf5_files, json_files


def load_json_files(json_files: list[Path]) -> dict[str, Any]:
    """加载所有 JSON 文件"""
    all_data = {}
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            all_data[json_path.name] = json.load(f)
    return all_data


def update_video_name(video_name: str, source_split: str, target_split: str) -> str:
    """
    更新 video_name 中的子集划分标识

    例如: "rgb/WLASL100/test/14675.mp4" -> "rgb/WLASL100/train/14675.mp4"
    """
    # 转换为小写进行替换（video_name 中的路径通常是小写）
    source_lower = source_split.lower()
    target_lower = target_split.lower()

    # 替换路径中的子集标识
    updated_name = video_name.replace(f"/{source_lower}/", f"/{target_lower}/")

    return updated_name


def get_next_available_id(target_file: h5py.File) -> int:
    """
    获取目标文件中下一个可用的数字ID

    遍历所有键，找到最大的数字键，返回 max + 1
    如果没有任何数字键，返回 0
    """
    max_id = -1
    for key in target_file.keys():
        if key.isdigit():
            val = int(key)
            if val > max_id:
                max_id = val
    return max_id + 1


def copy_sample_to_target(
    source_file: h5py.File,
    target_file: h5py.File,
    sample_key: str,
    source_split: str,
    target_split: str,
    target_key_name: str,
    verbose: bool = True,
) -> str:
    """
    将单个样本从源文件完整复制到目标文件

    保持所有数据结构和属性不变，仅修改 video_name 中的子集划分标识

    Args:
        source_file: 源 HDF5 文件
        target_file: 目标 HDF5 文件
        sample_key: 源样本的键名
        source_split: 源数据集划分
        target_split: 目标数据集划分
        target_key_name: 目标文件中的键名（必须指定）
        verbose: 是否显示详细信息

    Returns:
        实际使用的目标键名
    """
    source_group = source_file[sample_key]

    target_key = target_key_name

    # 如果目标文件中已存在该键，先删除
    if target_key in target_file:
        if verbose:
            print(f"  警告: 目标文件中已存在键 '{target_key}'，将覆盖")
        del target_file[target_key]

    # 创建新的组
    target_group = target_file.create_group(target_key)

    # 复制组的属性
    for attr_name, attr_value in source_group.attrs.items():
        target_group.attrs[attr_name] = attr_value

    # 复制所有数据集
    for dataset_name in source_group.keys():  # type: ignore[union-attr]
        source_dataset = source_group[dataset_name]  # type: ignore[index]

        if isinstance(source_dataset, h5py.Dataset):
            # 读取原始数据
            data = source_dataset[()]

            # 如果是 video_name，需要修改子集标识
            if dataset_name == "video_name":
                if isinstance(data, bytes):
                    video_name = data.decode("utf-8")
                else:
                    video_name = str(data)

                # 更新子集划分
                updated_video_name = update_video_name(
                    video_name, source_split, target_split
                )
                data = updated_video_name

            # 创建新的数据集，保持相同的 dtype
            if source_dataset.dtype.kind == "O":
                # 对象类型（字符串等）
                dt = h5py.special_dtype(vlen=str)
                target_group.create_dataset(dataset_name, data=data, dtype=dt)
            else:
                # 数值类型
                target_group.create_dataset(
                    dataset_name, data=data, dtype=source_dataset.dtype
                )

            # 复制数据集的属性
            for attr_name, attr_value in source_dataset.attrs.items():
                target_group[dataset_name].attrs[attr_name] = attr_value

        elif isinstance(source_dataset, h5py.Group):
            # 递归复制子组（如果存在）
            source_file.copy(source_dataset, target_group, name=dataset_name)

    return target_key


def transfer_data(
    dataset_scale: str,
    source_split: str,
    target_split: str,
    transfer_percentage: float,
    delete_from_source: bool = False,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    执行数据迁移操作

    Args:
        dataset_scale: 数据集规模（100、300、2000）
        source_split: 源数据集划分（Train、Val、Test）
        target_split: 目标数据集划分（Train、Val、Test）
        transfer_percentage: 迁移百分比（0.0 ~ 1.0）
        delete_from_source: 是否从源文件删除已复制的数据
        random_seed: 随机种子
        verbose: 是否显示详细信息

    Returns:
        包含迁移统计信息的字典
    """
    # 验证参数
    if not 0.0 < transfer_percentage <= 1.0:
        raise ValueError("transfer_percentage 必须在 (0.0, 1.0] 范围内")

    if source_split == target_split:
        raise ValueError("source_split 和 target_split 不能相同")

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

    # 构建文件路径
    source_path = build_hdf5_path(dataset_scale, source_split)
    target_path = build_hdf5_path(dataset_scale, target_split)

    if not source_path.exists():
        raise FileNotFoundError(f"源文件不存在: {source_path}")

    if not target_path.exists():
        raise FileNotFoundError(f"目标文件不存在: {target_path}")

    if verbose:
        print(f"\n{'=' * 60}")
        print("WLASL 数据迁移工具")
        print(f"{'=' * 60}")
        print(f"数据集规模: WLASL{dataset_scale}")
        print(f"源文件: {source_path.name}")
        print(f"目标文件: {target_path.name}")
        print(f"迁移百分比: {transfer_percentage * 100:.1f}%")
        print(f"删除源数据: {'是' if delete_from_source else '否'}")
        print(f"随机种子: {random_seed}")
        print(f"{'=' * 60}\n")

    # 打开源文件读取样本键
    with h5py.File(source_path, "r") as source_file:
        all_keys = list(source_file.keys())
        total_samples = len(all_keys)

        # 计算要迁移的样本数量
        num_to_transfer = int(total_samples * transfer_percentage)

        if num_to_transfer == 0:
            print("警告: 根据当前百分比，没有样本需要迁移")
            return {"transferred": 0, "total": total_samples}

        # 随机选择要迁移的样本
        samples_to_transfer = random.sample(all_keys, num_to_transfer)

        if verbose:
            print(f"源文件样本总数: {total_samples}")
            print(f"将要迁移的样本数: {num_to_transfer}")
            print("\n开始迁移数据...")

    # 迁移数据
    transferred_count = 0
    source_keys = []  # 源文件中的键（用于删除）
    target_keys = []  # 目标文件中实际使用的键（用于验证）

    # 确定源文件的打开模式
    source_mode = "r+" if delete_from_source else "r"

    with h5py.File(source_path, source_mode) as source_file:
        with h5py.File(target_path, "r+") as target_file:
            original_target_count = len(target_file.keys())

            # 获取下一个可用的数字ID
            next_id = get_next_available_id(target_file)
            if verbose:
                print(f"新样本将从 ID {next_id} 开始编号")

            for i, sample_key in enumerate(samples_to_transfer):
                try:
                    # 使用纯数字递增 ID
                    new_target_key = str(next_id)
                    next_id += 1

                    target_key = copy_sample_to_target(
                        source_file,
                        target_file,
                        sample_key,
                        source_split,
                        target_split,
                        target_key_name=new_target_key,
                        verbose=False,  # 避免过多输出
                    )
                    transferred_count += 1
                    source_keys.append(sample_key)
                    target_keys.append(target_key)

                    if verbose and (i + 1) % 50 == 0:
                        print(f"  已迁移: {i + 1}/{num_to_transfer}")

                except Exception as e:
                    print(f"  错误: 迁移样本 '{sample_key}' 时出错: {e}")

            # 如果需要，从源文件删除已迁移的数据
            if delete_from_source and source_keys:
                if verbose:
                    print(f"\n从源文件删除已迁移的 {len(source_keys)} 个样本...")
                for key in source_keys:
                    if key in source_file:
                        del source_file[key]

            final_target_count = len(target_file.keys())

    # 统计结果
    result = {
        "source_file": source_path.name,
        "target_file": target_path.name,
        "total_in_source": total_samples,
        "transferred": transferred_count,
        "transfer_percentage": transfer_percentage,
        "original_target_count": original_target_count,
        "final_target_count": final_target_count,
        "deleted_from_source": delete_from_source,
        "source_keys": source_keys,
        "target_keys": target_keys,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("迁移完成!")
        print(f"{'=' * 60}")
        print(f"成功迁移样本数: {transferred_count}")
        print(f"目标文件样本数: {original_target_count} -> {final_target_count}")
        if delete_from_source:
            remaining = total_samples - transferred_count
            print(f"源文件剩余样本数: {remaining}")
        print(f"{'=' * 60}\n")

    return result


def verify_transfer(
    dataset_scale: str, target_split: str, sample_keys: list[str], verbose: bool = True
) -> bool:
    """
    验证迁移后的数据完整性

    Args:
        dataset_scale: 数据集规模
        target_split: 目标数据集划分
        sample_keys: 要验证的样本键列表
        verbose: 是否显示详细信息

    Returns:
        验证是否通过
    """
    target_path = build_hdf5_path(dataset_scale, target_split)

    if verbose:
        print("\n验证迁移后的数据完整性...")

    issues = []

    with h5py.File(target_path, "r") as f:
        keys_to_check = sample_keys[:5] if len(sample_keys) > 5 else sample_keys
        for key in keys_to_check:  # 只验证前几个样本
            if key not in f:
                issues.append(f"样本 '{key}' 未找到")
                continue

            group = f[key]
            missing = missing_required_fields(group.keys())  # type: ignore[union-attr]
            if missing:
                issues.append(f"样本 '{key}' 缺少字段 {missing}")

            # 检查 video_name 是否已正确更新
            if "video_name" in group:  # type: ignore[operator]
                video_name = group["video_name"][()]  # type: ignore[index]
                if isinstance(video_name, bytes):
                    video_name = video_name.decode("utf-8")

                target_lower = target_split.lower()
                if f"/{target_lower}/" not in str(video_name):
                    issues.append(
                        f"样本 '{key}' 的 video_name 未正确更新: {video_name}"
                    )

    if issues:
        print("验证发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    if verbose:
        print("验证通过! 所有检查的样本数据完整，video_name 已正确更新。")

    return True


def show_dataset_info(dataset_scale: str) -> None:
    """显示数据集的基本信息"""
    print(f"\n{'=' * 60}")
    print(f"WLASL{dataset_scale} 数据集信息")
    print(f"{'=' * 60}")

    hdf5_files, json_files = list_files_in_dataset(dataset_scale)

    print("\nHDF5 文件:")
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            sample_count = len(f.keys())
            print(f"  - {hdf5_path.name}: {sample_count} 个样本")

    print("\nJSON 文件:")
    for json_path in json_files:
        print(f"  - {json_path.name}")

    print(f"{'=' * 60}\n")


def main() -> None:
    """主函数"""
    _enable_utf8_console_output()

    # 使用硬编码的配置
    runtime_config = CONFIG()

    # 显示数据集信息
    show_dataset_info(runtime_config.DATASET_SCALE)

    # 执行数据迁移
    result = transfer_data(
        dataset_scale=runtime_config.DATASET_SCALE,
        source_split=runtime_config.SOURCE_SPLIT,
        target_split=runtime_config.TARGET_SPLIT,
        transfer_percentage=runtime_config.TRANSFER_PERCENTAGE,
        delete_from_source=runtime_config.DELETE_FROM_SOURCE,
        random_seed=runtime_config.RANDOM_SEED,
        verbose=runtime_config.VERBOSE,
    )

    # 验证迁移结果
    if result["transferred"] > 0:
        verify_transfer(
            runtime_config.DATASET_SCALE,
            runtime_config.TARGET_SPLIT,
            result["target_keys"],
            verbose=runtime_config.VERBOSE,
        )


if __name__ == "__main__":
    main()
