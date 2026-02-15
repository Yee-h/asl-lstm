import json
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Add src to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import src.config as cfg


BODY_LEFT_RIGHT_PAIRS = [
    (2, 5),
    (3, 6),
    (4, 7),
    (9, 12),
    (10, 13),
    (11, 14),
    (15, 16),
    (17, 18),
    (19, 22),
    (20, 23),
    (21, 24),
]


def load_feature_stats(stats_path: str) -> dict | None:
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_velocity(pos_data: np.ndarray, valid_len: int) -> np.ndarray:
    velocity = np.zeros_like(pos_data, dtype=np.float32)
    if valid_len > 1:
        velocity[1:valid_len] = pos_data[1:valid_len] - pos_data[: valid_len - 1]
    return velocity


def _compute_acceleration(velocity_data: np.ndarray, valid_len: int) -> np.ndarray:
    acceleration = np.zeros_like(velocity_data, dtype=np.float32)
    if valid_len > 1:
        acceleration[1:valid_len] = (
            velocity_data[1:valid_len] - velocity_data[: valid_len - 1]
        )
    return acceleration


def _ensure_feature_channels(data: np.ndarray, valid_len: int) -> np.ndarray:
    channels = data.shape[1]

    if channels == 2:
        velocity = _compute_velocity(data, valid_len)
        output = np.concatenate([data, velocity], axis=1)
        if cfg.ENABLE_ACCEL_FEATURE:
            acceleration = _compute_acceleration(velocity, valid_len)
            output = np.concatenate([output, acceleration], axis=1)
        return output.astype(np.float32)

    if channels == 4:
        output = data
        if cfg.ENABLE_ACCEL_FEATURE:
            velocity = data[:, 2:4, :]
            acceleration = _compute_acceleration(velocity, valid_len)
            output = np.concatenate([output, acceleration], axis=1)
        return output.astype(np.float32)

    if channels == 6 and cfg.ENABLE_ACCEL_FEATURE:
        return data.astype(np.float32)

    raise ValueError(f"Unsupported feature channel count: {channels}")


def _apply_standardization(data: np.ndarray, valid_len: int, stats: dict) -> np.ndarray:
    means = np.asarray(stats.get("mean", []), dtype=np.float32)
    stds = np.asarray(stats.get("std", []), dtype=np.float32)

    if means.size == 0 or stds.size == 0:
        return data
    if means.size != data.shape[1] or stds.size != data.shape[1]:
        return data

    standardized = data.copy()
    eps = getattr(cfg, "STANDARDIZE_EPS", 1e-6)
    standardized[:valid_len] = (standardized[:valid_len] - means[None, :, None]) / (
        stds[None, :, None] + eps
    )
    return standardized


def swap_left_right_keypoints(data: np.ndarray) -> np.ndarray:
    """Swap left/right landmarks while preserving channel values."""
    swapped = data.copy()

    left_hand = swapped[:, :, 25:46].copy()
    right_hand = swapped[:, :, 46:67].copy()
    swapped[:, :, 25:46] = right_hand
    swapped[:, :, 46:67] = left_hand

    for left_idx, right_idx in BODY_LEFT_RIGHT_PAIRS:
        left_values = swapped[:, :, left_idx].copy()
        swapped[:, :, left_idx] = swapped[:, :, right_idx]
        swapped[:, :, right_idx] = left_values

    return swapped


def preprocess_keypoints(
    data: np.ndarray,
    max_frames: int = cfg.MAX_FRAMES,
    valid_len: int | None = None,
    stats: dict | None = None,
    standardize: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Preprocess keypoint sequence into flattened tensor with fixed length.

    Supports both:
    - (T, 2, V): x, y
    - (T, 4, V): x, y, dx, dy
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (T, C, V), got {data.shape}")

    data = np.asarray(data, dtype=np.float32)
    total_len = data.shape[0]
    valid_len = total_len if valid_len is None else int(valid_len)
    valid_len = max(0, min(valid_len, total_len, max_frames))

    data = _ensure_feature_channels(data, valid_len)

    if data.shape[0] > max_frames:
        data = data[:max_frames]
        valid_len = min(valid_len, max_frames)
    elif data.shape[0] < max_frames:
        pad = np.zeros(
            (max_frames - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32
        )
        data = np.concatenate([data, pad], axis=0)

    if standardize and stats is not None and valid_len > 0:
        data = _apply_standardization(data, valid_len, stats)

    data_tensor = torch.tensor(data.reshape(data.shape[0], -1), dtype=torch.float32)
    return data_tensor, valid_len


class CSLDataset(Dataset):
    """
    手语识别数据集类，负责加载 HDF5 格式特征和标签。
    """

    def __init__(
        self, hdf5_path, label_map_path, max_frames=cfg.MAX_FRAMES, augment=False
    ):
        self.max_frames = max_frames
        self.augment = augment
        self.feature_stats = None

        if cfg.ENABLE_STANDARDIZE:
            self.feature_stats = load_feature_stats(cfg.FEATURE_STATS_PATH)

        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        if "id_to_label" in label_map:
            self.label_to_id = label_map["id_to_label"]
        elif "label_to_id" in label_map:
            sample_key = next(iter(label_map["label_to_id"].keys()))
            if sample_key.isdigit():
                self.label_to_id = {
                    v: int(k) for k, v in label_map["label_to_id"].items()
                }
            else:
                self.label_to_id = label_map["label_to_id"]
        else:
            self.label_to_id = label_map

        self.data_cache = []
        skipped_low_quality = 0

        print(f"正在加载 {hdf5_path} 数据到内存...")
        with h5py.File(hdf5_path, "r") as f:
            for key in list(f.keys()):
                item = f[key]
                if not isinstance(item, h5py.Group):
                    continue

                if "data" not in item.keys() or "label" not in item.keys():
                    continue

                feature = np.array(item["data"], dtype=np.float32)

                if "length" in item.keys():
                    length = int(np.asarray(item["length"])[()])
                else:
                    length = min(feature.shape[0], self.max_frames)

                mask = (
                    np.array(item["mask"], dtype=np.uint8)
                    if "mask" in item.keys()
                    else None
                )

                quality = (
                    float(np.asarray(item["quality"])[()])
                    if "quality" in item.keys()
                    else None
                )

                if quality is not None and quality < cfg.MIN_VALID_RATIO_PER_SAMPLE:
                    skipped_low_quality += 1
                    continue

                label_raw = np.asarray(item["label"])[()]
                label_str = (
                    label_raw.decode("utf-8")
                    if isinstance(label_raw, bytes)
                    else str(label_raw)
                )

                if label_str in self.label_to_id:
                    label_id = int(self.label_to_id[label_str])
                    self.data_cache.append(
                        {
                            "feature": feature,
                            "label_id": label_id,
                            "length": length,
                            "mask": mask,
                            "quality": quality,
                        }
                    )

        print(f"成功加载 {len(self.data_cache)} 条样本。")
        if skipped_low_quality > 0:
            print(f"因低质量阈值过滤样本: {skipped_low_quality}")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        sample = self.data_cache[idx]
        data = sample["feature"].copy()
        label_id = sample["label_id"]
        valid_len = min(int(sample["length"]), self.max_frames, data.shape[0])

        if self.augment and valid_len > 0:
            data = self._apply_augmentation(data, valid_len)

        data_tensor, valid_len = preprocess_keypoints(
            data,
            self.max_frames,
            valid_len=valid_len,
            stats=self.feature_stats,
            standardize=cfg.ENABLE_STANDARDIZE and self.feature_stats is not None,
        )

        return data_tensor, label_id, valid_len

    def _random_time_warp(self, data: np.ndarray, valid_len: int) -> np.ndarray:
        if valid_len < 3:
            return data
        if np.random.random() >= cfg.AUG_TIME_WARP_PROB:
            return data

        scale = np.random.uniform(cfg.AUG_TIME_WARP_MIN, cfg.AUG_TIME_WARP_MAX)
        src_t = np.arange(valid_len, dtype=np.float32)
        dst_t = np.clip(
            np.linspace(0, valid_len - 1, valid_len, dtype=np.float32) * scale,
            0.0,
            valid_len - 1,
        )

        warped = data.copy()
        for c in range(data.shape[1]):
            for v in range(data.shape[2]):
                warped[:valid_len, c, v] = np.interp(
                    dst_t, src_t, data[:valid_len, c, v]
                )
        return warped

    def _random_frame_dropout(self, data: np.ndarray, valid_len: int) -> np.ndarray:
        if valid_len < 3:
            return data
        if np.random.random() >= cfg.AUG_FRAME_DROPOUT_PROB:
            return data

        max_drop = int(valid_len * cfg.AUG_FRAME_DROPOUT_MAX_RATIO)
        if max_drop <= 0:
            return data

        num_drop = np.random.randint(1, max_drop + 1)
        drop_indices = np.random.choice(
            np.arange(1, valid_len), size=num_drop, replace=False
        )

        dropped = data.copy()
        for i in drop_indices:
            dropped[i] = dropped[i - 1]
        return dropped

    def _apply_augmentation(self, data: np.ndarray, valid_len: int) -> np.ndarray:
        """应用几何与时序增强（适配 4 通道: x, y, dx, dy）。"""
        transformed = data.copy()
        work = transformed[:valid_len]

        work = self._random_time_warp(work, valid_len)
        work = self._random_frame_dropout(work, valid_len)

        angle = np.random.uniform(-cfg.AUG_ROTATION_RANGE, cfg.AUG_ROTATION_RANGE)
        theta = np.radians(angle)
        scale = np.random.uniform(cfg.AUG_SCALE_MIN, cfg.AUG_SCALE_MAX)
        tx = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)
        ty = np.random.uniform(-cfg.AUG_TRANSLATE, cfg.AUG_TRANSLATE)

        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )

        pos = work[:, 0:2, :].transpose(0, 2, 1).reshape(-1, 2)
        pos = (pos @ rotation_matrix.T) * scale + np.array([tx, ty], dtype=np.float32)
        work[:, 0:2, :] = pos.reshape(valid_len, -1, 2).transpose(0, 2, 1)

        if work.shape[1] >= 4:
            vel = work[:, 2:4, :].transpose(0, 2, 1).reshape(-1, 2)
            vel = (vel @ rotation_matrix.T) * scale
            work[:, 2:4, :] = vel.reshape(valid_len, -1, 2).transpose(0, 2, 1)

        if work.shape[1] >= 6:
            acc = work[:, 4:6, :].transpose(0, 2, 1).reshape(-1, 2)
            acc = (acc @ rotation_matrix.T) * scale
            work[:, 4:6, :] = acc.reshape(valid_len, -1, 2).transpose(0, 2, 1)

        noise = np.random.normal(
            loc=0.0, scale=cfg.AUG_NOISE_STD, size=work.shape
        ).astype(np.float32)
        work = work + noise

        if np.random.random() < cfg.AUG_HFLIP_PROB:
            if cfg.AUG_HFLIP_ZERO_CENTERED:
                work[:, 0, :] = -work[:, 0, :]
            else:
                work[:, 0, :] = 1.0 - work[:, 0, :]

            if work.shape[1] >= 4:
                work[:, 2, :] = -work[:, 2, :]
            if work.shape[1] >= 6:
                work[:, 4, :] = -work[:, 4, :]

            if cfg.AUG_HFLIP_SWAP_LR:
                work = swap_left_right_keypoints(work)

        transformed[:valid_len] = work
        return transformed


def get_dataloaders():
    """创建并返回训练、验证和测试数据加载器。"""
    train_dataset = CSLDataset(cfg.TRAIN_DATA_PATH, cfg.LABEL_MAP_PATH, augment=True)
    val_dataset = CSLDataset(cfg.VAL_DATA_PATH, cfg.LABEL_MAP_PATH, augment=False)
    test_dataset = CSLDataset(cfg.TEST_DATA_PATH, cfg.LABEL_MAP_PATH, augment=False)

    pin_memory = torch.cuda.is_available() and cfg.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_dataloaders()
    for batch_idx, (data, label, lengths) in enumerate(train_loader):
        print(
            f"Batch {batch_idx}: Data Shape {data.shape}, Label Shape {label.shape}, Lengths Shape {lengths.shape}"
        )
        if batch_idx == 0:
            break
