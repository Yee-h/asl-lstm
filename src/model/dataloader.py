import json
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.config as cfg
from src.core.labels import load_label_to_id_map


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


def compute_sample_weights(label_ids: list[int], power: float = 1.0) -> np.ndarray:
    """Compute inverse-frequency sample weights for class-balanced sampling."""
    if not label_ids:
        return np.asarray([], dtype=np.float32)

    labels = np.asarray(label_ids, dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_count_map = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    weights = np.asarray(
        [1.0 / (class_count_map[int(label)] ** float(power)) for label in labels],
        dtype=np.float32,
    )

    mean_weight = float(weights.mean()) if weights.size > 0 else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights


def build_weighted_sampler(
    label_ids: list[int], power: float = 1.0
) -> WeightedRandomSampler | None:
    weights = compute_sample_weights(label_ids, power=power)
    if weights.size == 0:
        return None

    return WeightedRandomSampler(
        weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )


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
        acceleration[1:valid_len] = velocity_data[1:valid_len] - velocity_data[: valid_len - 1]
    return acceleration


def _ensure_feature_channels(data: np.ndarray, valid_len: int) -> np.ndarray:
    channels = data.shape[1]

    if channels == 2:
        velocity = _compute_velocity(data, valid_len)
        output = np.concatenate([data, velocity], axis=1)
        if cfg.SEQUENCE.enable_accel_feature:
            acceleration = _compute_acceleration(velocity, valid_len)
            output = np.concatenate([output, acceleration], axis=1)
        return output.astype(np.float32)

    if channels == 4:
        output = data
        if cfg.SEQUENCE.enable_accel_feature:
            velocity = data[:, 2:4, :]
            acceleration = _compute_acceleration(velocity, valid_len)
            output = np.concatenate([output, acceleration], axis=1)
        return output.astype(np.float32)

    if channels == 6 and cfg.SEQUENCE.enable_accel_feature:
        return data.astype(np.float32)

    raise ValueError(f"Unsupported feature channel count: {channels}")


def _apply_standardization(
    data: np.ndarray,
    valid_len: int,
    stats: dict,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    means = np.asarray(stats.get("mean", []), dtype=np.float32)
    stds = np.asarray(stats.get("std", []), dtype=np.float32)

    if means.size == 0 or stds.size == 0:
        return data
    if means.size != data.shape[1] or stds.size != data.shape[1]:
        return data

    standardized = data.copy()
    if valid_len <= 0:
        return standardized

    eps = cfg.PREPROCESS.standardize_eps

    normalized = (standardized[:valid_len] - means[None, :, None]) / (stds[None, :, None] + eps)

    if mask is None:
        standardized[:valid_len] = normalized
        return standardized

    valid_mask = np.asarray(mask[:valid_len], dtype=bool)
    if valid_mask.ndim != 2 or valid_mask.shape[1] != data.shape[2]:
        standardized[:valid_len] = normalized
        return standardized

    standardized[:valid_len] = np.where(valid_mask[:, None, :], normalized, 0.0)
    return standardized


def swap_left_right_keypoints(data: np.ndarray) -> np.ndarray:
    """Swap left/right landmarks while preserving channel values."""
    swapped = data.copy()
    landmark_count = swapped.shape[2]

    if landmark_count >= 67:
        left_hand = swapped[:, :, 25:46].copy()
        right_hand = swapped[:, :, 46:67].copy()
        swapped[:, :, 25:46] = right_hand
        swapped[:, :, 46:67] = left_hand

    for left_idx, right_idx in BODY_LEFT_RIGHT_PAIRS:
        if left_idx >= landmark_count or right_idx >= landmark_count:
            continue
        left_values = swapped[:, :, left_idx].copy()
        swapped[:, :, left_idx] = swapped[:, :, right_idx]
        swapped[:, :, right_idx] = left_values

    return swapped


def preprocess_keypoints(
    data: np.ndarray,
    max_frames: int = cfg.SEQUENCE.max_frames,
    valid_len: int | None = None,
    stats: dict | None = None,
    standardize: bool = False,
    mask: np.ndarray | None = None,
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

    mask_array = None
    if mask is not None:
        candidate_mask = np.asarray(mask, dtype=np.uint8)
        if candidate_mask.ndim == 2 and candidate_mask.shape[1] == data.shape[2]:
            mask_array = candidate_mask

    data = _ensure_feature_channels(data, valid_len)

    if data.shape[0] > max_frames:
        data = data[:max_frames]
        valid_len = min(valid_len, max_frames)
    elif data.shape[0] < max_frames:
        pad = np.zeros((max_frames - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    if mask_array is not None:
        if mask_array.shape[0] > max_frames:
            mask_array = mask_array[:max_frames]
        elif mask_array.shape[0] < max_frames:
            mask_pad = np.zeros(
                (max_frames - mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8
            )
            mask_array = np.concatenate([mask_array, mask_pad], axis=0)

        if valid_len > 0:
            data[:valid_len] = np.where(
                mask_array[:valid_len, None, :].astype(bool),
                data[:valid_len],
                0.0,
            )

    if standardize and stats is not None and valid_len > 0:
        data = _apply_standardization(data, valid_len, stats, mask=mask_array)

    if mask_array is not None and valid_len > 0:
        data[:valid_len] = np.where(
            mask_array[:valid_len, None, :].astype(bool),
            data[:valid_len],
            0.0,
        )

    data_tensor = torch.tensor(data.reshape(data.shape[0], -1), dtype=torch.float32)
    return data_tensor, valid_len


class CSLDataset(Dataset):
    """
    手语识别数据集类，负责加载 HDF5 格式特征和标签。
    """

    def __init__(
        self,
        hdf5_path,
        label_map_path,
        max_frames=cfg.SEQUENCE.max_frames,
        augment=False,
    ):
        self.max_frames = max_frames
        self.augment = augment
        self.feature_stats = None

        if cfg.PREPROCESS.enable_standardize:
            self.feature_stats = load_feature_stats(cfg.PREPROCESS.feature_stats_path)

        self.label_to_id = load_label_to_id_map(label_map_path)

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

                mask = np.array(item["mask"], dtype=np.uint8) if "mask" in item.keys() else None

                quality = (
                    float(np.asarray(item["quality"])[()]) if "quality" in item.keys() else None
                )

                if quality is not None and quality < cfg.PREPROCESS.min_valid_ratio_per_sample:
                    skipped_low_quality += 1
                    continue

                label_raw = np.asarray(item["label"])[()]
                label_str = (
                    label_raw.decode("utf-8") if isinstance(label_raw, bytes) else str(label_raw)
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
        mask = sample.get("mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=np.uint8).copy()
            if mask.ndim != 2 or mask.shape[1] != data.shape[2]:
                mask = None

        if self.augment and valid_len > 0:
            data, mask = self._apply_augmentation(data, valid_len, mask=mask)

        data_tensor, valid_len = preprocess_keypoints(
            data,
            self.max_frames,
            valid_len=valid_len,
            stats=self.feature_stats,
            standardize=cfg.PREPROCESS.enable_standardize and self.feature_stats is not None,
            mask=mask,
        )

        return data_tensor, label_id, valid_len

    def _random_time_warp(
        self,
        data: np.ndarray,
        valid_len: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if valid_len < 3:
            return data, mask
        if np.random.random() >= cfg.AUGMENTATION.time_warp_prob:
            return data, mask

        scale = np.random.uniform(cfg.AUGMENTATION.time_warp_min, cfg.AUGMENTATION.time_warp_max)
        src_t = np.arange(valid_len, dtype=np.float32)
        dst_t = np.clip(
            np.linspace(0, valid_len - 1, valid_len, dtype=np.float32) * scale,
            0.0,
            valid_len - 1,
        )

        warped = data.copy()
        for c in range(data.shape[1]):
            for v in range(data.shape[2]):
                warped[:valid_len, c, v] = np.interp(dst_t, src_t, data[:valid_len, c, v])

        if mask is None:
            return warped, None

        warped_mask = np.asarray(mask, dtype=np.uint8).copy()
        if warped_mask.ndim != 2 or warped_mask.shape[1] != data.shape[2]:
            return warped, None

        warped_mask = warped_mask[:valid_len]
        for v in range(warped_mask.shape[1]):
            warped_mask[:, v] = (
                np.interp(dst_t, src_t, warped_mask[:, v].astype(np.float32)) >= 0.5
            ).astype(np.uint8)

        warped[:valid_len] = np.where(warped_mask[:, None, :].astype(bool), warped[:valid_len], 0.0)
        return warped, warped_mask

    def _random_frame_dropout(
        self,
        data: np.ndarray,
        valid_len: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if valid_len < 3:
            return data, mask
        if np.random.random() >= cfg.AUGMENTATION.frame_dropout_prob:
            return data, mask

        max_drop = int(valid_len * cfg.AUGMENTATION.frame_dropout_max_ratio)
        if max_drop <= 0:
            return data, mask

        num_drop = np.random.randint(1, max_drop + 1)
        drop_indices = np.random.choice(np.arange(1, valid_len), size=num_drop, replace=False)

        dropped = data.copy()
        dropped_mask = None
        if mask is not None:
            dropped_mask = np.asarray(mask, dtype=np.uint8).copy()
            if dropped_mask.ndim != 2 or dropped_mask.shape[1] != data.shape[2]:
                dropped_mask = None

        for i in drop_indices:
            dropped[i] = dropped[i - 1]
            if dropped_mask is not None:
                dropped_mask[i] = dropped_mask[i - 1]

        return dropped, dropped_mask

    def _apply_augmentation(
        self,
        data: np.ndarray,
        valid_len: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """应用几何与时序增强（适配 4 通道: x, y, dx, dy）。"""
        transformed = data.copy()
        work = transformed[:valid_len]

        if mask is None:
            work_mask = ((work[:, 0, :] != 0) | (work[:, 1, :] != 0)).astype(np.uint8)
            output_mask = np.zeros((data.shape[0], data.shape[2]), dtype=np.uint8)
        else:
            output_mask = np.asarray(mask, dtype=np.uint8).copy()
            if output_mask.ndim != 2 or output_mask.shape[1] != data.shape[2]:
                output_mask = np.zeros((data.shape[0], data.shape[2]), dtype=np.uint8)
            if output_mask.shape[0] < data.shape[0]:
                pad_rows = data.shape[0] - output_mask.shape[0]
                output_mask = np.concatenate(
                    [
                        output_mask,
                        np.zeros((pad_rows, output_mask.shape[1]), dtype=np.uint8),
                    ],
                    axis=0,
                )
            output_mask = output_mask[: data.shape[0]]
            work_mask = output_mask[:valid_len].copy()

        work, work_mask = self._random_time_warp(work, valid_len, mask=work_mask)
        work, work_mask = self._random_frame_dropout(work, valid_len, mask=work_mask)

        if work_mask is None:
            work_mask = ((work[:, 0, :] != 0) | (work[:, 1, :] != 0)).astype(np.uint8)

        valid_mask = work_mask[:, None, :].astype(bool)

        angle = np.random.uniform(-cfg.AUGMENTATION.rotation_range, cfg.AUGMENTATION.rotation_range)
        theta = np.radians(angle)
        scale = np.random.uniform(cfg.AUGMENTATION.scale_min, cfg.AUGMENTATION.scale_max)
        tx = np.random.uniform(-cfg.AUGMENTATION.translate, cfg.AUGMENTATION.translate)
        ty = np.random.uniform(-cfg.AUGMENTATION.translate, cfg.AUGMENTATION.translate)

        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )

        pos = work[:, 0:2, :].transpose(0, 2, 1).reshape(-1, 2)
        pos = (pos @ rotation_matrix.T) * scale + np.array([tx, ty], dtype=np.float32)
        work[:, 0:2, :] = np.where(
            valid_mask,
            pos.reshape(valid_len, -1, 2).transpose(0, 2, 1),
            work[:, 0:2, :],
        )

        if work.shape[1] >= 4:
            vel = work[:, 2:4, :].transpose(0, 2, 1).reshape(-1, 2)
            vel = (vel @ rotation_matrix.T) * scale
            work[:, 2:4, :] = np.where(
                valid_mask,
                vel.reshape(valid_len, -1, 2).transpose(0, 2, 1),
                work[:, 2:4, :],
            )

        if work.shape[1] >= 6:
            acc = work[:, 4:6, :].transpose(0, 2, 1).reshape(-1, 2)
            acc = (acc @ rotation_matrix.T) * scale
            work[:, 4:6, :] = np.where(
                valid_mask,
                acc.reshape(valid_len, -1, 2).transpose(0, 2, 1),
                work[:, 4:6, :],
            )

        noise = np.random.normal(loc=0.0, scale=cfg.AUGMENTATION.noise_std, size=work.shape).astype(
            np.float32
        )
        work = work + noise * valid_mask.astype(np.float32)

        if np.random.random() < cfg.AUGMENTATION.hflip_prob:
            if cfg.AUGMENTATION.hflip_zero_centered:
                work[:, 0, :] = np.where(valid_mask[:, 0, :], -work[:, 0, :], 0.0)
            else:
                work[:, 0, :] = np.where(valid_mask[:, 0, :], 1.0 - work[:, 0, :], 0.0)

            if work.shape[1] >= 4:
                work[:, 2, :] = np.where(valid_mask[:, 0, :], -work[:, 2, :], 0.0)
            if work.shape[1] >= 6:
                work[:, 4, :] = np.where(valid_mask[:, 0, :], -work[:, 4, :], 0.0)

            if cfg.AUGMENTATION.hflip_swap_lr:
                work = swap_left_right_keypoints(work)

        work[:, 0:2, :] = np.where(valid_mask, work[:, 0:2, :], 0.0)
        work = np.where(valid_mask, work, 0.0)
        transformed[:valid_len] = work
        output_mask[:valid_len] = work_mask
        return transformed, output_mask


def get_dataloaders(
    train_augment: bool = True,
    use_weighted_sampler: bool | None = None,
):
    """创建并返回训练、验证和测试数据加载器。"""
    train_dataset = CSLDataset(
        cfg.PATHS.train_data_path,
        cfg.PATHS.label_map_path,
        augment=bool(train_augment),
    )
    val_dataset = CSLDataset(cfg.PATHS.val_data_path, cfg.PATHS.label_map_path, augment=False)
    test_dataset = CSLDataset(cfg.PATHS.test_data_path, cfg.PATHS.label_map_path, augment=False)

    pin_memory = torch.cuda.is_available() and cfg.TRAINING.device == "cuda"

    weighted_sampler_enabled = (
        cfg.TRAINING.use_weighted_sampler
        if use_weighted_sampler is None
        else bool(use_weighted_sampler)
    )

    train_sampler = None
    if weighted_sampler_enabled:
        train_labels = [sample["label_id"] for sample in train_dataset.data_cache]
        train_sampler = build_weighted_sampler(train_labels, power=cfg.TRAINING.sampler_power)

    train_shuffle = train_sampler is None
    num_workers = max(0, int(cfg.TRAINING.dataloader_num_workers))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAINING.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAINING.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAINING.batch_size,
        shuffle=False,
        num_workers=num_workers,
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
