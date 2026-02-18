import os
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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


def _swap_left_right_keypoints_torch(data: torch.Tensor) -> torch.Tensor:
    """Swap left/right landmarks for tensor shaped (B, T, C, V)."""
    swapped = data.clone()
    landmark_count = swapped.shape[-1]

    if landmark_count >= 67:
        left_hand = swapped[..., 25:46].clone()
        right_hand = swapped[..., 46:67].clone()
        swapped[..., 25:46] = right_hand
        swapped[..., 46:67] = left_hand

    for left_idx, right_idx in BODY_LEFT_RIGHT_PAIRS:
        if left_idx >= landmark_count or right_idx >= landmark_count:
            continue
        left_values = swapped[..., left_idx].clone()
        swapped[..., left_idx] = swapped[..., right_idx]
        swapped[..., right_idx] = left_values

    return swapped


def build_hflip_tta_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Build horizontally flipped TTA batch for flattened sequence inputs."""
    if inputs.ndim != 3:
        raise ValueError(f"Expected input shape (B, T, F), got {tuple(inputs.shape)}")

    channels = cfg.SEQUENCE.landmark_dim
    landmarks = cfg.SEQUENCE.num_landmarks
    expected_feature_dim = channels * landmarks
    if inputs.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"Input feature dim mismatch: got {inputs.shape[-1]}, expected {expected_feature_dim}"
        )

    batch, seq_len, _ = inputs.shape
    reshaped = inputs.view(batch, seq_len, channels, landmarks).clone()

    if cfg.AUGMENTATION.hflip_zero_centered:
        reshaped[:, :, 0, :] = -reshaped[:, :, 0, :]
    else:
        reshaped[:, :, 0, :] = 1.0 - reshaped[:, :, 0, :]

    if channels >= 4:
        reshaped[:, :, 2, :] = -reshaped[:, :, 2, :]
    if channels >= 6:
        reshaped[:, :, 4, :] = -reshaped[:, :, 4, :]

    if cfg.AUGMENTATION.hflip_swap_lr:
        reshaped = _swap_left_right_keypoints_torch(reshaped)

    return reshaped.reshape(batch, seq_len, expected_feature_dim)
