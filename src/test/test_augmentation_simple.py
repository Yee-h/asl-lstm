"""
Simple test for data augmentation
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src import config as cfg
from src.model.dataloader import CSLDataset


def main():
    print("Testing data augmentation...")

    # Load dataset with augmentation
    print("\nLoading training dataset with augmentation=True...")
    train_dataset = CSLDataset(
        cfg.PATHS.train_data_path, cfg.PATHS.label_map_path, augment=True
    )

    # Get same sample multiple times
    idx = 0
    samples = []
    print(f"\nGetting sample {idx} three times:")
    for i in range(3):
        data, label, length = train_dataset[idx]
        samples.append(data.numpy())
        print(
            f"  Attempt {i + 1}: shape={data.shape}, valid_len={length}, first_5_values={data[0, :5].numpy()}"
        )

    # Check randomness
    diff_1_2 = np.abs(samples[0] - samples[1]).mean()
    diff_2_3 = np.abs(samples[1] - samples[2]).mean()
    print(f"\nMean difference between attempt 1 and 2: {diff_1_2:.6f}")
    print(f"Mean difference between attempt 2 and 3: {diff_2_3:.6f}")

    if diff_1_2 > 1e-6 and diff_2_3 > 1e-6:
        print("\n[PASS] Augmentation is working - data is different each time")
    else:
        print("\n[FAIL] Augmentation may not be working - data is identical")

    # Test validation dataset (should NOT have augmentation)
    print("\n\nLoading validation dataset with augmentation=False...")
    val_dataset = CSLDataset(
        cfg.PATHS.val_data_path, cfg.PATHS.label_map_path, augment=False
    )

    val_samples = []
    print(f"Getting validation sample 0 twice:")
    for i in range(2):
        data, _, length = val_dataset[0]
        val_samples.append(data.numpy())
        print(f"  Attempt {i + 1}: shape={data.shape}, valid_len={length}")

    diff_val = np.abs(val_samples[0] - val_samples[1]).mean()
    print(f"\nMean difference: {diff_val:.10f}")

    if diff_val < 1e-6:
        print("[PASS] Validation dataset correctly has NO augmentation")
    else:
        print("[FAIL] Validation dataset incorrectly has augmentation")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
