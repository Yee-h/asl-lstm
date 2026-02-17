from pathlib import Path

import src.config as cfg


def project_root() -> Path:
    return Path(cfg.PATHS.project_root)


def processed_data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return data_root
    return project_root() / "dataset" / "processed"


def dataset_subset_dir(dataset_scale: int | str, data_root: Path | None = None) -> Path:
    return processed_data_root(data_root) / f"WLASL{dataset_scale}"


def hdf5_file_path(
    dataset_scale: int | str,
    split: str,
    data_root: Path | None = None,
) -> Path:
    filename = f"WLASL{dataset_scale}_135-{split}.hdf5"
    return dataset_subset_dir(dataset_scale, data_root=data_root) / filename


def label_map_file_path(
    dataset_scale: int | str,
    data_root: Path | None = None,
) -> Path:
    filename = f"wlasl_{dataset_scale}_maplabels.json"
    return dataset_subset_dir(dataset_scale, data_root=data_root) / filename
