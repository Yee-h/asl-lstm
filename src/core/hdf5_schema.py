from __future__ import annotations

from collections.abc import Iterable


REQUIRED_SAMPLE_FIELDS: tuple[str, ...] = (
    "data",
    "length",
    "label",
    "video_name",
    "width",
    "height",
)

OPTIONAL_SAMPLE_FIELDS: tuple[str, ...] = (
    "mask",
    "quality",
)


def missing_required_fields(sample_group: Iterable[str]) -> list[str]:
    field_set = set(sample_group)
    return [field for field in REQUIRED_SAMPLE_FIELDS if field not in field_set]


def validate_required_fields(sample_group: Iterable[str]) -> None:
    missing = missing_required_fields(sample_group)
    if missing:
        raise ValueError(f"HDF5 样本缺少必填字段: {missing}")
