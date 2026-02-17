import json
from pathlib import Path


def _load_label_map_payload(label_map_path: str | Path) -> dict:
    with open(label_map_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError("标签映射文件必须是 JSON 对象")

    if "id_to_label" not in payload or "label_to_id" not in payload:
        raise ValueError("标签映射文件缺少 id_to_label 或 label_to_id 字段")

    return payload


def load_label_to_id_map(label_map_path: str | Path) -> dict[str, int]:
    payload = _load_label_map_payload(label_map_path)
    id_to_label_raw = payload["id_to_label"]
    label_to_id_raw = payload["label_to_id"]

    if not isinstance(id_to_label_raw, dict) or not isinstance(label_to_id_raw, dict):
        raise ValueError("id_to_label 和 label_to_id 必须是对象")

    if id_to_label_raw and not all(str(key).isdigit() for key in id_to_label_raw.keys()):
        raise ValueError("检测到旧格式标签映射：id_to_label 键不是数字字符串")

    id_to_label: dict[int, str] = {}
    for key, label in id_to_label_raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ValueError("id_to_label 键必须是数字字符串")
        if not isinstance(label, str):
            raise ValueError("id_to_label 值必须是标签字符串")
        id_to_label[int(key)] = label

    label_to_id: dict[str, int] = {}
    for label, idx in label_to_id_raw.items():
        if not isinstance(label, str):
            raise ValueError("label_to_id 键必须是标签字符串")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError("label_to_id 值必须是整数 ID")
        label_to_id[label] = idx

    expected_label_to_id = {label: idx for idx, label in id_to_label.items()}
    if label_to_id != expected_label_to_id:
        raise ValueError("label_to_id 与 id_to_label 不一致")

    return label_to_id


def load_id_to_label_map(label_map_path: str | Path) -> dict[int, str]:
    label_to_id = load_label_to_id_map(label_map_path)
    return {idx: label for label, idx in label_to_id.items()}
