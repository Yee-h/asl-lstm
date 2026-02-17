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


def _coerce_legacy_index(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError("旧格式标签映射中的 ID 必须是整数")


def _convert_legacy_payload(
    id_to_label_raw: dict,
    label_to_id_raw: dict,
) -> dict[str, dict]:
    label_to_id: dict[str, int] = {}

    for key, label in label_to_id_raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ValueError("旧格式 label_to_id 的键必须是数字字符串")
        if not isinstance(label, str):
            raise ValueError("旧格式 label_to_id 的值必须是标签字符串")
        label_to_id[label] = int(key)

    for label, idx in id_to_label_raw.items():
        if not isinstance(label, str):
            raise ValueError("旧格式 id_to_label 的键必须是标签字符串")
        idx_int = _coerce_legacy_index(idx)
        existing = label_to_id.get(label)
        if existing is not None and existing != idx_int:
            raise ValueError("旧格式 id_to_label 与 label_to_id 不一致")
        label_to_id[label] = idx_int

    id_to_label_tmp: dict[int, str] = {}
    for label, idx in label_to_id.items():
        existing = id_to_label_tmp.get(idx)
        if existing is not None and existing != label:
            raise ValueError("旧格式中存在重复 ID 对应多个标签")
        id_to_label_tmp[idx] = label

    id_to_label = {
        str(idx): label for idx, label in sorted(id_to_label_tmp.items(), key=lambda item: item[0])
    }
    return {
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
    }


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


def migrate_legacy_label_map_file(label_map_path: str | Path) -> bool:
    payload = _load_label_map_payload(label_map_path)
    id_to_label_raw = payload["id_to_label"]
    label_to_id_raw = payload["label_to_id"]

    if not isinstance(id_to_label_raw, dict) or not isinstance(label_to_id_raw, dict):
        raise ValueError("id_to_label 和 label_to_id 必须是对象")

    if id_to_label_raw and all(str(key).isdigit() for key in id_to_label_raw.keys()):
        return False

    converted = _convert_legacy_payload(id_to_label_raw, label_to_id_raw)
    label_map_path = Path(label_map_path)
    backup_path = label_map_path.with_suffix(label_map_path.suffix + ".legacy.bak")
    if not backup_path.exists():
        with open(backup_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=4, ensure_ascii=False)

    with open(label_map_path, "w", encoding="utf-8") as file_obj:
        json.dump(converted, file_obj, indent=4, ensure_ascii=False)

    return True


def load_label_to_id_map_compat(label_map_path: str | Path) -> dict[str, int]:
    try:
        return load_label_to_id_map(label_map_path)
    except ValueError as exc:
        if "旧格式" not in str(exc):
            raise

    migrated = migrate_legacy_label_map_file(label_map_path)
    if not migrated:
        raise ValueError("标签映射加载失败，且未触发旧格式迁移")
    return load_label_to_id_map(label_map_path)


def load_id_to_label_map_compat(label_map_path: str | Path) -> dict[int, str]:
    label_to_id = load_label_to_id_map_compat(label_map_path)
    return {idx: label for label, idx in label_to_id.items()}
