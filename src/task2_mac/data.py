from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    path: Path
    style_label: int | None
    fruit_label: int | None


def image_id_from_name(path: Path) -> int | None:
    match = re.search(r"\d+", path.stem)
    if not match:
        return None
    return int(match.group(0))


def find_images(image_dirs: list[str]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for raw_dir in image_dirs:
        root = Path(raw_dir).expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            image_id = image_id_from_name(path)
            if image_id is None:
                continue
            found.setdefault(image_id, path.resolve())
    return dict(sorted(found.items()))


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    match = re.search(r"-?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def read_labels(label_files: list[str]) -> dict[int, tuple[int | None, int | None]]:
    for raw_path in label_files:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            id_col = _pick_column(reader.fieldnames, ("id", "image_id", "image", "file_id", "index"))
            file_col = _pick_column(reader.fieldnames, ("file_name", "filename", "path", "image_name", "img"))
            style_col = _pick_column(reader.fieldnames, ("style_label", "style", "style_id", "style class"))
            fruit_col = _pick_column(reader.fieldnames, ("fruit_label", "fruit", "fruit_id", "class", "fruit class"))
            if id_col is None and file_col is None:
                continue

            labels: dict[int, tuple[int | None, int | None]] = {}
            for row in reader:
                image_id = _to_int(row.get(id_col)) if id_col else None
                if image_id is None and file_col:
                    image_id = image_id_from_name(Path(row[file_col]))
                if image_id is None:
                    continue
                labels[image_id] = (_to_int(row.get(style_col)), _to_int(row.get(fruit_col)))
            if labels:
                return labels
    return {}


def build_records(image_dirs: list[str], label_files: list[str]) -> list[ImageRecord]:
    images = find_images(image_dirs)
    labels = read_labels(label_files)
    records = []
    for image_id, path in images.items():
        style_label, fruit_label = labels.get(image_id, (None, None))
        records.append(ImageRecord(image_id, path, style_label, fruit_label))
    return records


def numeric_sort_records(records: list[ImageRecord]) -> list[ImageRecord]:
    return sorted(records, key=lambda item: item.image_id)


def require_labeled(records: list[ImageRecord], need_style: bool = True) -> list[ImageRecord]:
    missing = [
        record.image_id
        for record in records
        if (need_style and record.style_label is None) or record.fruit_label is None
    ]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            "Missing labels for image ids "
            f"{preview}. Provide a CSV with style_label and fruit_label."
        )
    return records
