from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_dir"] = str(path.parent.resolve())
    return cfg


def project_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = Path(item).expanduser()
        if path.exists():
            return path.resolve()
    return None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
