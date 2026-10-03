#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from task2_mac.config import ensure_dir, load_config
from task2_mac.data import build_records, require_labeled


def caption_for(cfg: dict, style_id: int, fruit_id: int | None) -> str:
    style = cfg["styles"][style_id]
    fruit = cfg["fruits"].get(fruit_id, "fruit")
    return f"a {fruit}, {style['token']} {style['name']} style, clean centered fruit illustration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_mac.yaml")
    parser.add_argument("--out-dir", default="work/lora_data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = require_labeled(build_records(cfg["train_images"], cfg["label_files"]))
    out_root = ensure_dir(args.out_dir)

    for style_id_raw in cfg["styles"].keys():
        style_id = int(style_id_raw)
        style_dir = ensure_dir(out_root / f"style_{style_id}")
        image_dir = ensure_dir(style_dir / "images")
        metadata_path = style_dir / "metadata.jsonl"

        count = 0
        with metadata_path.open("w", encoding="utf-8") as f:
            for record in records:
                if record.style_label != style_id:
                    continue
                target_name = f"{record.image_id}{record.path.suffix.lower()}"
                target_path = image_dir / target_name
                shutil.copy2(record.path, target_path)
                f.write(
                    json.dumps(
                        {
                            "file_name": f"images/{target_name}",
                            "text": caption_for(cfg, style_id, record.fruit_label),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
        print(f"style {style_id}: {count} images -> {style_dir}")


if __name__ == "__main__":
    main()
