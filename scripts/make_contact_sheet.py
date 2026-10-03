#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from task2_mac.config import ensure_dir, load_config
from task2_mac.data import build_records, numeric_sort_records


def fit_image(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(20, 20, 20), font=ImageFont.load_default())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_mac.yaml")
    parser.add_argument("--image-size", type=int, default=220)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--out", default="release/contact_sheet.jpg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_dir = Path(cfg["generation"]["output_dir"])
    out_path = Path(args.out)
    records = numeric_sort_records(build_records(cfg["test_images"], cfg["label_files"]))
    if not records:
        records = numeric_sort_records(build_records(cfg["train_images"], cfg["label_files"]))
    records = records[: int(cfg["generation"]["max_images"])]

    style_ids = sorted(int(style_id) for style_id in cfg["styles"].keys())
    cell = args.image_size
    label_h = 34
    gap = args.gap
    cols = 3
    width = cols * cell + (cols + 1) * gap
    row_h = cell + label_h + gap
    height = max(1, len(records)) * row_h + gap
    sheet = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)

    for row_idx, record in enumerate(records):
        y = gap + row_idx * row_h
        original = fit_image(Image.open(record.path), cell)
        x = gap
        sheet.paste(original, (x, y + label_h))
        fruit = cfg["fruits"].get(record.fruit_label, "fruit")
        original_style = cfg["styles"].get(record.style_label, {}).get("name", "unknown")
        draw_label(draw, (x, y), f"{record.image_id}.jpg | {fruit} | original: {original_style}")

        target_styles = [style_id for style_id in style_ids if style_id != record.style_label]
        for col_idx, style_id in enumerate(target_styles[:2], start=1):
            x = gap + col_idx * (cell + gap)
            generated_path = gen_dir / f"{record.image_id}_generate_{style_id}.jpg"
            if generated_path.exists():
                image = fit_image(Image.open(generated_path), cell)
                sheet.paste(image, (x, y + label_h))
                label = cfg["styles"][style_id]["name"]
            else:
                draw.rectangle((x, y + label_h, x + cell, y + label_h + cell), fill=(230, 230, 230))
                label = f"missing style {style_id}"
            draw_label(draw, (x, y), label)

    ensure_dir(out_path.parent)
    sheet.save(out_path, quality=95)
    print(f"Saved contact sheet -> {out_path}")


if __name__ == "__main__":
    main()
