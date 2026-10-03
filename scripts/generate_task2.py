#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm.auto import tqdm

from task2_mac.config import ensure_dir, load_config
from task2_mac.data import build_records, numeric_sort_records, require_labeled


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_for_sd(image: Image.Image, resolution: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = resolution / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    left = (image.width - resolution) // 2
    top = (image.height - resolution) // 2
    return image.crop((left, top, left + resolution, top + resolution))


def make_prompt(cfg: dict, target_style_id: int, fruit_label: int | None) -> str:
    style = cfg["styles"][target_style_id]
    fruit = cfg["fruits"].get(fruit_label, "fruit")
    return (
        f"a single {fruit}, {style['token']} {style['name']} style, "
        "clean centered fruit illustration, high quality, detailed, plain background"
    )


def load_pipe(cfg: dict, lora_dir: Path, device: torch.device) -> StableDiffusionImg2ImgPipeline:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg["base_model"],
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float32,
    )
    pipe.load_lora_weights(str(lora_dir))
    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_mac.yaml")
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg["generation"]
    device = best_device()
    records = numeric_sort_records(build_records(cfg["test_images"], cfg["label_files"]))
    if not records:
        print("No test images found. Falling back to train images for a local smoke test.")
        records = numeric_sort_records(build_records(cfg["train_images"], cfg["label_files"]))
    records = records[: int(gen_cfg["max_images"])]
    if not args.allow_missing_labels:
        records = require_labeled(records, need_style=True)

    output_dir = ensure_dir(gen_cfg["output_dir"])
    lora_root = Path(cfg["training"]["output_dir"])
    resolution = int(cfg["resolution"])
    target_style_ids = sorted(int(style_id) for style_id in cfg["styles"].keys())

    print(f"Generating {len(records)} source images on {device} -> {output_dir}")
    print("Loading one pipeline per style. This is slower, but avoids adapter state mistakes.")

    pipes: dict[int, StableDiffusionImg2ImgPipeline] = {}
    try:
        for style_id in target_style_ids:
            lora_dir = lora_root / f"style_{style_id}"
            if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
                raise FileNotFoundError(f"Missing LoRA for style {style_id}: {lora_dir}")
            pipes[style_id] = load_pipe(cfg, lora_dir, device)

        generator = torch.Generator(device="cpu").manual_seed(int(gen_cfg["seed"]))
        for record in tqdm(records):
            original_style = record.style_label
            if original_style is None and args.allow_missing_labels:
                styles_to_generate = target_style_ids
            else:
                styles_to_generate = [style_id for style_id in target_style_ids if style_id != original_style]

            init_image = resize_for_sd(Image.open(record.path), resolution)
            for target_style_id in styles_to_generate:
                prompt = make_prompt(cfg, target_style_id, record.fruit_label)
                image = pipes[target_style_id](
                    prompt=prompt,
                    negative_prompt=gen_cfg["negative_prompt"],
                    image=init_image,
                    strength=float(gen_cfg["strength"]),
                    guidance_scale=float(gen_cfg["guidance_scale"]),
                    num_inference_steps=int(gen_cfg["num_inference_steps"]),
                    generator=generator,
                    cross_attention_kwargs={"scale": float(gen_cfg["lora_scale"])},
                ).images[0]
                out_path = output_dir / f"{record.image_id}_generate_{target_style_id}.jpg"
                image.save(out_path, quality=95)
    finally:
        for pipe in pipes.values():
            del pipe
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"Done -> {output_dir}")


if __name__ == "__main__":
    main()
