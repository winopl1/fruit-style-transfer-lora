#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from task2_mac.config import ensure_dir, load_config


class JsonlImageCaptionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        tokenizer: CLIPTokenizer,
        size: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.root = root
        self.tokenizer = tokenizer
        metadata_path = root / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run prepare_lora_data.py first: {metadata_path}")
        self.items = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        if not self.items:
            raise ValueError(f"No training items found in {metadata_path}")

        crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        ops = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
        ]
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transforms = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        image = Image.open(self.root / item["file_name"]).convert("RGB")
        pixel_values = self.image_transforms(image)
        tokenized = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2_mac.yaml")
    parser.add_argument("--style-id", type=int, required=True)
    parser.add_argument("--data-dir", default="work/lora_data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    style_id = args.style_id
    style_cfg = cfg["styles"][style_id]
    device = best_device()
    dtype = torch.float32

    torch.manual_seed(int(train_cfg["seed"]) + style_id)

    base_model = cfg["base_model"]
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=int(train_cfg["rank"]),
        lora_alpha=int(train_cfg["rank"]),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    text_encoder.eval()
    vae.eval()

    dataset = JsonlImageCaptionDataset(
        Path(args.data_dir) / f"style_{style_id}",
        tokenizer,
        size=int(cfg["resolution"]),
        center_crop=bool(train_cfg["center_crop"]),
        random_flip=bool(train_cfg["random_flip"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["train_batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["dataloader_num_workers"]),
        collate_fn=collate,
    )

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]), weight_decay=0.01)
    max_steps = int(train_cfg["max_train_steps"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    epochs = math.ceil(max_steps / updates_per_epoch)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(100, max_steps // 10),
        num_training_steps=max_steps,
    )

    out_dir = ensure_dir(Path(train_cfg["output_dir"]) / f"style_{style_id}")
    print(f"Training style {style_id} ({style_cfg['name']}) on {device}")
    print(f"images={len(dataset)} steps={max_steps} output={out_dir}")

    global_step = 0
    progress = tqdm(total=max_steps)
    unet.train()
    for _epoch in range(epochs):
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum
            loss.backward()

            is_update_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

                if global_step % int(train_cfg["checkpoint_every"]) == 0:
                    checkpoint_dir = ensure_dir(out_dir / f"checkpoint-{global_step}")
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=checkpoint_dir,
                        unet_lora_layers=get_peft_model_state_dict(unet),
                        safe_serialization=True,
                    )

                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    StableDiffusionPipeline.save_lora_weights(
        save_directory=out_dir,
        unet_lora_layers=get_peft_model_state_dict(unet),
        safe_serialization=True,
    )
    (out_dir / "style_info.json").write_text(
        json.dumps({"style_id": style_id, **style_cfg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved LoRA -> {out_dir}")


if __name__ == "__main__":
    main()
