import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class EyebrowDataset(Dataset):
    def __init__(self, root_dir, resolution=1024):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        
        # Support common image formats
        self.image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            self.image_paths.extend(list(self.root_dir.glob(ext)))
        
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = (image.height, image.width)
        
        pixel_values = self.transforms(image)
        
        # In CenterCrop, we assume crop is at the center
        # For SDXL micro-conditioning, we provide (crop_top, crop_left)
        # Simplified: assume center crop for now
        crop_top = (original_size[0] - self.resolution) // 2 if original_size[0] > self.resolution else 0
        crop_left = (original_size[1] - self.resolution) // 2 if original_size[1] > self.resolution else 0
        
        # Look for matching .txt file for caption
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            caption = "a photo of eyebrows"
            
        return {
            "pixel_values": pixel_values, 
            "caption": caption,
            "original_size": original_size,
            "crop_coords": (crop_top, crop_left)
        }

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids

# Adapted from StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt, text_encoders, tokenizers):
    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds

def parse_args():
    parser = argparse.ArgumentParser(description="Professional LoRA training script for SDXL Eyebrows.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eyebrow-lora-sdxl")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    
    return parser.parse_args()

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    if args.seed is not None:
        set_seed(args.seed)

    # 1. Load Tokenizers & Models
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None, subfolder="text_encoder_2")
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # 2. Freeze Models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. Add LoRA
    from diffusers.models.attention_processor import LoRAAttnProcessor
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = torch.nn.ModuleList(unet.attn_processors.values())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers not available")

    # 4. Optimizer & Scheduler
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("bitsandbytes not found")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(lora_layers.parameters(), lr=args.learning_rate)

    # 5. Dataset
    train_dataset = EyebrowDataset(args.train_data_dir, resolution=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # 6. Prepare for training
    lora_layers, optimizer, train_dataloader = accelerator.prepare(lora_layers, optimizer, train_dataloader)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # 7. Training Loop
    global_step = 0
    progress_bar = tqdm(range(args.num_train_epochs * len(train_dataloader)), desc="Training LoRA", disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_layers):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                
                # VAE encoding
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Forward Diffusion
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text Embeddings (SDXL style)
                # Note: For efficiency, we just use the first caption in the batch for simplicity in this loop
                prompt_embeds, pooled_prompt_embeds = encode_prompt(batch["caption"][0], [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two])
                prompt_embeds = prompt_embeds.repeat(bsz, 1, 1).to(weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(bsz, 1).to(weight_dtype)
                
                # SDXL Micro-conditioning (Proper time_ids)
                # batch["original_size"] is (H, W), batch["crop_coords"] is (Y, X)
                # target_size is (resolution, resolution)
                orig_h, orig_w = batch["original_size"]
                crop_y, crop_x = batch["crop_coords"]
                
                # Construct time_ids for EACH item in batch
                time_ids_list = []
                for i in range(bsz):
                    t_ids = [orig_h[i].item(), orig_w[i].item(), crop_y[i].item(), crop_x[i].item(), args.resolution, args.resolution]
                    time_ids_list.append(torch.tensor(t_ids))
                
                add_time_ids = torch.stack(time_ids_list).to(accelerator.device, dtype=weight_dtype)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                
                # Model Prediction
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                # Loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            progress_bar.set_postfix({"loss": loss.detach().item()})

    # 8. Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_attn_procs(args.output_dir)
        print(f"Improved LoRA saved to {args.output_dir}")

if __name__ == "__main__":
    main()
