import os
import sys

# Ensure we can import local modules and submodules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "brushnet/src"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler

from util.crop_face import get_zoom_crop_info, apply_crop, get_actor_face_crop_info, get_random_actor_face_crop_info
from util.dilate_mask import dilate_mask
from util.invert_mask import invert_mask
from util.augment import augment_image_and_mask, get_random_zoom_crop_info

# ==========================================
# ⚙️ Professional Configuration
# ==========================================
BASE_MODEL_ID = "emilianJR/epiCRealism" 
BASE_DATA_PATH = os.path.join(root_path, "data", "preprocessed")
MODEL_SAVE_PATH = os.path.join(root_path, "data", "ckpt")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

TRAIN_TARGET = 'all' # 'all' or '신세경', '고윤정', '홍수주'
LORA_NAME = f"{TRAIN_TARGET}_eyebrows_pro_v2" if TRAIN_TARGET != 'all' else "celeb_eyebrows_all_pro_v2"

EPOCHS = 20
MAX_TRAIN_STEPS = 30000  # Set to 30000 steps as requested for overnight training
SAVE_STEPS = 1500  # Save a checkpoint every 1500 steps to avoid excessive disk space (approx. 20 checkpoints total)
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
TEXT_ENCODER_LR = 5e-5
LR_WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
OFFSET_NOISE = 0.1
MIN_SNR_GAMMA = 5.0

# ==========================================
# 📦 Enhanced Dataset
# ==========================================

class EyebrowDatasetPro(Dataset):
    def __init__(self, base_path, target, size=512, augment=True):
        import glob
        from pathlib import Path

        self.base_path = base_path
        self.celebs = ['신세경', '고윤정', '홍수주','탑','최시원','뷔','차은우'] if target == 'all' else [target]
        self.size = size
        self.augment = augment
        self.data_list = []

        celeb_data = {celeb: [] for celeb in self.celebs}

        for celeb in self.celebs:
            mask_base_dir = os.path.join(base_path, f"{celeb}_mask")
            extracted_dir = os.path.join(mask_base_dir, "extracted")
            tight_mask_dir = os.path.join(mask_base_dir, "tight")
            padded_2px_dir = os.path.join(mask_base_dir, "padded_2px")
            raw_actor_dir = os.path.abspath(os.path.join(base_path, "..", "actor_raw_data", celeb))

            if not os.path.exists(extracted_dir): continue

            # Build a mapping from computed stem-name back to raw image file
            raw_paths = []
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            if os.path.exists(raw_actor_dir):
                for ext in image_extensions:
                    raw_paths.extend(glob.glob(os.path.join(raw_actor_dir, f"*{ext}")))
                    raw_paths.extend(glob.glob(os.path.join(raw_actor_dir, f"*{ext.upper()}")))

            stem_counts = {}
            for p in raw_paths:
                p_path = Path(p)
                stem_counts[p_path.stem] = stem_counts.get(p_path.stem, 0) + 1

            stem_to_raw = {}
            for p in raw_paths:
                p_path = Path(p)
                if stem_counts[p_path.stem] == 1:
                    computed_stem = p_path.stem
                else:
                    computed_stem = f"{p_path.stem}_{p_path.suffix.lower().lstrip('.')}"
                stem_to_raw[computed_stem] = p

            for fname in os.listdir(extracted_dir):
                if not fname.endswith('_tight_white_bg.png'): continue
                
                base_name = fname.replace('_tight_white_bg.png', '')
                e_p = os.path.join(extracted_dir, fname)
                m_p = os.path.join(tight_mask_dir, f"{base_name}_tight_mask.png")
                
                # Dilated mask (for edge blending learning)
                m_p_2px = os.path.join(padded_2px_dir, f"{base_name}_padded_2px_mask.png")
                if not os.path.exists(m_p_2px):
                    m_p_2px = m_p

                raw_img_path = stem_to_raw.get(base_name)

                # Pair and combine data aspects:
                # 1. Pure Eyebrow (White BG + Tight Mask)
                if os.path.exists(e_p) and os.path.exists(m_p):
                    celeb_data[celeb].append({
                        "img_path": e_p,
                        "mask_path": m_p,
                        "celeb": celeb,
                        "is_white_bg": True
                    })

                    # If matched raw face image exists:
                    if raw_img_path and os.path.exists(raw_img_path):
                        # 2. Face Context (Raw Face + Tight Mask)
                        celeb_data[celeb].append({
                            "img_path": raw_img_path,
                            "mask_path": m_p,
                            "celeb": celeb,
                            "is_white_bg": False
                        })
                        # 3. Edge Blending (Raw Face + Padded 2px Mask)
                        celeb_data[celeb].append({
                            "img_path": raw_img_path,
                            "mask_path": m_p_2px,
                            "celeb": celeb,
                            "is_white_bg": False
                        })

        # Dataset Balancing
        if len(self.celebs) > 1:
            max_samples = max(len(items) for items in celeb_data.values())
            print(f"📊 Dataset Balancing: Target max samples per celebrity = {max_samples}")
            for celeb, items in celeb_data.items():
                curr_len = len(items)
                if curr_len == 0:
                    continue
                if curr_len < max_samples:
                    repeated_items = (items * (max_samples // curr_len + 1))[:max_samples]
                    print(f"   - {celeb}: Upsampled from {curr_len} to {max_samples} samples")
                    self.data_list.extend(repeated_items)
                else:
                    print(f"   - {celeb}: Already has {curr_len} samples (maximum)")
                    self.data_list.extend(items)
        else:
            for celeb, items in celeb_data.items():
                self.data_list.extend(items)
                print(f"📊 Dataset Single Celeb: {celeb} has {len(items)} samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = cv2.cvtColor(cv2.imread(item["img_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        is_white_bg = item["is_white_bg"]

        if self.augment:
            img, mask = augment_image_and_mask(img, mask)
            # Use tight crop for white_bg close-ups, wide crop for raw face context
            if is_white_bg:
                crop_info = get_random_zoom_crop_info(mask, img.shape, min_padding=1.8, max_padding=2.6, max_shift=15)
            else:
                crop_info = get_random_actor_face_crop_info(mask, img.shape, min_padding=3.6, max_padding=4.4, max_shift=15)
        else:
            if is_white_bg:
                crop_info = get_zoom_crop_info(mask, img.shape, padding_ratio=2.2)
            else:
                crop_info = get_actor_face_crop_info(mask, img.shape, padding_ratio=4.0)

        cropped_img = apply_crop(img, crop_info, self.size)
        cropped_mask = apply_crop(mask, crop_info, self.size)

        # 学习边缘过渡 (根据是否是白底，动态调整 Mask 膨胀大小与训练 Prompt)
        if is_white_bg:
            final_mask = dilate_mask(cropped_mask, pixels=4)
            prompt = f"a close-up photo of {item['celeb']} style eyebrows isolated on a pure white background"
        else:
            final_mask = dilate_mask(cropped_mask, pixels=15)
            prompt = f"a photo of {item['celeb']} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"

        return {
            "pixel_values": torch.from_numpy(cropped_img).permute(2, 0, 1).float() / 127.5 - 1.0,
            "masks": torch.from_numpy(final_mask).unsqueeze(0).float() / 255.0,
            "prompt": prompt
        }

# ==========================================
# 🔥 Professional Training Loop
# ==========================================

def train_pro():
    import datetime
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Generate run name with start timestamp and step count (no 'v' suffix)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_name = f"{TRAIN_TARGET}_eyebrows_{start_time}_{MAX_TRAIN_STEPS}" if TRAIN_TARGET != 'all' else f"celeb_eyebrows_all_{start_time}_{MAX_TRAIN_STEPS}"
    
    print(f"🚀 Pro Training Started: {lora_name} on {device}")

    # 1. Models & Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet").to(device)
    
    vae.requires_grad_(False)
    
    # 2. LoRA Config (Rank 128, 同时微调 UNet 和 Text Encoder)
    unet_lora_config = LoraConfig(r=128, lora_alpha=128, target_modules=["to_k", "to_q", "to_v", "to_out.0"], bias="none")
    unet = get_peft_model(unet, unet_lora_config)
    
    text_lora_config = LoraConfig(r=128, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
    text_encoder = get_peft_model(text_encoder, text_lora_config)
    
    # 3. Data
    dataset = EyebrowDatasetPro(BASE_DATA_PATH, TRAIN_TARGET)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_update_steps_per_epoch = len(train_dataloader)
    if MAX_TRAIN_STEPS is not None:
        max_train_steps = MAX_TRAIN_STEPS
        EPOCHS = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        max_train_steps = EPOCHS * num_update_steps_per_epoch

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {"params": unet.parameters(), "lr": LEARNING_RATE},
        {"params": text_encoder.parameters(), "lr": TEXT_ENCODER_LR}
    ])
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=LR_WARMUP_STEPS, num_training_steps=max_train_steps)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

    # 5. Training Loop
    unet.train()
    text_encoder.train()
    global_step = 0
    
    for epoch in range(EPOCHS):
        steps_in_epoch = min(len(train_dataloader), max_train_steps - global_step)
        if steps_in_epoch <= 0:
            break
        progress_bar = tqdm(total=steps_in_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(train_dataloader):
            if global_step >= max_train_steps:
                break
                
            pixel_values = batch["pixel_values"].to(device)
            masks = batch["masks"].to(device)
            
            # Encode inputs
            latents = vae.encode(pixel_values).latent_dist.sample().detach() * 0.18215
            mask_latent = F.interpolate(masks, size=(latents.shape[2], latents.shape[3]))
            
            # Noise & Offset Noise
            noise = torch.randn_like(latents)
            if OFFSET_NOISE:
                noise += OFFSET_NOISE * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=device)
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text embedding
            inputs = tokenizer(batch["prompt"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            # Predict & Loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # ✨ Min-SNR Gamma Logic ✨
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
            sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            snr = (sigmas ** -2).index_select(0, timesteps)
            mse_loss_weights = torch.stack([snr, MIN_SNR_GAMMA * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

            loss_elementwise = (model_pred.float() - noise.float()) ** 2
            loss_elementwise = loss_elementwise * mask_latent # 关键：只看掩码区域
            
            loss = (loss_elementwise.mean([1, 2, 3]) * mse_loss_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                "step": f"{global_step}/{max_train_steps}"
            })
            
            # Save checkpoint every SAVE_STEPS steps
            if global_step > 0 and global_step % SAVE_STEPS == 0:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, lora_name, f"checkpoint-{global_step}")
                unet.save_pretrained(os.path.join(checkpoint_path, "unet"))
                text_encoder.save_pretrained(os.path.join(checkpoint_path, "text_encoder"))
                print(f"\n💾 Saved checkpoint to: {checkpoint_path}")

    # 6. Save final model
    final_save_path = os.path.join(MODEL_SAVE_PATH, lora_name)
    unet.save_pretrained(os.path.join(final_save_path, "unet"))
    text_encoder.save_pretrained(os.path.join(final_save_path, "text_encoder"))
    print(f"✅ Pro LoRA Saved: {lora_name} at {final_save_path}")

if __name__ == "__main__":
    train_pro()
