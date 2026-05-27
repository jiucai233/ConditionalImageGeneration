import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Ensure we can import local modules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "brushnet/src"))

from util.smooth_mask import smooth_mask
from util.dilate_mask import dilate_mask
from util.crop_face import apply_crop, restore_crop, get_actor_face_crop_info
from util.color_transfer import color_transfer
from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from peft import PeftModel

#======= Configuration
base_model_path = "emilianJR/epiCRealism" 
controlnet_id = "lllyasviel/sd-controlnet-canny"
output_dir = os.path.join(root_path, "pipeline/results_multiple")
os.makedirs(output_dir, exist_ok=True)

# Test images to run (10 samples)
test_images = [
    "seed1056395.png",
    "seed1000166.png",
    "seed1000187.png",
    "seed1000020.png",
    "seed1000022.png",
    "seed1000095.png",
    "seed1000163.png",
    "seed1000211.png",
    "seed1000237.png",
    "seed1000243.png"
]

# Golden Hyperparameters
STABLE_STRENGTH = 0.60
STABLE_LORA_SCALE = 1.15
STABLE_CN_SCALE = 0.75

comparison_cases = [
    { "celeb": "고윤정", "display_name": "Go Youn Jung" },
    { "celeb": "신세경", "display_name": "Shin Se Kyung" },
    { "celeb": "홍수주", "display_name": "Hong Su Zu" },
    { "celeb": "탑", "display_name": "T.O.P" },
    { "celeb": "최시원", "display_name": "Choi Si Won" },
    { "celeb": "뷔", "display_name": "V" },
    { "celeb": "차은우", "display_name": "Cha Eun Woo" }
]

prompt_template = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
negative_prompt = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, purple patches, colorful noise, hard edges, dirty skin"

#======= Device Setup
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

def get_canny_guide(image_np):
    img = cv2.Canny(image_np, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)



def main():
    # 1. Load pipeline and components
    print(f"Loading base model {base_model_path} and Canny ControlNet on {device}...")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, text_encoder=text_encoder, vae=vae, unet=unet,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # 2. Check and Load Unified/Individual LoRA
    unified_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_female_integrated")
    unified_lora_v2_path = os.path.join(root_path, "data/ckpt/celeb_eyebrows_all_pro_v2")
    
    # We will load LoRA using V4 unified as default
    selected_lora_path = None
    if os.path.exists(os.path.join(unified_lora_path, "unet")):
        selected_lora_path = unified_lora_path
        print(f"✅ Selected Unified V4 LoRA checkpoint.")
    elif os.path.exists(os.path.join(unified_lora_v2_path, "unet")):
        selected_lora_path = unified_lora_v2_path
        print(f"✅ Selected Unified V2 LoRA checkpoint (fallback).")
        
    if selected_lora_path is not None:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(selected_lora_path, "unet"), adapter_name="celebs")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(selected_lora_path, "text_encoder"), adapter_name="celebs")
        pipe.set_adapters(["celebs"], adapter_weights=[STABLE_LORA_SCALE])
        print(f"✅ Loaded LoRA Adapter with scale {STABLE_LORA_SCALE}")
    else:
        print("⚠️ Warning: No unified LoRA checkpoint found. Proceeding without LoRA.")

    # 3. Process test images
    for idx, img_name in enumerate(test_images):
        image_path = os.path.join(root_path, "data/raw_face_data", img_name)
        print(f"\n=========================================")
        print(f"[{idx+1}/{len(test_images)}] Processing: {img_name}")
        print(f"=========================================")
        
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            print(f"Error: Could not read image at {image_path}")
            continue

        h, w = original_bgr.shape[:2]
        
        # Masking
        raw_mask = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
        raw_mask_base = dilate_mask(raw_mask, pixels=15)
        processed_mask = smooth_mask(raw_mask_base)
        
        # Cropping
        crop_info = get_actor_face_crop_info(processed_mask, original_bgr.shape, padding_ratio=4.0)
        image_512 = apply_crop(original_bgr, crop_info, target_size=512)
        mask_512_binary = apply_crop(processed_mask, crop_info, target_size=512)
        
        # Telea Eyebrow Eraser
        textured_fill = cv2.inpaint(image_512, mask_512_binary, 3, cv2.INPAINT_TELEA)
        mask_3ch_smooth = np.repeat(smooth_mask(mask_512_binary)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
        
        # Prep inputs
        image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
        pipe_mask_pil = Image.new("RGB", (512, 512), "white")
        control_image_pil = get_canny_guide(image_512)
        
        # Grid visual components
        orig_crop_rgb = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)
        cv2.putText(orig_crop_rgb, "Original Crop", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        mask_crop_rgb = cv2.cvtColor(mask_512_binary, cv2.COLOR_GRAY2RGB)
        cv2.putText(mask_crop_rgb, "Eyebrow Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        panels = [orig_crop_rgb, mask_crop_rgb]
        
        # Run celebrity styles
        for case in comparison_cases:
            celeb = case["celeb"]
            display_name = case["display_name"]
            current_prompt = prompt_template.format(celeb=celeb)
            
            print(f"  - Generating {display_name}...")
            generator = torch.Generator(device).manual_seed(42)
            
            if selected_lora_path is not None:
                pipe.unet.set_adapter("celebs")
                pipe.text_encoder.set_adapter("celebs")

            output_512_pil = pipe(
                prompt=current_prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=pipe_mask_pil,
                control_image=control_image_pil,
                controlnet_conditioning_scale=STABLE_CN_SCALE,
                strength=STABLE_STRENGTH,
                num_inference_steps=40,
                guidance_scale=6.0,
                generator=generator
            ).images[0]
            
            # Post process & color transfer
            output_512_bgr = cv2.cvtColor(np.array(output_512_pil), cv2.COLOR_RGB2BGR)
            corrected_bgr_512 = color_transfer(output_512_bgr, image_512, mask_512_binary)
            
            # Restore to full
            restored_full = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)
            
            # Blend using the original processed mask directly to prevent alignment issues
            mask_np = processed_mask.astype(np.float32) / 255.0
            if len(mask_np.shape) == 2:
                mask_np = mask_np[:, :, np.newaxis]
            
            # Dynamic kernel size based on 1.5% of max image dimension (must be odd)
            ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
            mask_blurred = cv2.GaussianBlur(mask_np, (ksize, ksize), 0)
            
            if len(mask_blurred.shape) == 2:
                mask_blurred = mask_blurred[:, :, np.newaxis]
                
            final_np = (restored_full * mask_blurred + original_bgr * (1.0 - mask_blurred)).astype(np.uint8)
            
            # Crop back for close-up side-by-side comparison
            final_cropped_bgr = apply_crop(final_np, crop_info, target_size=512)
            final_cropped_rgb = cv2.cvtColor(final_cropped_bgr, cv2.COLOR_BGR2RGB)
            cv2.putText(final_cropped_rgb, display_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            panels.append(final_cropped_rgb)
            
        # Stitch and save
        grid = np.hstack(panels)
        output_grid_path = os.path.join(output_dir, f"grid_{img_name}")
        Image.fromarray(grid).save(output_grid_path)
        print(f"🎉 Success! Grid saved to {output_grid_path}")

    print(f"\nAll 10 samples processed successfully! Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
