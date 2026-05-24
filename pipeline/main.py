import os
import sys
import datetime
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
from util.crop_face import get_crop_info, apply_crop, restore_crop, get_actor_face_crop_info
from util.color_transfer import color_transfer
from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

#======= Configuration
# Base realistic checkpoint (automatically cached from HuggingFace)
base_model_path = "emilianJR/epiCRealism" 

# Choose target celebrity: '고윤정' (Go Youn Jung), '신세경' (Shin Se Kyung), or '홍수주' (Hong Su Zu)
TARGET_CELEB = "고윤정"

# Inputs
image_path = os.path.join(root_path, "data/bro.jpg")
input_name = os.path.splitext(os.path.basename(image_path))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(root_path, f"pipeline/outputs/result_{input_name}_{TARGET_CELEB}_{timestamp}.png")

# Inpainting Hyperparameters
STABLE_STRENGTH = 0.60
STABLE_LORA_SCALE = 1.15
STABLE_CN_SCALE = 0.75
controlnet_id = "lllyasviel/sd-controlnet-canny"

# Prompt construction
prompt = f"a photo of {TARGET_CELEB} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
negative_prompt = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, purple patches, colorful noise, hard edges, dirty skin"

#======= Device Setup (Mac MPS / CUDA / CPU)
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



def run_pipeline():
    #======= 1. Load Models & Pipelines
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

    #======= 2. Load Unified/Individual LoRA
    # Check for the unified LoRA model first (V4), fallback to V2 or individual if not trained yet
    unified_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_all_pro_v4")
    unified_lora_v2_path = os.path.join(root_path, "data/ckpt/celeb_eyebrows_all_pro_v2")
    individual_lora_path = os.path.join(root_path, f"data/ckpt/{TARGET_CELEB}_eyebrows_pro_v2")
    
    selected_lora_path = None
    if os.path.exists(os.path.join(unified_lora_path, "unet")):
        selected_lora_path = unified_lora_path
        print(f"✅ Selected Unified V4 LoRA checkpoint.")
    elif os.path.exists(os.path.join(unified_lora_v2_path, "unet")):
        selected_lora_path = unified_lora_v2_path
        print(f"✅ Selected Unified V2 LoRA checkpoint (fallback).")
    elif os.path.exists(os.path.join(individual_lora_path, "unet")):
        selected_lora_path = individual_lora_path
        print(f"✅ Selected Individual {TARGET_CELEB} LoRA checkpoint (fallback).")
        
    if selected_lora_path is not None:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(selected_lora_path, "unet"), adapter_name="celebs")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(selected_lora_path, "text_encoder"), adapter_name="celebs")
        pipe.set_adapters(["celebs"], adapter_weights=[STABLE_LORA_SCALE])
        print(f"✅ Loaded LoRA Adapter from {selected_lora_path} with scale {STABLE_LORA_SCALE}")
    else:
        print("⚠️ Warning: No pre-trained LoRA adapter found. Proceeding without LoRA.")

    #======= 3. Prepare Image & Generate Mask
    print(f"Generating eyebrows mask for: {TARGET_CELEB}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"Error: Could not find input image at {image_path}")
        return

    # Generate eyebrow mask using BiSeNet (dilated and smoothed for clean blending boundaries)
    raw_mask = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
    raw_mask_base = dilate_mask(raw_mask, pixels=15)
    processed_mask = smooth_mask(raw_mask_base)

    # Crop target face area locally for stable generation scale (resolves scale mismatch)
    h, w = original_bgr.shape[:2]
    crop_info = get_actor_face_crop_info(processed_mask, original_bgr.shape, padding_ratio=4.0)
    
    # 512x512 Local Crops
    image_512 = apply_crop(original_bgr, crop_info, target_size=512)
    mask_512_binary = apply_crop(processed_mask, crop_info, target_size=512)
    
    # Telea Fill to completely erase eyebrows from the input image to avoid concealer patches
    textured_fill = cv2.inpaint(image_512, mask_512_binary, 3, cv2.INPAINT_TELEA)
    mask_3ch_smooth = np.repeat(smooth_mask(mask_512_binary)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
    masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
    
    # Preprocess Image & Mask for Pipeline
    image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
    pipe_mask_pil = Image.new("RGB", (512, 512), "white")
    
    # Canny edge guide from the masked crop (allows correct style/angle shape generation)
    control_image_pil = get_canny_guide(masked_image_512)

    #======= 4. Inference
    print(f"Generating eyebrows via StableDiffusionControlNetInpaintPipeline (Strength: {STABLE_STRENGTH})...")
    generator = torch.Generator(device).manual_seed(42)
    
    # Enable LoRA scaling safely
    pipe.unet.set_adapter("celebs")
    pipe.text_encoder.set_adapter("celebs")

    output_512_pil = pipe(
        prompt=prompt,
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

    #======= 5. Integrate & Restore Back to Full Image
    output_512_bgr = cv2.cvtColor(np.array(output_512_pil), cv2.COLOR_RGB2BGR)
    
    # Color transfer to match the tone of the generated patch with the original face
    corrected_bgr_512 = color_transfer(output_512_bgr, image_512, mask_512_binary)
    
    # Restore the local 512 patch back to original resolution
    restored_full = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)

    # Soft alpha-blending using the original processed mask directly to prevent alignment issues
    mask_np = processed_mask.astype(np.float32) / 255.0
    if len(mask_np.shape) == 2:
        mask_np = mask_np[:, :, np.newaxis]
    
    # Dynamic kernel size based on 1.5% of max image dimension (must be odd)
    ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
    mask_blurred = cv2.GaussianBlur(mask_np, (ksize, ksize), 0)
    
    if len(mask_blurred.shape) == 2:
        mask_blurred = mask_blurred[:, :, np.newaxis]

    final_np = (restored_full * mask_blurred + original_bgr * (1.0 - mask_blurred)).astype(np.uint8)

    #======= 6. Create Preview & Save
    scale = 0.5
    new_size = (int(w * scale), int(h * scale))
    preview_orig = cv2.resize(original_bgr, new_size)
    preview_mask = cv2.resize(cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR), new_size)
    preview_res = cv2.resize(final_np, new_size)

    comparison = np.hstack((preview_orig, preview_mask, preview_res))
    cv2.imwrite(output_path, comparison)
    print(f"🎉 Success! Result saved to {output_path}")

if __name__ == "__main__":
    run_pipeline()
