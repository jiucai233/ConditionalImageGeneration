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

from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from util.dilate_mask import dilate_mask
from util.smooth_mask import smooth_mask
from util.crop_face import get_crop_info, apply_crop, restore_crop
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
import transformers
if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor

#======= Configuration
base_model_path = "emilianJR/epiCRealism" 
USE_CONTROLNET = False
controlnet_id = "lllyasviel/sd-controlnet-canny"

#======= Paths and Setup
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_dir = os.path.join(root_path, "tests/data/eyebrow_tests")
os.makedirs(output_dir, exist_ok=True)

MASK_FILL_TYPE = "telea" 

# 锁定黄金参数
STABLE_STRENGTH = 0.50
STABLE_CN_SCALE = 0
STABLE_LORA_SCALE = 0.90

UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows, highly detailed, natural hair texture, masterpiece, 8k uhd"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"

comparison_cases = [
    { "celeb": "고윤정", "display_name": "Go Youn Jung" },
    { "celeb": "신세경", "display_name": "Shin Se Kyung" },
    { "celeb": "홍수주", "display_name": "Hong Su Zu" }
]

class DiffusionBackbone:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", dtype=torch.float32):
        self.model_id = model_id
        self.dtype = dtype
    def load_modules(self):
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", torch_dtype=self.dtype)
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet", torch_dtype=self.dtype)
        return text_encoder, vae, unet

def get_canny_guide(image_np):
    img = cv2.Canny(image_np, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)

#======= Device Setup
if torch.cuda.is_available():
    device = "cuda"; dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32

def load_pipeline():
    print(f"Loading models on {device}...")
    backbone = DiffusionBackbone(model_id=base_model_path, dtype=dtype)
    text_encoder, vae, unet = backbone.load_modules()
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_path, text_encoder=text_encoder, vae=vae, unet=unet,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None
    )
    
    from peft import PeftModel
    loaded_any = False
    
    # 1. Load UNIFIED LoRA model "all"
    unified_lora_path = os.path.join(root_path, "data/ckpt/celeb_eyebrows_all_pro_v2")
    if os.path.exists(os.path.join(unified_lora_path, "unet")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(unified_lora_path, "unet"), adapter_name="all_celebs")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(unified_lora_path, "text_encoder"), adapter_name="all_celebs")
        loaded_any = True
        print("✅ Loaded UNIFIED LoRA model (all celebs)")
    
    # 2. Load INDIVIDUAL LoRA models
    for celeb in ["고윤정", "신세경", "홍수주"]:
        lora_path = os.path.join(root_path, f"data/ckpt/{celeb}_eyebrows_pro_v2")
        unet_path = os.path.join(lora_path, "unet")
        te_path = os.path.join(lora_path, "text_encoder")
        
        if os.path.exists(unet_path):
            ind_adapter_name = f"ind_{celeb}"
            if not loaded_any:
                pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path, adapter_name=ind_adapter_name)
                pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_path, adapter_name=ind_adapter_name)
                loaded_any = True
            else:
                pipe.unet.load_adapter(unet_path, adapter_name=ind_adapter_name)
                pipe.text_encoder.load_adapter(te_path, adapter_name=ind_adapter_name)
            print(f"✅ Loaded INDIVIDUAL LoRA model for: {celeb}")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device != "cuda":
        pipe.to(device); pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
    else:
        pipe.enable_model_cpu_offload()
    return pipe

def run_single_image_test(pipe, image_path):
    img_basename = os.path.basename(image_path).split('.')[0]
    print(f"\n>>> Processing: {img_basename}")
    
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: return
    rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)

    # 1. Mask Generation
    raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
    raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
    raw_mask_base = smooth_mask(raw_mask_base)
    
    # Pre-calculate Mask and Crop
    crop_info = get_crop_info(raw_mask_base, original_bgr.shape, target_size=512)
    image_512 = apply_crop(original_bgr, crop_info, target_size=512)
    mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)
    
    # Telea Fill for base
    textured_fill = cv2.inpaint(image_512, mask_512_binary, 3, cv2.INPAINT_TELEA)
    mask_3ch_smooth = np.repeat(smooth_mask(mask_512_binary)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
    masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
    
    image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
    
    # For full_face_gen we use white mask
    pipe_mask_pil = Image.new("RGB", (512, 512), "white")
    control_image_pil = get_canny_guide(image_512)
    
    results = []
    
    # 2. Iterate each Celeb
    for case in comparison_cases:
        celeb = case["celeb"]
        display_name = case["display_name"]
        current_prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=celeb)
        
        ind_adapter_name = f"ind_{celeb}"
        all_adapter_name = "all_celebs"
        
        generator = torch.Generator(device).manual_seed(42)
        
        # --- Generation 1: Individual LoRA ---
        if hasattr(pipe.unet, "peft_config") and ind_adapter_name in pipe.unet.peft_config:
            pipe.enable_lora()
            pipe.set_adapters([ind_adapter_name], adapter_weights=[STABLE_LORA_SCALE])
        else:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
                
        output_ind = pipe(
            prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
            image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
            controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
            guidance_scale=6.0, strength=STABLE_STRENGTH, generator=generator
        ).images[0]
        
        # --- Generation 2: All LoRA ---
        if hasattr(pipe.unet, "peft_config") and all_adapter_name in pipe.unet.peft_config:
            pipe.enable_lora()
            pipe.set_adapters([all_adapter_name], adapter_weights=[STABLE_LORA_SCALE])
        else:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
                
        output_all = pipe(
            prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
            image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
            controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
            guidance_scale=6.0, strength=STABLE_STRENGTH, generator=generator
        ).images[0]
        
        # --- Restoration for both ---
        def process_output(output_pil, title):
            result_np_512 = np.array(output_pil)
            full_result_np = restore_crop(cv2.cvtColor(result_np_512, cv2.COLOR_RGB2BGR), crop_info, original_bgr.shape)
            mask_float = smooth_mask(raw_mask_base).astype(np.float32) / 255.0
            mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
            final_result_bgr = (original_bgr.astype(np.float32) * (1 - mask_3d) + full_result_np.astype(np.float32) * mask_3d).astype(np.uint8)
            preview = cv2.resize(cv2.cvtColor(final_result_bgr, cv2.COLOR_BGR2RGB), (512, 512))
            cv2.putText(preview, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            return preview

        preview_ind = process_output(output_ind, f"{display_name} (Ind)")
        preview_all = process_output(output_all, f"{display_name} (All)")
        
        results.append(preview_ind)
        results.append(preview_all)

    # 3. Save Grid
    preview_orig = cv2.resize(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), (512, 512))
    cv2.putText(preview_orig, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    preview_mask_vis = cv2.cvtColor(apply_crop(raw_mask_base, crop_info, 512), cv2.COLOR_GRAY2RGB)
    cv2.putText(preview_mask_vis, "Mask", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    grid = np.hstack([preview_orig, preview_mask_vis] + results)
    grid_dir = os.path.join(output_dir, "grids_compare")
    os.makedirs(grid_dir, exist_ok=True)
    grid_path = os.path.join(grid_dir, f"grid_compare_{img_basename}.png")
    Image.fromarray(grid).save(grid_path)
    print(f"Comparison Grid saved: {grid_path}")

if __name__ == "__main__":
    test_pipe = load_pipeline()
    all_imgs = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for img_file in all_imgs[:10]:
        run_single_image_test(test_pipe, os.path.join(input_images_dir, img_file))
