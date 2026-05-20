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
from util.erode_mask import erode_mask
from util.smooth_mask import smooth_mask
from util.resize_and_pad import resize_and_pad
from util.restore_from_pad import restore_from_pad
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel

#======= Configuration
MODEL_SOURCE = "online" # 选择 "online" (在线 HuggingFace repo) 或 "local" (本地权重路径)

# 在线模型 ID
ONLINE_BASE_MODEL = "emilianJR/epiCRealism" 

# 本地模型路径 (支持 diffusers 文件夹结构，或单个 .safetensors / .ckpt 文件)
LOCAL_BASE_MODEL_PATH = "/Users/jiucai/my_codes/models/realisticVisionV51_v51VAE.safetensors"

base_model_path = ONLINE_BASE_MODEL if MODEL_SOURCE == "online" else LOCAL_BASE_MODEL_PATH
brushnet_path = os.path.join(root_path, "data/ckpt/brushnetx") # 🚨 注意: 确保这里面是 SD1.5 对应的 segmentation_mask 权重

class DiffusionBackbone:
    """
    Stable Diffusion의 3대 핵심 요소(Text Encoder, VAE, UNet)를
    관리하고 불러오는 클래스입니다.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", dtype=torch.float32):
        self.model_id = model_id
        self.dtype = dtype

    def load_modules(self):
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", torch_dtype=self.dtype)
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet", torch_dtype=self.dtype)
        return text_encoder, vae, unet

#======= Helper Functions
# Util functions are now imported from the util module.

# Use a default test image
image_path = os.path.join(root_path, "data/raw_face_data/seed1056395.png")
output_dir = os.path.join(root_path, "tests/data/eyebrow_tests")
os.makedirs(output_dir, exist_ok=True)

MODE = "lora"

# How to fill the area where the eyebrows were removed:
# "black": Complete erasure (standard BrushNet)
# "telea": Realistic erasure using surrounding skin (BEST for "making them disappear")
# "gray": Neutral gray fill
MASK_FILL_TYPE = "telea" 

celeb_style = "고윤정"
UNIFIED_PROMPT = f"a photo of {celeb_style} style eyebrows, highly detailed, natural hair texture, masterpiece, 8k uhd"
# Unified base prompt for all cases
# UNIFIED_PROMPT = "RAW photo, a close up portrait of a face, highly detailed, natural skin texture, dark hair, photorealistic, highly detailed skin, natural eyebrows, detailed hair strokes, 8k uhd, dslr, macro photography"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic"

# Test cases for "mask" mode
mask_test_cases = [
    { "name": "original_shape", "dilate_y": 0, "dilate_x": 0, "extra_prompt": "" },
    { "name": "dense", "dilate_y": 0, "dilate_x": 0, "extra_prompt": "very dense, thick, bushy, bold dark eyebrows, highly concentrated hair" },
    { "name": "sparse", "dilate_y": 0, "dilate_x": 0, "extra_prompt": "sparse, faint, light-colored, thin delicate hair, barely visible eyebrows" },
    { "name": "wide_vertical", "dilate_y": 15, "dilate_x": 0, "extra_prompt": "" },
    { "name": "narrow_vertical", "erode_y": 8, "dilate_x": 0, "extra_prompt": "" },
    { "name": "long_horizontal", "dilate_x": 20, "dilate_y": 2, "extra_prompt": "long extending eyebrows, reaching the temples" },
    { "name": "short_horizontal", "dilate_x": 10, "dilate_y": 5, "extra_prompt": "very short cropped eyebrows, mostly bare skin on the brow ridge" }
]

# Test cases for "prompt" mode
prompt_test_cases = [
    { "name": "long", "extra_prompt": "extremely long eyebrows, extending far horizontally towards the temples", "negative_prompt": "short eyebrows" },
    { "name": "short", "extra_prompt": "very short eyebrows, horizontally clipped and short", "negative_prompt": "long eyebrows" },
    { "name": "narrow", "extra_prompt": "very narrow thin eyebrows, thin line of hair", "negative_prompt": "thick eyebrows, wide eyebrows" },
    { "name": "wide", "extra_prompt": "very wide thick eyebrows, thick bushy eyebrows", "negative_prompt": "thin eyebrows, narrow eyebrows" },
    { "name": "dense", "extra_prompt": "extremely dense and dark eyebrows, very thick hair", "negative_prompt": "sparse eyebrows, light eyebrows" },
    { "name": "sparse", "extra_prompt": "very sparse and faint eyebrows, light delicate hair", "negative_prompt": "dense eyebrows, dark eyebrows" }
]

# Test cases for "lora" mode
lora_test_cases = [
    { "name": "without_lora", "use_lora": False },
    { "name": "with_lora", "use_lora": True }
]

# Test cases for "default" mode
default_test_cases = [
    { "name": "control_group", "use_lora": False, "extra_prompt": "" }
]

if MODE == "lora":
    test_cases = lora_test_cases
elif MODE == "prompt":
    test_cases = prompt_test_cases
elif MODE == "default":
    test_cases = default_test_cases
else:
    test_cases = mask_test_cases

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

def run_eyebrow_test():
    #======= 1. Load Models
    print(f"Loading models on {device} (Mode: {MODE}, Source: {MODEL_SOURCE})...")
    try:
        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype)
        
        if base_model_path.endswith(".safetensors") or base_model_path.endswith(".ckpt"):
            print(f"Loading base model from single local file: {base_model_path}")
            from diffusers import StableDiffusionPipeline
            temp_pipe = StableDiffusionPipeline.from_single_file(base_model_path, torch_dtype=dtype, local_files_only=True)
            pipe = StableDiffusionBrushNetPipeline(
                vae=temp_pipe.vae,
                text_encoder=temp_pipe.text_encoder,
                tokenizer=temp_pipe.tokenizer,
                unet=temp_pipe.unet,
                scheduler=temp_pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                brushnet=brushnet
            )
        else:
            print(f"Loading base model from standard format: {base_model_path}")
            backbone = DiffusionBackbone(model_id=base_model_path, dtype=dtype)
            text_encoder, vae, unet = backbone.load_modules()
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model_path, 
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                brushnet=brushnet, 
                torch_dtype=dtype, 
                low_cpu_mem_usage=True, 
                safety_checker=None
            )
        
        lora_path = os.path.join(root_path, "data/ckpt/고윤정_eyebrows_pro_v2")
        print(f"Loading LoRA from {lora_path}...")
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(lora_path, "unet"), adapter_name="gyj_brow")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(lora_path, "text_encoder"), adapter_name="gyj_brow")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device != "cuda":
        pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    else:
        pipe.enable_model_cpu_offload()

    #======= 2. Prepare Base Image
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"Error: Could not find input image at {image_path}")
        return

    h, w = original_bgr.shape[:2]
    rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    print("Generating base raw mask for eyebrows...")
    raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])

    if MODE in ["prompt", "lora", "default"]:
        dilation_amount = 25 if MODE in ["lora", "default"] else 20
        raw_mask_base = dilate_mask(raw_mask_base, pixels=dilation_amount)
        raw_mask_base = smooth_mask(raw_mask_base)

    #======= 3. Iterate through test cases
    results = []
    
    for case in test_cases:
        name = case["name"]
        print(f"\n--- Testing mode {MODE}: {name} ---")

        # Process mask
        if MODE == "mask":
            processed_mask = raw_mask_base.copy()
            if "dilate_y" in case and case["dilate_y"] > 0:
                kernel_y = np.ones((case["dilate_y"] * 2 + 1, 1), np.uint8)
                processed_mask = cv2.dilate(processed_mask, kernel_y, iterations=1)
            if "erode_y" in case and case["erode_y"] > 0:
                kernel_y = np.ones((case["erode_y"] * 2 + 1, 1), np.uint8)
                processed_mask = cv2.erode(processed_mask, kernel_y, iterations=1)
                
            if "dilate_x" in case and case["dilate_x"] > 0:
                kernel_x = np.ones((1, case["dilate_x"] * 2 + 1), np.uint8)
                processed_mask = cv2.dilate(processed_mask, kernel_x, iterations=1)
            if "erode_x" in case and case["erode_x"] > 0:
                kernel_x = np.ones((1, case["erode_x"] * 2 + 1), np.uint8)
                processed_mask = cv2.erode(processed_mask, kernel_x, iterations=1)

            if "dilate" in case:
                processed_mask = dilate_mask(processed_mask, pixels=case["dilate"])
            elif "erode" in case:
                processed_mask = erode_mask(processed_mask, pixels=case["erode"])
            
            processed_mask = smooth_mask(processed_mask)
        else:
            processed_mask = raw_mask_base

        image_512, crop_info, _ = resize_and_pad(rgb_image, target_size=512)
        mask_512, _, _ = resize_and_pad(processed_mask, target_size=512)
        mask_512_binary = (mask_512 > 127).astype(np.uint8) * 255

        mask_3ch_512 = mask_512_binary[:, :, np.newaxis] / 255.0
        
        # Apply MASK_FILL_TYPE
        if MASK_FILL_TYPE == "telea":
            # 1. 传统 Telea 填充底色
            inpainted_bgr = cv2.inpaint(cv2.cvtColor(image_512, cv2.COLOR_RGB2BGR), mask_512_binary, 3, cv2.INPAINT_TELEA)
            base_fill = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # 2. 注入模拟毛孔的高斯噪声 (Scale 10-15 即可，不要太脏)
            noise = np.random.normal(loc=0, scale=12, size=image_512.shape).astype(np.float32)
            textured_fill = np.clip(base_fill + noise, 0, 255)
            
            # 3. 这里的掩码最好也加上一点模糊，避免接缝处一刀切
            mask_3ch_smooth = cv2.GaussianBlur(mask_512_binary, (15, 15), 0)[:, :, np.newaxis] / 255.0
            
            masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
        elif MASK_FILL_TYPE == "gray":
            gray = np.full((512, 512, 3), 128, dtype=np.uint8)
            masked_image_512 = (image_512 * (1.0 - mask_3ch_512) + gray * mask_3ch_512).astype(np.uint8)
        else: # "black"
            masked_image_512 = (image_512 * (1.0 - mask_3ch_512)).astype(np.uint8)

        image_pil = Image.fromarray(masked_image_512).convert("RGB")
        mask_pil = Image.fromarray(mask_512_binary).convert("RGB")
        
        generator = torch.Generator(device).manual_seed(42)
        
        full_prompt = f"{UNIFIED_PROMPT}, {case.get('extra_prompt', '')}"
        neg_prompt = f"{UNIFIED_NEGATIVE_PROMPT}, {case.get('negative_prompt', '')}" if "negative_prompt" in case else UNIFIED_NEGATIVE_PROMPT
        
        if "use_lora" in case:
            if case["use_lora"]:
                pipe.set_adapters(["gyj_brow"], adapter_weights=[0.8])
            else:
                pipe.disable_lora()
        
        # 🚨 [核心修复 3]：注入 guidance_scale 逼迫模型刻画毛发，稍微放宽 conditioning_scale 给底模留融合空间
        output = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=image_pil,           
            mask=mask_pil,             
            num_inference_steps=50,
            guidance_scale=8.5,        # <-- 强化细节纹理
            generator=generator,
            brushnet_conditioning_scale=0.85 # <-- 防止边缘生硬
        ).images[0]

        # Blending and Extraction
        result_np_512 = np.array(output)
        
        mask_512_norm = mask_512_binary[:, :, np.newaxis].astype(np.float32) / 255.0
        white_bg = np.full((512, 512, 3), 255, dtype=np.float32)
        extracted_np_512 = (result_np_512 * mask_512_norm + white_bg * (1.0 - mask_512_norm)).astype(np.uint8)
        
        result_np = restore_from_pad(result_np_512, crop_info, (h, w))
        
        mask_np = np.array(processed_mask).astype(np.float32) / 255.0
        if len(mask_np.shape) == 2:
            mask_np = mask_np[:, :, np.newaxis]
        
        mask_blurred = cv2.GaussianBlur(mask_np, (21, 21), 0)
        if len(mask_blurred.shape) == 2:
            mask_blurred = mask_blurred[:, :, np.newaxis]

        final_np = (result_np * mask_blurred + rgb_image * (1.0 - mask_blurred)).astype(np.uint8)
        
        # Save individual results
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_dir, f"eyebrow_{MODE}_{name}.png")
        cv2.imwrite(save_path, final_bgr)
        
        extracted_bgr = cv2.cvtColor(extracted_np_512, cv2.COLOR_RGB2BGR)
        extracted_path = os.path.join(output_dir, f"eyebrow_{MODE}_{name}_extracted.png")
        cv2.imwrite(extracted_path, extracted_bgr)
        print(f"Saved: {save_path} and {extracted_path}")
        
        # Prepare for comparison grid
        preview_final = cv2.resize(final_np, (512, 512))
        cv2.putText(preview_final, name.capitalize(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        preview_extracted = extracted_np_512.copy()
        cv2.putText(preview_extracted, "Extracted", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 0), 2)
        
        preview_mask_case = cv2.cvtColor(mask_512_binary, cv2.COLOR_GRAY2RGB)
        cv2.putText(preview_mask_case, f"{name} Mask", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        results.append({
            "mask": preview_mask_case,
            "extracted": preview_extracted,
            "result": preview_final
        })

    #======= 4. Create Comparison Grid
    print("\nCreating summary comparison grid...")
    
    mask_strip = np.hstack([r["mask"] for r in results])
    extracted_strip = np.hstack([r["extracted"] for r in results])
    result_strip = np.hstack([r["result"] for r in results])
    
    preview_orig = cv2.resize(rgb_image, (512, 512))
    cv2.putText(preview_orig, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    white_spacer = np.full((512, 512, 3), 255, dtype=np.uint8)
    cv2.putText(white_spacer, "Background", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 0), 2)
    
    orig_column = np.vstack([preview_orig, white_spacer, preview_orig])
    
    main_grid = np.vstack([mask_strip, extracted_strip, result_strip])
    final_grid = np.hstack([orig_column, main_grid])
    
    grid_image = Image.fromarray(final_grid)
    grid_path = os.path.join(output_dir, f"eyebrow_comparison_grid_{MODE}.png")
    grid_image.save(grid_path)
    print(f"Summary grid saved to: {grid_path}")

if __name__ == "__main__":
    run_eyebrow_test()