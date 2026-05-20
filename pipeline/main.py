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
from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler

#======= Configuration
# Model paths (Ensure you have downloaded these)
base_model_path = "runwayml/stable-diffusion-v1-5" 
brushnet_path = os.path.join(root_path, "data/ckpt/brushnetx")

# Inputs
image_path = os.path.join(root_path, "data/raw_face_data/seed1056395.png")      # Path to your original image
output_path = os.path.join(root_path, "pipeline/result_face.png")
prompt = '''RAW photo, a close up portrait of a face, highly detailed, natural skin texture, 
            realistic lighting, 8k uhd, dslr, soft lighting, high quality, film grain, 
            beautiful thick natural eyebrows'''
negative_prompt = "low quality, distorted, blurry, messy, ugly"

# Face parts to refine: lips, nose, left_eyebrow, right_eyebrow, eyebrows, eyes
target_parts = ["eyebrows"]

#======= Device Setup (Mac MPS / CUDA / CPU)
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32 # MPS often has precision issues with float16
else:
    device = "cpu"
    dtype = torch.float32

def run_pipeline():
    #======= 1. Load Models
    print(f"Loading BrushNet on {device}...")
    try:
        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            base_model_path, brushnet=brushnet, torch_dtype=dtype, low_cpu_mem_usage=False, safety_checker=None
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure checkpoints exist in 'data/ckpt/'")
        return

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    #======= 2. Data Prepare & Generate Mask
    print(f"Processing image and generating mask for: {target_parts}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"Error: Could not find input image at {image_path}")
        return

    # Generate face part mask using BiSeNet
    raw_mask = generate_bisenet_face_parts_mask(original_bgr, parts=target_parts)
    
    # Post-process mask for better blending
    processed_mask = smooth_mask(raw_mask)
    processed_mask = dilate_mask(processed_mask, pixels=5)

    # Prepare for BrushNet
    # BrushNet input is the original image + a binary mask
    rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # SD 1.5 works best with 512x512; 1024x1024 causes Mac OOM and distorted generations
    h, w = rgb_image.shape[:2]
    image_512 = cv2.resize(rgb_image, (512, 512))
    mask_512 = cv2.resize(processed_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Force binarization (0 or 255) for smooth edge masks, as BrushNet requires strict 0/1 Masks.
    # Otherwise, semi-transparent edges will cause "ghosting" artifacts during BrushNet feature extraction.
    mask_512_binary = (mask_512 > 127).astype(np.uint8) * 255

    # The region to be inpainted (facial features) MUST be blacked out in the original image
    mask_3ch_512 = mask_512_binary[:, :, np.newaxis] / 255.0
    masked_image_512 = (image_512 * (1.0 - mask_3ch_512)).astype(np.uint8)

    image_pil = Image.fromarray(masked_image_512).convert("RGB")
    # Mask polarity: White (255) indicates regions to modify, Black (0) indicates regions to preserve
    mask_pil = Image.fromarray(mask_512_binary).convert("RGB")

    #======= 3. BrushNet Inference
    print("Generating new facial features (512x512)...")
    generator = torch.Generator(device).manual_seed(42)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask=mask_pil,
        num_inference_steps=50,
        generator=generator,
        brushnet_conditioning_scale=1.0
    ).images[0]

    #======= 4. Integrate Result (Blending)
    print("Blending output with original image...")
    # Resize the generated 512x512 image back to the original image dimensions
    result_np_512 = np.array(output)
    result_np = cv2.resize(result_np_512, (w, h))
    
    mask_np = np.array(processed_mask).astype(np.float32) / 255.0
    if len(mask_np.shape) == 2:
        mask_np = mask_np[:, :, np.newaxis]

    # Use Gaussian blur on mask for seamless transition
    mask_blurred = cv2.GaussianBlur(mask_np, (21, 21), 0)
    if len(mask_blurred.shape) == 2:
        mask_blurred = mask_blurred[:, :, np.newaxis]

    # Linear interpolation between original and result based on blurred mask
    final_np = (result_np * mask_blurred + rgb_image * (1.0 - mask_blurred)).astype(np.uint8)
    
    #======= 5. Create Comparison & Save Result
    print("Creating comparison image...")
    # Convert mask to 3-channel RGB for stacking
    mask_3ch = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2RGB)
    
    # Scale down to avoid creating a massive image (e.g. 3072x1024)
    scale = 0.5
    new_size = (int(w * scale), int(h * scale))
    preview_original = cv2.resize(rgb_image, new_size)
    preview_mask = cv2.resize(mask_3ch, new_size)
    preview_raw_gen = cv2.resize(result_np_512, new_size) # Raw generation from diffusion model
    preview_final = cv2.resize(final_np, new_size)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview_original, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(preview_mask, "Mask", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(preview_raw_gen, "Raw Gen", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(preview_final, "Result", (10, 30), font, 1, (0, 255, 0), 2)

    comparison = np.hstack((preview_original, preview_mask, preview_raw_gen, preview_final))
    comparison_image = Image.fromarray(comparison)

    comparison_image.save(output_path)
    print(f"Done! Comparison result saved to {output_path}")

if __name__ == "__main__":
    run_pipeline()
