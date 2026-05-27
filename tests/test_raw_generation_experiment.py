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
from util.crop_face import get_actor_face_crop_info, get_zoom_crop_info, apply_crop, restore_crop
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from simple_lama_inpainting import SimpleLama
import transformers
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor

#======= MediaPipe Model Download & Setup
model_path = os.path.join(root_path, "data", "face_landmarker.task")
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task...")
    import urllib.request
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )
    print("✅ face_landmarker.task download complete.")

# Initialize MediaPipe detector once
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def get_landmarks_new(image_np):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]

def make_brow_mask_from_landmarks(image_np, padding_ratio=0.5):
    h, w = image_np.shape[:2]
    lm = get_landmarks_new(image_np)
    if lm is None:
        return np.zeros((h, w), dtype=np.uint8)

    brow_mask = np.zeros((h, w), dtype=np.uint8)
    eye_mask  = np.zeros((h, w), dtype=np.uint8)

    # Eyebrows mask
    for brow_idx in [LEFT_BROW, RIGHT_BROW]:
        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in brow_idx])
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        brow_w = x_max - x_min
        brow_h = y_max - y_min

        pad_x = int(brow_w * padding_ratio)
        pad_y = int(brow_h * padding_ratio * 2)

        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)

        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(brow_mask, hull, 255)
        brow_mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(
            brow_mask[y_min:y_max, x_min:x_max],
            np.full((y_max-y_min, x_max-x_min), 255, dtype=np.uint8)
        )

    # Eyes mask (exclusion area)
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in eye_idx])
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(eye_mask, hull, 255)
    
    # Dilate eye mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    eye_mask = cv2.dilate(eye_mask, k)

    # Remove eyes from eyebrow mask
    final_mask = cv2.bitwise_and(brow_mask, cv2.bitwise_not(eye_mask))
    final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    return final_mask

#======= Configuration
base_model_path = "emilianJR/epiCRealism" 
v4_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_female_integrated")
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_dir = os.path.join(root_path, "tests/data/eyebrow_tests/raw_generation_experiment")

os.makedirs(output_dir, exist_ok=True)

UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"
STABLE_CN_SCALE = 0

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

#======= Device Setup
if torch.cuda.is_available():
    device = "cuda"; dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32

def get_canny_guide(image_np):
    img = cv2.Canny(image_np, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)

def color_transfer(src, ref, mask):
    bg_mask = (mask == 0)
    if not np.any(bg_mask): return src
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        src_channel = src_lab[:, :, i]
        ref_channel = ref_lab[:, :, i]
        mean_src, std_src = src_channel[bg_mask].mean(), src_channel[bg_mask].std()
        mean_ref, std_ref = ref_channel[bg_mask].mean(), ref_channel[bg_mask].std()
        if std_src > 1e-5:
            src_lab[:, :, i] = (src_channel - mean_src) * (std_ref / std_src) + mean_ref
        else:
            src_lab[:, :, i] = src_channel - mean_src + mean_ref
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def load_pipeline():
    print(f"Loading base pipeline and loading V4 LoRA checkpoint...")
    backbone = DiffusionBackbone(model_id=base_model_path, dtype=dtype)
    text_encoder, vae, unet = backbone.load_modules()
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_path, text_encoder=text_encoder, vae=vae, unet=unet,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None
    )
    
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(v4_lora_path, "unet"), adapter_name="unified_v4")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(v4_lora_path, "text_encoder"), adapter_name="unified_v4")
    print(f"✅ Loaded LoRA V4 checkpoint.")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device != "cuda":
        pipe.to(device); pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
    else:
        pipe.enable_model_cpu_offload()
    return pipe

def main():
    lora_scale = 1.15
    strength = 0.60
    
    # Use only actor.jpeg (same as test_mediapipe_mask.py)
    test_img_paths = []
    
    actor_path = os.path.join(root_path, "data", "actor.jpeg")
    if os.path.exists(actor_path):
        test_img_paths.append(actor_path)
    else:
        print(f"Error: actor.jpeg not found at {actor_path}")
        return
        
    print(f"Starting Raw Generation Experiment on {len(test_img_paths)} test images.")
    
    pipe = load_pipeline()
    pipe.set_adapters(["unified_v4"], adapter_weights=[lora_scale])
    
    # Initialize LaMa model
    print("Loading LaMa inpainting model...")
    lama = SimpleLama()
    
    for idx, image_path in enumerate(test_img_paths):
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n[{idx+1}/{len(test_img_paths)}] Processing image: {image_path}")
        
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        # 1. Mask Generation on original face
        raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
        raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
        raw_mask_base = smooth_mask(raw_mask_base)
        
        # 2. Crop to 512x512 (Using closer zoom crop for higher detail resolution)
        crop_info = get_zoom_crop_info(raw_mask_base, original_bgr.shape, padding_ratio=1.3, min_size=512)
        image_512 = apply_crop(original_bgr, crop_info, target_size=512)
        mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)
        
        # 3. LaMa Eraser (Erase Eyebrows using MediaPipe landmarks adaptive mask)
        mask_512_adaptive = make_brow_mask_from_landmarks(image_512, padding_ratio=0.5)
        if np.sum(mask_512_adaptive) == 0:
            mask_512_adaptive = mask_512_binary
            
        image_pil = Image.fromarray(cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_512_adaptive).convert('L')
        
        # Erase eyebrows (three passes)
        no_brow_pil = lama(image_pil, mask_pil)
        no_brow_pil = lama(no_brow_pil, mask_pil)
        no_brow_pil = lama(no_brow_pil, mask_pil)
        masked_image_512 = cv2.cvtColor(np.array(no_brow_pil), cv2.COLOR_RGB2BGR)
        
        image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
        pipe_mask_pil = Image.new("RGB", (512, 512), "white")
        control_image_pil = get_canny_guide(image_512)
        
        # Create full-size erased image to avoid double-eyebrow/concealer patches underneath
        restored_erased_full = restore_crop(masked_image_512, crop_info, original_bgr.shape)
        orig_mask_np = raw_mask_base.astype(np.float32) / 255.0
        if len(orig_mask_np.shape) == 2:
            orig_mask_np = orig_mask_np[:, :, np.newaxis]
        
        ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
        orig_mask_blurred = cv2.GaussianBlur(orig_mask_np, (ksize, ksize), 0)
        if len(orig_mask_blurred.shape) == 2:
            orig_mask_blurred = orig_mask_blurred[:, :, np.newaxis]
        original_erased_bgr = (restored_erased_full * orig_mask_blurred + original_bgr * (1.0 - orig_mask_blurred)).astype(np.uint8)
        
        # Save intermediate erased face for user review
        erased_path = os.path.join(output_dir, f"erased_{img_basename}.png")
        cv2.imwrite(erased_path, original_erased_bgr)
        print(f"  - Saved intermediate erased face to: {erased_path}")
        
        def overlay_mask(img_rgb, mask_2d, color=(255, 60, 60), alpha=0.45):
            """Draw a semi-transparent mask overlay on an RGB image (mediapipe mask style)."""
            out = img_rgb.copy()
            colored = np.zeros_like(out)
            colored[mask_2d > 127] = color
            cv2.addWeighted(colored, alpha, out, 1 - alpha, 0, out)
            # Draw mask contour
            contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
            return out

        def add_label(img_rgb, label, color=(255, 255, 255)):
            out = img_rgb.copy()
            cv2.rectangle(out, (0, 0), (img_rgb.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
            return out

        # Grid col 1: Original + mask overlay (mediapipe mask style)
        preview_orig = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)
        preview_orig = overlay_mask(preview_orig, mask_512_adaptive)
        preview_orig = add_label(preview_orig, "Original + Mask")

        # Grid col 2: LaMa erased (diffusion input)
        preview_inpainted = cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB)
        preview_inpainted = add_label(preview_inpainted, "Erased (Diffusion Input)", color=(180, 230, 255))

        grid_cols = [preview_orig, preview_inpainted]
        
        # 4. Generate each celeb style
        for case in comparison_cases:
            celeb = case["celeb"]
            display_name = case["display_name"]
            current_prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=celeb)
            
            print(f"  - Generating {display_name}...")
            generator = torch.Generator(device).manual_seed(42)
            
            output_pil = pipe(
                prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
                image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
                controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
                guidance_scale=6.0, strength=strength, generator=generator
            ).images[0]
            
            # Post-processing
            result_np_512 = np.array(output_pil)
            result_bgr_512 = cv2.cvtColor(result_np_512, cv2.COLOR_RGB2BGR)
            
            # Apply color transfer correction to SD output (matching to erased crop)
            corrected_bgr_512 = color_transfer(result_bgr_512, masked_image_512, mask_512_binary)
            
            # Restore and blend
            restored_full = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)
            
            # Detect generated eyebrows dynamically on the output crop image
            new_raw_mask = generate_bisenet_face_parts_mask(corrected_bgr_512, parts=["eyebrows"])
            if np.sum(new_raw_mask) == 0:
                new_processed_mask = mask_512_binary
            else:
                new_raw_mask_base = dilate_mask(new_raw_mask, pixels=15)
                new_processed_mask = smooth_mask(new_raw_mask_base)

            # Restore the new crop mask back to original resolution (2D shape)
            new_restored_mask = restore_crop(new_processed_mask, crop_info, original_bgr.shape[:2])

            # Soft alpha-blending using the new mask directly to prevent alignment issues
            new_mask_np = new_restored_mask.astype(np.float32) / 255.0
            if len(new_mask_np.shape) == 2:
                new_mask_np = new_mask_np[:, :, np.newaxis]
            
            new_mask_blurred = cv2.GaussianBlur(new_mask_np, (ksize, ksize), 0)
            if len(new_mask_blurred.shape) == 2:
                new_mask_blurred = new_mask_blurred[:, :, np.newaxis]

            # Blend restored full generated image with the erased full-size background
            final_result_bgr = (restored_full * new_mask_blurred + original_erased_bgr * (1.0 - new_mask_blurred)).astype(np.uint8)
            
            # Apply crop to final result to get blended face view
            blended_cropped = apply_crop(final_result_bgr, crop_info, target_size=512)
            
            # Visualizations for grid (mediapipe mask output style)
            # SD output: show with detected new eyebrow mask overlay
            preview_raw = cv2.cvtColor(corrected_bgr_512, cv2.COLOR_BGR2RGB)
            preview_raw = overlay_mask(preview_raw, new_processed_mask, color=(60, 200, 80))
            preview_raw = add_label(preview_raw, f"{display_name} SD + New Mask", color=(180, 255, 180))

            # Final blended result (crop region)
            preview_blended = cv2.cvtColor(blended_cropped, cv2.COLOR_BGR2RGB)
            preview_blended = add_label(preview_blended, f"{display_name} Final", color=(255, 230, 120))

            grid_cols.extend([preview_raw, preview_blended])
            
        # Combine column panels into a single wide image row
        grid_row = np.hstack(grid_cols)
        grid_path = os.path.join(output_dir, f"grid_{img_basename}.png")
        Image.fromarray(grid_row).save(grid_path)
        print(f"  - Saved Grid row to: {grid_path}")
        
    print("\nExperiment completed successfully! Output directory:", output_dir)

if __name__ == "__main__":
    main()
