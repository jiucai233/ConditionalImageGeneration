import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure we can import local modules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "brushnet/src"))

from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from util.dilate_mask import dilate_mask
from util.smooth_mask import smooth_mask
from util.crop_face import get_zoom_crop_info, apply_crop, restore_crop
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from simple_lama_inpainting import SimpleLama
import transformers
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor

# ======= MediaPipe Setup
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

    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in eye_idx])
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(eye_mask, hull, 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    eye_mask = cv2.dilate(eye_mask, k)
    final_mask = cv2.bitwise_and(brow_mask, cv2.bitwise_not(eye_mask))
    final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    return final_mask

# ======= Configuration
base_model_path = "emilianJR/epiCRealism"
v4_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_female_integrated")
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_dir = os.path.join(root_path, "tests/data/eyebrow_tests/pipeline_stages")

os.makedirs(output_dir, exist_ok=True)

# Use only one celeb style for the stage visualization
TARGET_CELEB = "고윤정"
TARGET_CELEB_DISPLAY = "Go Youn Jung"

UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"
STABLE_CN_SCALE = 0

STAGE_LABELS = [
    "1. Original",
    "2. Erased",
    "3. Crop Input\n(to Diffusion)",
    "4. SD Output",
    "5. Final Result",
]

# ======= Device
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

def add_label(img_rgb, label, font_scale=0.65, thickness=2, bg_alpha=0.55):
    """Add a text label banner at the bottom of a 512x512 RGB image."""
    img = img_rgb.copy()
    h, w = img.shape[:2]
    lines = label.split('\n')
    line_h = int(font_scale * 28 + 6)
    bar_h = line_h * len(lines) + 8
    # Semi-transparent dark banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    for i, line in enumerate(lines):
        y = h - bar_h + line_h * i + line_h - 4
        # Shadow
        cv2.putText(img, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Text
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def load_pipeline():
    print(f"Loading base pipeline and V4 LoRA checkpoint...")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=dtype)

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

def process_image(image_path, pipe, lama, lora_scale=1.15, strength=0.60):
    """
    Run the full pipeline on one image and return a list of 5 stage images (RGB, 512x512).
    Stages:
        0: Original (crop region, 512x512)
        1: Erased   (crop region of erased full-size image, 512x512)
        2: Crop input to diffusion (512x512)
        3: SD raw output (512x512)
        4: Final blended result (crop region, 512x512)
    """
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n  Processing: {image_path}")

    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"  ❌ Failed to read image: {image_path}")
        return None

    # --- Stage 1: Generate eyebrow mask on original ---
    raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
    raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
    raw_mask_base = smooth_mask(raw_mask_base)

    # --- Stage 2: Zoom crop to 512x512 ---
    crop_info = get_zoom_crop_info(raw_mask_base, original_bgr.shape, padding_ratio=1.3, min_size=512)
    image_512 = apply_crop(original_bgr, crop_info, target_size=512)
    mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)

    # Stage 0 image: original face crop (512x512 BGR → RGB)
    stage0_orig = cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB)

    # --- Stage 3: LaMa erase eyebrows (3 passes) ---
    mask_512_adaptive = make_brow_mask_from_landmarks(image_512, padding_ratio=0.5)
    if np.sum(mask_512_adaptive) == 0:
        mask_512_adaptive = mask_512_binary

    image_pil = Image.fromarray(cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB))
    mask_pil  = Image.fromarray(mask_512_adaptive).convert('L')

    no_brow_pil = lama(image_pil, mask_pil)
    no_brow_pil = lama(no_brow_pil, mask_pil)
    no_brow_pil = lama(no_brow_pil, mask_pil)
    masked_image_512 = cv2.cvtColor(np.array(no_brow_pil), cv2.COLOR_RGB2BGR)

    # Create full-size erased background (for final blending)
    restored_erased_full = restore_crop(masked_image_512, crop_info, original_bgr.shape)
    orig_mask_np = raw_mask_base.astype(np.float32) / 255.0
    if len(orig_mask_np.shape) == 2:
        orig_mask_np = orig_mask_np[:, :, np.newaxis]
    ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
    orig_mask_blurred = cv2.GaussianBlur(orig_mask_np, (ksize, ksize), 0)
    if len(orig_mask_blurred.shape) == 2:
        orig_mask_blurred = orig_mask_blurred[:, :, np.newaxis]
    original_erased_bgr = (restored_erased_full * orig_mask_blurred + original_bgr * (1.0 - orig_mask_blurred)).astype(np.uint8)

    # Stage 1 image: erased face (crop the erased full-size image back to same region)
    erased_crop = apply_crop(original_erased_bgr, crop_info, target_size=512)
    stage1_erased = cv2.cvtColor(erased_crop, cv2.COLOR_BGR2RGB)

    # Stage 2 image: crop input sent to diffusion (the lama-erased 512x512)
    stage2_crop_input = cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB)

    # --- Stage 4: Stable Diffusion inpaint ---
    pipe.set_adapters(["unified_v4"], adapter_weights=[lora_scale])
    prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=TARGET_CELEB)
    pipe_mask_pil = Image.new("RGB", (512, 512), "white")
    control_image_pil = get_canny_guide(image_512)
    sd_input_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))

    generator = torch.Generator(device).manual_seed(42)
    output_pil = pipe(
        prompt=prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
        image=sd_input_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
        controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
        guidance_scale=6.0, strength=strength, generator=generator
    ).images[0]

    result_bgr_512 = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

    # Stage 3 image: raw SD output
    stage3_sd_output = cv2.cvtColor(result_bgr_512, cv2.COLOR_BGR2RGB)

    # --- Stage 5: Post-processing & blending ---
    corrected_bgr_512 = color_transfer(result_bgr_512, masked_image_512, mask_512_binary)
    restored_full = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)

    # Detect new eyebrows dynamically on SD output
    new_raw_mask = generate_bisenet_face_parts_mask(corrected_bgr_512, parts=["eyebrows"])
    if np.sum(new_raw_mask) == 0:
        new_processed_mask = mask_512_binary
    else:
        new_raw_mask_base = dilate_mask(new_raw_mask, pixels=15)
        new_processed_mask = smooth_mask(new_raw_mask_base)

    new_restored_mask = restore_crop(new_processed_mask, crop_info, original_bgr.shape[:2])
    new_mask_np = new_restored_mask.astype(np.float32) / 255.0
    if len(new_mask_np.shape) == 2:
        new_mask_np = new_mask_np[:, :, np.newaxis]
    new_mask_blurred = cv2.GaussianBlur(new_mask_np, (ksize, ksize), 0)
    if len(new_mask_blurred.shape) == 2:
        new_mask_blurred = new_mask_blurred[:, :, np.newaxis]

    final_result_bgr = (restored_full * new_mask_blurred + original_erased_bgr * (1.0 - new_mask_blurred)).astype(np.uint8)

    # Stage 4 image: final result (crop back to the same region for comparison)
    final_crop = apply_crop(final_result_bgr, crop_info, target_size=512)
    stage4_final = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)

    return [stage0_orig, stage1_erased, stage2_crop_input, stage3_sd_output, stage4_final]


def make_stage_row(stages, img_name):
    """Combine 5 stage images into one horizontal strip with labels."""
    CELL = 512
    labeled = []
    for img, label in zip(stages, STAGE_LABELS):
        cell = cv2.resize(img, (CELL, CELL))  # already 512
        cell = add_label(cell, label)
        labeled.append(cell)

    row = np.hstack(labeled)

    # Add image name banner on top
    header_h = 44
    header = np.zeros((header_h, CELL * 5, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    title = f"Image: {img_name}  |  Target: {TARGET_CELEB_DISPLAY}"
    cv2.putText(header, title, (16, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (200, 220, 255), 2, cv2.LINE_AA)

    final = np.vstack([header, row])
    return final


def main():
    # Build list: actor.jpeg + first 5 from raw_face_data
    test_img_paths = []

    actor_path = os.path.join(root_path, "data", "actor.jpeg")
    if os.path.exists(actor_path):
        test_img_paths.append(actor_path)

    all_imgs = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for img_file in all_imgs[:5]:
        test_img_paths.append(os.path.join(input_images_dir, img_file))

    print(f"Pipeline Stage Visualization")
    print(f"Target: {len(test_img_paths)} images  |  Style: {TARGET_CELEB_DISPLAY}")
    print(f"Output dir: {output_dir}\n")

    print("Loading LaMa inpainting model...")
    lama = SimpleLama()

    pipe = load_pipeline()

    all_rows = []

    for idx, image_path in enumerate(test_img_paths):
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n[{idx+1}/{len(test_img_paths)}] {img_basename}")

        stages = process_image(image_path, pipe, lama)
        if stages is None:
            continue

        # Save individual stage images
        stage_names = ["1_original", "2_erased", "3_crop_input", "4_sd_output", "5_final"]
        for s_img, s_name in zip(stages, stage_names):
            out_path = os.path.join(output_dir, f"{img_basename}_{s_name}.png")
            Image.fromarray(s_img).save(out_path)

        # Make and save row strip
        row = make_stage_row(stages, img_basename)
        row_path = os.path.join(output_dir, f"row_{img_basename}.png")
        Image.fromarray(row).save(row_path)
        print(f"  ✅ Saved row: {row_path}")

        all_rows.append(row)

    # Combine all rows into one big grid
    if all_rows:
        # Pad rows to the same width (should already be equal)
        max_w = max(r.shape[1] for r in all_rows)
        padded = []
        for r in all_rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)

        # Add separator line between rows
        sep = np.zeros((4, max_w, 3), dtype=np.uint8)
        sep[:] = (60, 60, 60)
        interleaved = []
        for i, r in enumerate(padded):
            interleaved.append(r)
            if i < len(padded) - 1:
                interleaved.append(sep)

        full_grid = np.vstack(interleaved)
        grid_path = os.path.join(output_dir, "FULL_pipeline_stages_grid.png")
        Image.fromarray(full_grid).save(grid_path)
        print(f"\n✅ Full grid saved: {grid_path}")

    print(f"\nDone! Output directory: {output_dir}")


if __name__ == "__main__":
    main()
