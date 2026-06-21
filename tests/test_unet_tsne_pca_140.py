import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Headless matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Ensure we can import local modules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from util.dilate_mask import dilate_mask
from util.smooth_mask import smooth_mask
from util.crop_face import get_zoom_crop_info, apply_crop, restore_crop
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from simple_lama_inpainting import SimpleLama
from peft import PeftModel
import transformers
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor

# ======= Configuration
base_model_path = "emilianJR/epiCRealism"
integrated_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_female_integrated")
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")
os.makedirs(output_dir, exist_ok=True)

# ======= MediaPipe Setup
mp_model_path = os.path.join(root_path, "data", "face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=mp_model_path),
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

def get_canny_guide(image_np):
    img = cv2.Canny(image_np, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)

class UNetEyebrowFeatureHook:
    def __init__(self, mask_512_binary, total_steps):
        self.mask_512_binary = mask_512_binary
        self.total_steps = total_steps
        self.step_counter = 0
        self.features = []

    def __call__(self, module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        idx = 1 if tensor.shape[0] > 1 else 0
        val = tensor[idx].detach().cpu().float().numpy()
        
        # Capture features in the final 3 steps of generation
        if self.step_counter >= max(0, self.total_steps - 3):
            if len(val.shape) == 3: # [C, H, W]
                c, h, w = val.shape
            elif len(val.shape) == 2: # [Seq_len, C]
                seq_len, c = val.shape
                import math
                h = w = int(math.sqrt(seq_len))
                val = val.reshape(h, w, c).transpose(2, 0, 1) # [C, H, W]
            else:
                self.step_counter += 1
                return
            
            mask_resized = cv2.resize(self.mask_512_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized.astype(np.float32) / 255.0
            
            mask_sum = mask_resized.sum()
            if mask_sum > 0:
                masked_avg = (val * mask_resized).sum(axis=(1, 2)) / mask_sum
            else:
                masked_avg = val.mean(axis=(1, 2))
                
            self.features.append(masked_avg)
            
        self.step_counter += 1

def load_pipeline():
    if torch.cuda.is_available():
        device = "cuda"; dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"; dtype = torch.float32
    else:
        device = "cpu"; dtype = torch.float32
        
    print(f"Loading pipeline on {device}...")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=dtype)
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_path, text_encoder=text_encoder, vae=vae, unet=unet,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None
    )
    
    # Load 30000 steps integrated LoRA weights
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(integrated_lora_path, "unet"), adapter_name="unified")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(integrated_lora_path, "text_encoder"), adapter_name="unified")
    # Set scale manually on PEFT modules to bypass pipeline set_adapters compatibility check
    for model in [pipe.unet, pipe.text_encoder]:
        for module in model.modules():
            if hasattr(module, "scaling") and "unified" in module.scaling:
                module.scaling["unified"] = 1.15
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    return pipe, device

def main():
    # 1. Load pipeline & LaMa
    pipe, device = load_pipeline()
    lama = SimpleLama()

    # Celeb Setup
    celebs = ["고윤정", "신세경", "홍수주", "탑", "최시원", "뷔", "차은우"]
    celeb_display_names = {
        "고윤정": "Go Youn Jung",
        "신세경": "Shin Se Kyung",
        "홍수주": "Hong Su Zu",
        "탑": "T.O.P",
        "최시원": "Choi Si Won",
        "뷔": "V",
        "차은우": "Cha Eun Woo"
    }

    # 2. Get first 20 valid raw face images
    all_files = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    test_images = all_files[:20]
    print(f"Using {len(test_images)} input face images for feature mapping.")

    # Storage
    feature_vectors = []
    feature_labels = []

    # 3. Main generation loop (20 images x 7 celebs = 140 runs)
    for img_idx, img_name in enumerate(test_images):
        img_path = os.path.join(input_images_dir, img_name)
        print(f"\n[{img_idx+1}/20] Preprocessing: {img_name}")
        
        original_bgr = cv2.imread(img_path)
        if original_bgr is None: continue
        
        # Preprocessing mask and crop
        raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
        raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
        raw_mask_base = smooth_mask(raw_mask_base)
        
        crop_info = get_zoom_crop_info(raw_mask_base, original_bgr.shape, padding_ratio=1.3, min_size=512)
        image_512 = apply_crop(original_bgr, crop_info, target_size=512)
        mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)
        
        # LaMa Erase
        mask_512_adaptive = make_brow_mask_from_landmarks(image_512, padding_ratio=0.5)
        if np.sum(mask_512_adaptive) == 0:
            mask_512_adaptive = mask_512_binary
            
        image_pil = Image.fromarray(cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_512_adaptive).convert('L')
        
        no_brow_pil = lama(image_pil, mask_pil)
        no_brow_pil = lama(no_brow_pil, mask_pil)
        no_brow_pil = lama(no_brow_pil, mask_pil)
        masked_image_512 = cv2.cvtColor(np.array(no_brow_pil), cv2.COLOR_RGB2BGR)
        
        image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
        pipe_mask_pil = Image.new("RGB", (512, 512), "white")
        control_image_pil = get_canny_guide(image_512)

        # Run for all 7 celebs
        for celeb in celebs:
            print(f"  - Generating eyebrows for: {celeb_display_names[celeb]}")
            UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
            UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"
            current_prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=celeb)
            
            generator = torch.Generator(device).manual_seed(42)
            
            # Setup Hook on UNet up block attention layer
            total_steps = int(40 * 0.60) # 24 steps
            hook = UNetEyebrowFeatureHook(mask_512_binary, total_steps)
            hook_handle = pipe.unet.up_blocks[1].attentions[1].register_forward_hook(hook)

            # Inference
            _ = pipe(
                prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
                image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
                controlnet_conditioning_scale=0, num_inference_steps=40,
                guidance_scale=6.0, strength=0.60, generator=generator
            ).images[0]

            # Remove Hook
            hook_handle.remove()

            # Process features: average the features extracted in the last 3 steps
            if len(hook.features) > 0:
                avg_feat = np.mean(hook.features, axis=0)
                feature_vectors.append(avg_feat)
                feature_labels.append(celeb)
            else:
                print(f"  ⚠️ Warning: No features captured for {celeb}")

    # 4. Dimensionality Reduction & Visualization
    if len(feature_vectors) == 0:
        print("❌ Error: No features collected!")
        return

    features_np = np.array(feature_vectors)
    labels_np = np.array(feature_labels)
    
    eng_labels = np.array([celeb_display_names[l] for l in labels_np])
    unique_labels = sorted(list(set(eng_labels)))
    
    colors = ['#FF4B4B', '#FF7F0E', '#2CA02C', '#1f77b4', '#9467BD', '#8C564B', '#00C0A3']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(unique_labels)}
    
    print("\nRunning PCA reduction...")
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(features_np)

    print("Running t-SNE reduction...")
    tsne = TSNE(n_components=2, perplexity=10, max_iter=1000, random_state=42)
    coords_tsne = tsne.fit_transform(features_np)

    # Plotting helper
    def draw_plot(coords, title, filename):
        score = silhouette_score(coords, eng_labels)
        plt.figure(figsize=(10, 8))
        
        for name in unique_labels:
            mask = (eng_labels == name)
            plt.scatter(
                coords[mask, 0], coords[mask, 1], 
                c=color_map[name], label=name,
                alpha=0.85, edgecolors='w', s=90
            )
            
        plt.title(f"{title} (Silhouette Score: {score:.4f})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Dimension 1", fontsize=11)
        plt.ylabel("Dimension 2", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=250)
        plt.close()
        print(f"Saved plot to: {save_path}")

    draw_plot(coords_pca, "UNet Eyebrow Latent Feature PCA Space", "unet_latent_space_pca.png")
    draw_plot(coords_tsne, "UNet Eyebrow Latent Feature t-SNE Space", "unet_latent_space_tsne.png")
    print("\n🎉 Visualization generation successfully completed!")

if __name__ == "__main__":
    main()
