import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Safe headless matplotlib backend
import matplotlib.pyplot as plt

# Ensure we can import local modules
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from util.dilate_mask import dilate_mask
from util.smooth_mask import smooth_mask
from util.crop_face import get_actor_face_crop_info, apply_crop, restore_crop
from util.color_transfer import color_transfer
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
import transformers

if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor

#======= Configuration
base_model_path = "emilianJR/epiCRealism" 
controlnet_id = "lllyasviel/sd-controlnet-canny"
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_grid_dir = os.path.join(root_path, "tests/data/eyebrow_tests/grids_v2_v3_v4")
output_vis_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")

os.makedirs(output_grid_dir, exist_ok=True)
os.makedirs(output_vis_dir, exist_ok=True)

#======= Golden Parameters
STABLE_STRENGTH = 0.60
STABLE_CN_SCALE = 0.4
STABLE_LORA_SCALE = 1.15

UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"

comparison_cases = [
    { "celeb": "고윤정", "display_name": "Go Youn Jung" },
    { "celeb": "신세경", "display_name": "Shin Se Kyung" },
    { "celeb": "홍수주", "display_name": "Hong Su Zu" }
]

#======= Feature Storage
feature_vectors = { "v2": [], "v3": [], "v4": [] }
feature_labels = { "v2": [], "v3": [], "v4": [] }

class EyebrowFeatureHook:
    """
    Hook to capture features from the UNet up-block attention layers.
    Specifically captures the final 3 steps of diffusion where eyebrow characteristics are fully defined.
    """
    def __init__(self, celeb, mask_512_binary, total_steps):
        self.celeb = celeb
        self.mask_512_binary = mask_512_binary
        self.total_steps = total_steps
        self.step_counter = 0
        self.features_extracted = []

    def __call__(self, module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        
        # Separate CFG batches: index 1 is positive prompt conditioning
        idx = 1 if tensor.shape[0] > 1 else 0
        val = tensor[idx].detach().cpu().float().numpy()
        
        # Extract features only during the last 3 steps of generation
        if self.step_counter >= max(0, self.total_steps - 3):
            if len(val.shape) == 3:  # [C, H, W]
                c, h, w = val.shape
            elif len(val.shape) == 2:  # [Seq_len, C]
                seq_len, c = val.shape
                import math
                h = w = int(math.sqrt(seq_len))
                val = val.reshape(h, w, c).transpose(2, 0, 1)  # [C, H, W]
            else:
                self.step_counter += 1
                return
            
            # Downsample the mask to match feature map spatial dimensions (H x W)
            mask_resized = cv2.resize(self.mask_512_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized.astype(np.float32) / 255.0
            
            # Masked average pooling
            mask_sum = mask_resized.sum()
            if mask_sum > 0:
                masked_avg = (val * mask_resized).sum(axis=(1, 2)) / mask_sum
            else:
                masked_avg = val.mean(axis=(1, 2))
                
            self.features_extracted.append(masked_avg)
            
        self.step_counter += 1

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

def load_pipeline_for_version(version):
    print(f"\n==================================================")
    print(f"Loading Base Pipeline and LoRA {version.upper()}...")
    print(f"==================================================")
    backbone = DiffusionBackbone(model_id=base_model_path, dtype=dtype)
    text_encoder, vae, unet = backbone.load_modules()
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, text_encoder=text_encoder, vae=vae, unet=unet,
        torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None
    )
    
    lora_path = os.path.join(root_path, f"lora_checkpoint/celeb_eyebrows_all_pro_{version}")
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(lora_path, "unet"), adapter_name="default")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(lora_path, "text_encoder"), adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[STABLE_LORA_SCALE])
    print(f"✅ Successfully loaded LoRA {version.upper()} from: {lora_path} with scale {STABLE_LORA_SCALE}")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device != "cuda":
        pipe.to(device); pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
    else:
        pipe.enable_model_cpu_offload()
    return pipe

def process_single_image(pipe, image_path, version, results_dict):
    img_basename = os.path.basename(image_path).split('.')[0]
    print(f"  - Version {version.upper()} processing: {img_basename}")
    
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: return
    
    # 1. Mask Generation
    raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
    raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
    raw_mask_base = smooth_mask(raw_mask_base)
    
    # Pre-calculate Mask and Crop using actor-scale cropping
    crop_info = get_actor_face_crop_info(raw_mask_base, original_bgr.shape, padding_ratio=4.0)
    image_512 = apply_crop(original_bgr, crop_info, target_size=512)
    mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)
    
    # Telea Fill for base texture
    textured_fill = cv2.inpaint(image_512, mask_512_binary, 3, cv2.INPAINT_TELEA)
    mask_3ch_smooth = np.repeat(smooth_mask(mask_512_binary)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
    masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
    
    image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
    pipe_mask_pil = Image.new("RGB", (512, 512), "white")
    control_image_pil = get_canny_guide(image_512)
    
    # Keep reference to the crop face original image
    preview_orig = cv2.resize(cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB), (512, 512))
    cv2.putText(preview_orig, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    results_dict[img_basename]["original"] = preview_orig

    # 2. Iterate each Celeb style
    for case in comparison_cases:
        celeb = case["celeb"]
        display_name = case["display_name"]
        current_prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=celeb)
        
        generator = torch.Generator(device).manual_seed(42)
        
        # Register Forward Hook for feature capture
        total_steps = int(40 * STABLE_STRENGTH)
        hook = EyebrowFeatureHook(celeb, mask_512_binary, total_steps)
        hook_handle = pipe.unet.up_blocks[1].attentions[1].register_forward_hook(hook)

        output_pil = pipe(
            prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
            image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
            controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
            guidance_scale=6.0, strength=STABLE_STRENGTH, generator=generator
        ).images[0]
        
        # Unregister hook
        hook_handle.remove()
        
        # Record features
        for feat in hook.features_extracted:
            feature_vectors[version].append(feat)
            feature_labels[version].append(celeb)
        
        # Restore cropped generated region back to full face BGR with color transfer correction
        result_np_512 = np.array(output_pil)
        result_bgr_512 = cv2.cvtColor(result_np_512, cv2.COLOR_RGB2BGR)
        corrected_bgr_512 = color_transfer(result_bgr_512, image_512, mask_512_binary)
        full_result_np = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)
        
        ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
        mask_float = smooth_mask(raw_mask_base, ksize=ksize).astype(np.float32) / 255.0
        mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
        final_result_bgr = (original_bgr.astype(np.float32) * (1 - mask_3d) + full_result_np.astype(np.float32) * mask_3d).astype(np.uint8)
        
        # Crop the final blended result for grid preview (keeps it close-up and highly readable)
        blended_cropped = apply_crop(final_result_bgr, crop_info, target_size=512)
        preview_final = cv2.cvtColor(blended_cropped, cv2.COLOR_BGR2RGB)
        cv2.putText(preview_final, f"{display_name} ({version.upper()})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        results_dict[img_basename][(celeb, version)] = preview_final

def plot_3d_comparison(coords_dict, labels_dict, title, save_path, algo_name="PCA"):
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    
    eng_label_map = {
        "고윤정": "Go Yoon-jung",
        "신세경": "Shin Se-kyung",
        "홍수주": "Hong Su-zu"
    }
    
    colors = ['#FF4B4B', '#00C0A3', '#3B82F6']
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"3D Latent Space Separation ({algo_name}) - V2 vs V3 vs V4", fontsize=16, fontweight='bold', y=0.98)
    
    versions = ["v2", "v3", "v4"]
    for idx, ver in enumerate(versions):
        coords = np.array(coords_dict[ver])
        labels = np.array(labels_dict[ver])
        
        mapped_labels = np.array([eng_label_map.get(l, l) for l in labels])
        unique_labels = sorted(list(set(mapped_labels)))
        color_map = {name: colors[i] for i, name in enumerate(unique_labels)}
        
        # Calculate Silhouette Score
        try:
            score = silhouette_score(coords, mapped_labels)
            score_str = f"Silhouette Score: {score:.4f}"
        except Exception as e:
            score_str = "Silhouette N/A"
            
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        for label_name in unique_labels:
            mask = (mapped_labels == label_name)
            ax.scatter(
                coords[mask, 0], coords[mask, 1], coords[mask, 2],
                c=color_map[label_name], label=label_name,
                alpha=0.8, edgecolors='none', s=40
            )
            
        ax.set_title(f"Version {ver.upper()}\n({score_str})", fontsize=12, fontweight='semibold')
        ax.set_xlabel("Dim 1", fontsize=8)
        ax.set_ylabel("Dim 2", fontsize=8)
        ax.set_zlabel("Dim 3", fontsize=8)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
        
        # Style axis
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {algo_name} comparison plot to: {save_path}")

def main():
    all_imgs = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Use 10 images for robust quality and separation validation
    test_imgs = all_imgs[:10]
    print(f"Starting LoRA comparison on {len(test_imgs)} test images.")
    
    # Store all images' generation results for grid generation
    # Structure: results_data[img_basename] = { "original": np_img, (celeb, version): np_img }
    results_data = { os.path.basename(p).split('.')[0]: {} for p in test_imgs }
    
    versions = ["v2", "v3", "v4"]
    
    for ver in versions:
        pipe = load_pipeline_for_version(ver)
        for img_file in test_imgs:
            image_path = os.path.join(input_images_dir, img_file)
            process_single_image(pipe, image_path, ver, results_data)
        
        # Free CUDA/MPS memory before loading the next version
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n🎉 All generation runs complete! Compiling comparison grids...")
    
    # Compile 3x4 grids for each image
    for img_file in test_imgs:
        img_basename = img_file.split('.')[0]
        res = results_data[img_basename]
        
        if "original" not in res:
            continue
            
        preview_orig = res["original"]
        
        rows = []
        for case in comparison_cases:
            celeb = case["celeb"]
            # Row layout: [Original, V2, V3, V4]
            row_images = [
                preview_orig,
                res.get((celeb, "v2"), np.zeros_like(preview_orig)),
                res.get((celeb, "v3"), np.zeros_like(preview_orig)),
                res.get((celeb, "v4"), np.zeros_like(preview_orig))
            ]
            rows.append(np.hstack(row_images))
            
        grid = np.vstack(rows)
        grid_path = os.path.join(output_grid_dir, f"grid_compare_{img_basename}.png")
        Image.fromarray(grid).save(grid_path)
        print(f"  - Saved Grid: {grid_path}")
        
    # Generate 3D scatter plots
    print("\n🎉 Performing 3D PCA and 3D t-SNE dimensionality reductions on extracted UNet features...")
    
    coords_pca = {}
    coords_tsne = {}
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    for ver in versions:
        vectors = np.array(feature_vectors[ver])
        print(f"  - Version {ver.upper()} extracted feature vectors shape: {vectors.shape}")
        
        # PCA
        pca = PCA(n_components=3, random_state=42)
        coords_pca[ver] = pca.fit_transform(vectors)
        
        # t-SNE (perplexity limited by dataset size)
        perp = min(30, max(2, len(vectors) // 3))
        tsne = TSNE(n_components=3, perplexity=perp, max_iter=1000, random_state=42)
        coords_tsne[ver] = tsne.fit_transform(vectors)
        
    plot_3d_comparison(coords_pca, feature_labels, "PCA Comparison", os.path.join(output_vis_dir, "v2_v3_v4_latent_space_pca.png"), algo_name="PCA")
    plot_3d_comparison(coords_tsne, feature_labels, "t-SNE Comparison", os.path.join(output_vis_dir, "v2_v3_v4_latent_space_tsne.png"), algo_name="t-SNE")
    
    print("\n🎉 Evaluation complete! Visualizations saved in: tests/data/")

if __name__ == "__main__":
    main()
