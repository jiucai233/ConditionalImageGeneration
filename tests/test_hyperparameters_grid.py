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
v4_lora_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_all_pro_v4")
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_base_dir = os.path.join(root_path, "tests/data/eyebrow_tests/hyperparams_grid")
output_vis_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")

os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(output_vis_dir, exist_ok=True)

UNIFIED_PROMPT_TEMPLATE = "a photo of {celeb} style eyebrows on a face, highly detailed, realistic skin texture, natural skin pores"
UNIFIED_NEGATIVE_PROMPT = "low quality, distorted, blurry, messy, ugly, asymmetric eyebrows, double eyebrows, painted, drawing, illustration, cartoon, fake, 3d render, smooth skin, blurry, plastic, purple patches, colorful noise, burnt, high contrast, hard edges, dirty skin"
STABLE_CN_SCALE = 0.75

comparison_cases = [
    { "celeb": "고윤정", "display_name": "Go Youn Jung" },
    { "celeb": "신세경", "display_name": "Shin Se Kyung" },
    { "celeb": "홍수주", "display_name": "Hong Su Zu" }
]

class EyebrowFeatureHook:
    def __init__(self, celeb, mask_512_binary, total_steps):
        self.celeb = celeb
        self.mask_512_binary = mask_512_binary
        self.total_steps = total_steps
        self.step_counter = 0
        self.features_extracted = []

    def __call__(self, module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        idx = 1 if tensor.shape[0] > 1 else 0
        val = tensor[idx].detach().cpu().float().numpy()
        
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
            
            mask_resized = cv2.resize(self.mask_512_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_resized = mask_resized.astype(np.float32) / 255.0
            
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




def load_pipeline():
    print(f"Loading base pipeline and loading V4 LoRA checkpoint...")
    backbone = DiffusionBackbone(model_id=base_model_path, dtype=dtype)
    text_encoder, vae, unet = backbone.load_modules()
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, text_encoder=text_encoder, vae=vae, unet=unet,
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

def plot_best_3d(coords, labels, score, title, save_path, algo_name="PCA"):
    from sklearn.metrics import silhouette_score
    from mpl_toolkits.mplot3d import Axes3D
    
    eng_label_map = {
        "고윤정": "Go Yoon-jung",
        "신세경": "Shin Se-kyung",
        "홍수주": "Hong Su-zu"
    }
    mapped_labels = np.array([eng_label_map.get(l, l) for l in labels])
    unique_labels = sorted(list(set(mapped_labels)))
    colors = ['#FF4B4B', '#00C0A3', '#3B82F6']
    color_map = {name: colors[i] for i, name in enumerate(unique_labels)}
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for label_name in unique_labels:
        mask = (mapped_labels == label_name)
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=color_map[label_name], label=label_name,
            alpha=0.8, edgecolors='none', s=60
        )
        
    ax.set_title(f"{title} ({algo_name})\n(Best Silhouette Score: {score:.4f})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.legend(loc='upper right', framealpha=0.9)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    # 1. Grid of parameters (4x3 grid for comprehensive search)
    lora_scales = [0.70, 0.85, 1.00, 1.15]
    strengths = [0.40, 0.50, 0.60]
    
    all_imgs = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Run on 6 images for robust cluster separation measurements
    test_imgs = all_imgs[:6]
    print(f"Starting Hyperparameter Grid Search on V4 over {len(test_imgs)} test images.")
    
    pipe = load_pipeline()
    
    grid_results = []
    
    # We will track the features collected for each config to calculate Silhouette scores
    # config_key: (vectors, labels)
    all_configs_features = {}
    
    for lora_scale in lora_scales:
        for strength in strengths:
            config_name = f"lora_{lora_scale:.2f}_strength_{strength:.2f}"
            print(f"\n>>> Running configuration: {config_name}")
            
            # Create output directory for this config
            config_out_dir = os.path.join(output_base_dir, config_name)
            os.makedirs(config_out_dir, exist_ok=True)
            
            # Set the adapter scale
            pipe.set_adapters(["unified_v4"], adapter_weights=[lora_scale])
            
            config_vectors = []
            config_labels = []
            
            for img_file in test_imgs:
                image_path = os.path.join(input_images_dir, img_file)
                img_basename = img_file.split('.')[0]
                
                original_bgr = cv2.imread(image_path)
                if original_bgr is None: continue
                
                # Mask Generation
                raw_mask_base = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
                raw_mask_base = dilate_mask(raw_mask_base, pixels=15)
                raw_mask_base = smooth_mask(raw_mask_base)
                
                # Crop
                crop_info = get_actor_face_crop_info(raw_mask_base, original_bgr.shape, padding_ratio=4.0)
                image_512 = apply_crop(original_bgr, crop_info, target_size=512)
                mask_512_binary = apply_crop(raw_mask_base, crop_info, target_size=512)
                
                # Telea Fill
                textured_fill = cv2.inpaint(image_512, mask_512_binary, 3, cv2.INPAINT_TELEA)
                mask_3ch_smooth = np.repeat(smooth_mask(mask_512_binary)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
                masked_image_512 = (image_512 * (1.0 - mask_3ch_smooth) + textured_fill * mask_3ch_smooth).astype(np.uint8)
                
                image_pil = Image.fromarray(cv2.cvtColor(masked_image_512, cv2.COLOR_BGR2RGB))
                pipe_mask_pil = Image.new("RGB", (512, 512), "white")
                control_image_pil = get_canny_guide(image_512)
                
                celeb_previews = []
                
                for case in comparison_cases:
                    celeb = case["celeb"]
                    display_name = case["display_name"]
                    current_prompt = UNIFIED_PROMPT_TEMPLATE.format(celeb=celeb)
                    
                    generator = torch.Generator(device).manual_seed(42)
                    
                    # Hook features
                    total_steps = int(40 * strength)
                    hook = EyebrowFeatureHook(celeb, mask_512_binary, total_steps)
                    hook_handle = pipe.unet.up_blocks[1].attentions[1].register_forward_hook(hook)

                    output_pil = pipe(
                        prompt=current_prompt, negative_prompt=UNIFIED_NEGATIVE_PROMPT,
                        image=image_pil, mask_image=pipe_mask_pil, control_image=control_image_pil,
                        controlnet_conditioning_scale=STABLE_CN_SCALE, num_inference_steps=40,
                        guidance_scale=6.0, strength=strength, generator=generator
                    ).images[0]
                    
                    hook_handle.remove()
                    
                    for feat in hook.features_extracted:
                        config_vectors.append(feat)
                        config_labels.append(celeb)
                    
                    # Restore and save with color transfer correction
                    result_np_512 = np.array(output_pil)
                    result_bgr_512 = cv2.cvtColor(result_np_512, cv2.COLOR_RGB2BGR)
                    corrected_bgr_512 = color_transfer(result_bgr_512, image_512, mask_512_binary)
                    full_result_np = restore_crop(corrected_bgr_512, crop_info, original_bgr.shape)
                    ksize = int(max(original_bgr.shape[:2]) * 0.015) | 1
                    mask_float = smooth_mask(raw_mask_base, ksize=ksize).astype(np.float32) / 255.0
                    mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
                    final_result_bgr = (original_bgr.astype(np.float32) * (1 - mask_3d) + full_result_np.astype(np.float32) * mask_3d).astype(np.uint8)
                    
                    # Blended close up
                    blended_cropped = apply_crop(final_result_bgr, crop_info, target_size=512)
                    preview_final = cv2.cvtColor(blended_cropped, cv2.COLOR_BGR2RGB)
                    cv2.putText(preview_final, f"{display_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    celeb_previews.append(preview_final)
                    
                # Save comparison grid for this image under this config
                preview_orig = cv2.resize(cv2.cvtColor(image_512, cv2.COLOR_BGR2RGB), (512, 512))
                cv2.putText(preview_orig, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                grid_row = np.hstack([preview_orig] + celeb_previews)
                grid_path = os.path.join(config_out_dir, f"grid_{img_basename}.png")
                Image.fromarray(grid_row).save(grid_path)
                
            # Calculate Silhouette Score for this config
            config_vectors = np.array(config_vectors)
            config_labels = np.array(config_labels)
            
            from sklearn.metrics import silhouette_score
            try:
                score = silhouette_score(config_vectors, config_labels)
            except Exception as e:
                score = -1.0
                
            print(f"  - Calculated Silhouette Score: {score:.5f}")
            grid_results.append({
                "lora_scale": lora_scale,
                "strength": strength,
                "score": score,
                "vectors": config_vectors,
                "labels": config_labels
            })
            
    # Print search report
    print("\n" + "="*60)
    print("      HYPERPARAMETER GRID SEARCH EVALUATION REPORT")
    print("="*60)
    best_config = None
    best_score = -2.0
    
    for res in grid_results:
        print(f"LoRA Scale: {res['lora_scale']:.2f} | Strength: {res['strength']:.2f} | Silhouette Score: {res['score']:.5f}")
        if res['score'] > best_score:
            best_score = res['score']
            best_config = res
            
    print("-"*60)
    print(f"Best Configuration Found:")
    print(f"  LoRA Scale: {best_config['lora_scale']:.2f}")
    print(f"  Inpaint Strength: {best_config['strength']:.2f}")
    print(f"  Silhouette Score: {best_config['score']:.5f}")
    print("="*60)
    
    # Generate 3D scatter plots for the best configuration
    best_vectors = best_config["vectors"]
    best_labels = best_config["labels"]
    best_title = f"V4 Latent Space (LoRA Scale: {best_config['lora_scale']:.2f}, Strength: {best_config['strength']:.2f})"
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\nGenerating 3D plots for the best hyperparameter configuration...")
    pca = PCA(n_components=3, random_state=42)
    best_pca = pca.fit_transform(best_vectors)
    plot_best_3d(best_pca, best_labels, best_score, best_title, os.path.join(output_vis_dir, "v4_best_hyperparams_pca.png"), algo_name="PCA")
    
    perp = min(30, max(2, len(best_vectors) // 3))
    tsne = TSNE(n_components=3, perplexity=perp, max_iter=1000, random_state=42)
    best_tsne = tsne.fit_transform(best_vectors)
    plot_best_3d(best_tsne, best_labels, best_score, best_title, os.path.join(output_vis_dir, "v4_best_hyperparams_tsne.png"), algo_name="t-SNE")
    
    print(f"🎉 Grid search complete! Plots saved in: {output_vis_dir}")

if __name__ == "__main__":
    main()
