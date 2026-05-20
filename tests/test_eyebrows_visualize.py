import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# 基础路径设置
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, "brushnet/src"))

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, ControlNetModel, DDIMScheduler
from masking_bisenet.generate_mask_bisenet import generate_bisenet_face_parts_mask
from util.dilate_mask import dilate_mask
from util.smooth_mask import smooth_mask
from util.crop_face import get_crop_info, apply_crop, restore_crop

# ======= 配置
base_model_path = "emilianJR/epiCRealism" 
brushnet_path = os.path.join(root_path, "data/ckpt/brushnetx")
controlnet_id = "lllyasviel/sd-controlnet-canny"
lora_path = os.path.join(root_path, "data/ckpt/고윤정_eyebrows_pro_v2")
input_images_dir = os.path.join(root_path, "data/raw_face_data")
output_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")
os.makedirs(output_dir, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 

# ======= 核心逻辑：拦截并融合 ControlNet (动态 Shape 匹配，防 1280 vs 320 爆显存)
def patch_pipe_for_viz(pipe, controlnet, cn_scale=0.4):
    orig_unet_forward = pipe.unet.forward

    def combined_forward(sample, timestep, encoder_hidden_states, **kwargs):
        # 记录 Latents 用于可视化
        pipe._latest_latents = sample[0].detach().cpu()
        
        cn_image = getattr(pipe, "_current_control_image", None)
        cn_down, cn_mid = None, None
        if cn_image is not None:
            cn_down, cn_mid = controlnet(
                sample, timestep, 
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cn_image,
                conditioning_scale=cn_scale,
                return_dict=False
            )

        # 兼容 Diffusers 和 BrushNet 的参数名
        down_key = 'down_block_additional_residuals' if 'down_block_additional_residuals' in kwargs else 'down_block_add_samples'
        mid_key = 'mid_block_additional_residual' if 'mid_block_additional_residual' in kwargs else 'mid_block_add_sample'
        
        brush_down = kwargs.get(down_key)
        brush_mid = kwargs.get(mid_key)

        # 🚨 核心修复：确保传入 UNet 的必定是 list，因为底层需要调用 .pop(0) 来消费特征
        if cn_down is not None and brush_down is not None:
            merged_down = []
            cn_down_list = list(cn_down)
            for b_tensor in brush_down:
                matched = False
                for i, c_tensor in enumerate(cn_down_list):
                    if b_tensor.shape == c_tensor.shape:
                        merged_down.append(b_tensor + c_tensor)
                        cn_down_list.pop(i)
                        matched = True
                        break
                if not matched:
                    merged_down.append(b_tensor)
            # merged_down 本身就是 list，直接赋值
            kwargs[down_key] = merged_down
        elif cn_down is not None:
            # 强制转换为 list
            kwargs[down_key] = list(cn_down)
        elif brush_down is not None:
            # 如果只有 brush_down，也必须确保它是 list
            kwargs[down_key] = list(brush_down)

        if cn_mid is not None and brush_mid is not None:
            kwargs[mid_key] = brush_mid + cn_mid
        elif cn_mid is not None:
            kwargs[mid_key] = cn_mid

        return orig_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)

    pipe.unet.forward = combined_forward

def visualize_latents(latents):
    """将 4 通道的 64x64 latent 转换为 2x2 网格图"""
    l_min, l_max = latents.min(), latents.max()
    normalized = ((latents - l_min) / (l_max - l_min) * 255).numpy().astype(np.uint8)
    
    rows = []
    rows.append(np.hstack([normalized[0], normalized[1]]))
    rows.append(np.hstack([normalized[2], normalized[3]]))
    grid = np.vstack(rows)
    return cv2.resize(grid, (512, 512))

def load_pipeline():
    print(f"Loading Models on {device}...")
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path, brushnet=brushnet, torch_dtype=dtype, 
        low_cpu_mem_usage=False, safety_checker=None
    )
    
    # 加载 LoRA
    from peft import PeftModel
    if os.path.exists(os.path.join(lora_path, "unet")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(lora_path, "unet"), adapter_name="gyj_brow")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(lora_path, "text_encoder"), adapter_name="gyj_brow")
        # 🚨 解决“海苔眉”：强制降权至 0.4
        pipe.set_adapters(["gyj_brow"], adapter_weights=[0.4])

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    controlnet.to(device) 
    
    patch_pipe_for_viz(pipe, controlnet, cn_scale=0.4)
    return pipe

def run_viz_test(pipe, image_path):
    img_basename = os.path.basename(image_path).split('.')[0]
    print(f"Processing: {img_basename}")
    
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: return
    rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # 1. Mask & Crop
    raw_mask = generate_bisenet_face_parts_mask(original_bgr, parts=["eyebrows"])
    # 🚨 核心膨胀：放大像素至 35，给模型边缘生长空间
    processed_mask = smooth_mask(dilate_mask(raw_mask, pixels=35))
    crop_info = get_crop_info(processed_mask, rgb_image.shape, target_size=512)
    
    image_512 = apply_crop(rgb_image, crop_info, 512)
    mask_512 = apply_crop(processed_mask, crop_info, 512)
    
    # 2. Prepare BrushNet Input (强制保留纯正黑洞)
    mask_3ch = np.repeat(mask_512[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
    masked_image_512 = (image_512 * (1.0 - mask_3ch)).astype(np.uint8)
    
    # 3. Prepare ControlNet Input
    canny_cv = cv2.Canny(image_512, 100, 200)
    canny_pil = Image.fromarray(cv2.cvtColor(canny_cv, cv2.COLOR_GRAY2RGB))
    
    # 🚨 放弃 Processor，直接用纯 Numpy 手捏 Tensor (杜绝缺失属性报错)
    cn_array = np.array(canny_pil).astype(np.float32) / 255.0
    cn_tensor = torch.from_numpy(cn_array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    cn_tensor = torch.cat([cn_tensor] * 2) # CFG = True
    pipe._current_control_image = cn_tensor

    # 4. Inference
    generator = torch.Generator(device).manual_seed(42)
    output_pil = pipe(
        # 🚨 斩首厚重感，强迫生出细丝
        prompt="RAW photo, delicate and sparse eyebrows, fine individual hair strokes, light textured hair, clear skin pores, photorealistic, 8k uhd, macro photography",
        negative_prompt="thick, dark, marker, painted, cartoon, symmetric, artificial, blocky, double eyebrows, distorted, blurry, face, nose",
        image=Image.fromarray(masked_image_512),
        mask=Image.fromarray(mask_512),
        num_inference_steps=30,
        guidance_scale=7.5,
        brushnet_conditioning_scale=0.9,
        generator=generator,
    ).images[0]
    
    generated_np = np.array(output_pil)
    
    latent_viz = visualize_latents(pipe._latest_latents)
    latent_viz_rgb = cv2.cvtColor(latent_viz, cv2.COLOR_GRAY2RGB)
    
    # 5. Restoration
    result_full_rgb = restore_crop(generated_np, crop_info, rgb_image.shape)
    mask_3d_full = np.repeat(smooth_mask(processed_mask)[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
    final_rgb = (rgb_image * (1 - mask_3d_full) + result_full_rgb * mask_3d_full).astype(np.uint8)
    
    # ======= Create Visualization Grid =======
    def add_label(img, text):
        img_cp = img.copy()
        cv2.putText(img_cp, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return img_cp

    row1 = np.hstack([
        add_label(image_512, "Original Crop"),
        add_label(cv2.cvtColor(mask_512, cv2.COLOR_GRAY2RGB), "Mask"),
        add_label(masked_image_512, "Masked Original")
    ])
    
    row2 = np.hstack([
        add_label(generated_np, "Generated Raw"),
        add_label(latent_viz_rgb, "Latent Map (4ch)"),
        add_label(cv2.resize(final_rgb, (512, 512)), "Final Result")
    ])
    
    full_grid = np.vstack([row1, row2])
    
    save_path = os.path.join(output_dir, f"viz_{img_basename}.png")
    Image.fromarray(full_grid).save(save_path)
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    viz_pipe = load_pipeline()
    all_imgs = sorted([f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg'))])
    for img_file in all_imgs[:3]:
        run_viz_test(viz_pipe, os.path.join(input_images_dir, img_file))