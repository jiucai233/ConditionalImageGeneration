import os
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# Setup paths
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
unet_safetensors = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_all_pro_v4/unet/adapter_model.safetensors")
output_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(unet_safetensors):
    print(f"Error: Unified UNet LoRA safetensors not found at {unet_safetensors}")
    sys.exit(1)

print(f"Loading UNet LoRA weights from: {unet_safetensors}")
state_dict = load_file(unet_safetensors)

# Find pairs of lora_A and lora_B
lora_pairs = {}
for key in state_dict.keys():
    if "lora_A" in key:
        base_name = key.replace(".lora_A.default.weight", "")
        b_key = key.replace("lora_A", "lora_B")
        if b_key in state_dict:
            lora_pairs[base_name] = (state_dict[key], state_dict[b_key])

print(f"Found {len(lora_pairs)} LoRA adapter layers.")

# Reconstruct Delta W and compute norms
delta_w_magnitudes = {}
all_weights = []

for layer_name, (A, B) in lora_pairs.items():
    # A shape: [r, d_in], B shape: [d_out, r]
    # Multiply B * A to get Delta W of shape [d_out, d_in]
    A = A.float()
    B = B.float()
    
    # Calculate product B @ A representing actual weight delta
    delta_w = torch.matmul(B, A)
    
    # Flatten and store for overall distribution
    all_weights.extend(delta_w.cpu().numpy().flatten())
    
    # Compute L1 norm (average absolute change)
    l1_norm = delta_w.abs().mean().item()
    
    # Clean up name for readability
    clean_name = layer_name.replace("base_model.model.unet.", "")
    delta_w_magnitudes[clean_name] = l1_norm

all_weights = np.array(all_weights)

# 1. Plot Weight Distribution
print("Generating weight distribution plot...")
plt.figure(figsize=(9, 6))
n, bins, patches = plt.hist(all_weights, bins=100, density=True, alpha=0.75, color='#3B82F6', edgecolor='none')

# Fit a normal distribution
mu, std = np.mean(all_weights), np.std(all_weights)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5*((x-mu)/std)**2) / (std * np.sqrt(2*np.pi))
plt.plot(x, p, 'r--', linewidth=2, label=f'Fit (μ={mu:.5f}, σ={std:.4f})')

plt.title("Unified LoRA Weight Change ($\Delta W = B \\times A$) Distribution", fontsize=13, fontweight='bold')
plt.xlabel("Weight Value", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()

dist_path = os.path.join(output_dir, "lora_weight_distribution.png")
plt.savefig(dist_path, dpi=200)
plt.close()
print(f"Saved weight distribution plot to: {dist_path}")

# 2. Sort layers by L1 norm and plot Top 15 active layers
print("Generating layer importance plot...")
sorted_layers = sorted(delta_w_magnitudes.items(), key=lambda x: x[1], reverse=True)

# Top 15
top_15 = sorted_layers[:15]
top_names = [x[0] for x in top_15]
top_values = [x[1] for x in top_15]

plt.figure(figsize=(11, 7))
y_pos = np.arange(len(top_names))
plt.barh(y_pos, top_values, align='center', color='#00C0A3', alpha=0.85)
plt.yticks(y_pos, top_names, fontsize=8)
plt.gca().invert_yaxis()  # top-down
plt.xlabel("Mean Absolute Weight Change ($|\\Delta W|$)", fontsize=11)
plt.title("Top 15 Most Active LoRA Layers in Unified Model", fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()

importance_path = os.path.join(output_dir, "lora_layer_importance.png")
plt.savefig(importance_path, dpi=200)
plt.close()
print(f"Saved layer importance plot to: {importance_path}")

# Print report
print("\n" + "="*60)
print("       UNIFIED LORA WEIGHTS ANALYSIS REPORT")
print("="*60)
print(f"Total LoRA Layers analyzed: {len(lora_pairs)}")
print(f"Total parameters in delta_W: {len(all_weights):,}")
print(f"Weight Mean: {mu:.6f}")
print(f"Weight Std:  {std:.6f}")
print(f"Weight Max:  {np.max(all_weights):.6f}")
print(f"Weight Min:  {np.min(all_weights):.6f}")
print("-"*60)
print("Top 5 Most Active Layers (Largest Weight Adjustments):")
for i, (name, val) in enumerate(sorted_layers[:5]):
    print(f" {i+1}. {name}")
    print(f"    Mean absolute change |ΔW|: {val:.6f}")
print("="*60)
print("\n🎉 Analysis finished successfully!")
