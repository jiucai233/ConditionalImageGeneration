# Import core libraries
import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Setup root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

# Configuration
base_model_path = "emilianJR/epiCRealism" 
lora_text_encoder_path = os.path.join(root_path, "lora_checkpoint/celeb_eyebrows_all_pro_v4/text_encoder")
output_dir = os.path.join(root_path, "tests/data/eyebrow_visualize")
os.makedirs(output_dir, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

def generate_embeddings(text_encoder, tokenizer, templates, names):
    embeddings = []
    labels = []
    
    for name in names:
        for temp in templates:
            prompt = temp.format(name=name)
            inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = text_encoder(inputs.input_ids)
                # outputs[0] is the last hidden state of shape [1, 77, 768]
                # We perform mean pooling across tokens to get a robust sentence representation
                emb = outputs[0].mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            labels.append(name)
            
    return np.array(embeddings), np.array(labels)

def plot_3d(data_dict, labels, title, filename, algo_name="PCA"):
    """
    data_dict: {"Base": base_3d_coords, "LoRA": lora_3d_coords}
    labels: array of actor names for coloring
    """
    from sklearn.metrics import silhouette_score
    
    # Map labels to English for clean legend rendering (avoids missing font warnings for Korean chars)
    eng_label_map = {
        "고윤정": "Go Yoon-jung",
        "신세경": "Shin Se-kyung",
        "홍수주": "Hong Su-zu"
    }
    mapped_labels = np.array([eng_label_map.get(l, l) for l in labels])
    
    unique_labels = sorted(list(set(mapped_labels)))
    colors = ['#FF4B4B', '#00C0A3', '#3B82F6'] # Premium Red, Teal, Blue colors
    color_map = {name: colors[i] for i, name in enumerate(unique_labels)}
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"3D Latent Space Visualization ({algo_name}) - Separation Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (model_name, coords) in enumerate(data_dict.items()):
        # Calculate silhouette score for quantitative comparison
        score = silhouette_score(coords, mapped_labels)
        print(f"  - {algo_name} Silhouette Score for {model_name}: {score:.4f}")
        
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # Plot each actor's points individually for legend
        for label_name in unique_labels:
            mask = (mapped_labels == label_name)
            ax.scatter(
                coords[mask, 0], coords[mask, 1], coords[mask, 2],
                c=color_map[label_name], label=label_name,
                alpha=0.8, edgecolors='none', s=40
            )
            
        ax.set_title(f"{model_name} Model\n(Silhouette Score: {score:.4f})", fontsize=13, fontweight='semibold')
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.set_zlabel("Dim 3", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Sleek axis styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {algo_name} 3D visualization to: {save_path}")
    return score

def main():
    print("Step 1: Loading Tokenizer and Base Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    base_text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder").to(device)
    base_text_encoder.eval()
    
    print("Step 2: Loading LoRA Text Encoder...")
    lora_text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")
    if os.path.exists(lora_text_encoder_path):
        lora_text_encoder = PeftModel.from_pretrained(lora_text_encoder, lora_text_encoder_path)
        print("Successfully loaded LoRA weights into Text Encoder!")
    else:
        print(f"WARNING: LoRA path not found at {lora_text_encoder_path}. Using base model weights for both.")
    lora_text_encoder = lora_text_encoder.to(device)
    lora_text_encoder.eval()
    
    # Step 3: Generate Prompt Templates
    # Using combinations to create a rich variety of semantic structures
    print("Step 3: Generating prompt variations...")
    views = ["close up", "extreme close up", "macro photography", "portrait", "front view"]
    styles = ["style eyebrows", "beautiful eyebrows", "natural eyebrows", "perfect eyebrows", "delicate eyebrows", "eyebrows"]
    qualities = ["highly detailed", "photorealistic", "8k", "ultra realistic", "clear skin pores", "sharp focus"]
    lightings = ["studio lighting", "natural light", "cinematic lighting", "soft lighting", "bright sunlight", "indoor light"]
    
    raw_templates = []
    for v in views:
        for s in styles:
            for q in qualities:
                for l in lightings:
                    raw_templates.append(f"a {v} photo of {{name}} {s}, {q}, {l}")
                    
    # Sample 100 templates randomly (deterministic seed)
    random.seed(42)
    templates = random.sample(raw_templates, min(100, len(raw_templates)))
    names = ["고윤정", "신세경", "홍수주"]
    
    print(f"Generated {len(templates)} templates. Total embeddings to compute: {len(templates) * len(names)}")
    
    # Step 4: Extract Embeddings
    print("Step 4: Extracting Base Model Embeddings...")
    base_embs, labels = generate_embeddings(base_text_encoder, tokenizer, templates, names)
    
    print("Step 5: Extracting LoRA Model Embeddings...")
    lora_embs, _ = generate_embeddings(lora_text_encoder, tokenizer, templates, names)
    
    # Step 5: Perform 3D PCA
    print("Step 6: Performing 3D PCA...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    
    base_pca = pca.fit_transform(base_embs)
    lora_pca = pca.fit_transform(lora_embs)
    
    plot_3d(
        data_dict={"Base": base_pca, "LoRA": lora_pca},
        labels=labels,
        title="3D PCA Comparison",
        filename="lora_latent_space_pca.png",
        algo_name="PCA"
    )
    
    # Step 6: Perform 3D t-SNE
    print("Step 7: Performing 3D t-SNE...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
    
    # Run t-SNE
    base_tsne = tsne.fit_transform(base_embs)
    lora_tsne = tsne.fit_transform(lora_embs)
    
    plot_3d(
        data_dict={"Base": base_tsne, "LoRA": lora_tsne},
        labels=labels,
        title="3D t-SNE Comparison",
        filename="lora_latent_space_tsne.png",
        algo_name="t-SNE"
    )
    
    print("\n🎉 Visualization script finished successfully!")
    print(f"Output files can be found in: {output_dir}/")

if __name__ == "__main__":
    main()
