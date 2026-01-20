"""
Aggregate per-gene importance (Integrated Gradients) per cancer class.
Outputs top 30 genes per class to JSON.

Usage:
  python -m Genotype_Induced_Drug_Design.PVAE.interpretability.run_gradcam
"""

import json
import torch
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from Genotype_Induced_Drug_Design.PVAE.MSE_CNN_VAE import MSECNNVAE
from Genotype_Induced_Drug_Design.PVAE.interpretability.gradcamm import IntegratedGradients


def main():
    # --- Config ---
    MODEL_PATH = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128_mask_off_mu_mse.pt"
    DNA_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl"
    GENE_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl"
    LABELS_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl"
    GENE_INDEX_PATH = "/home/dmlab/Devendra/gene_index_mapping.json"
    CANCER_CLASS_PATH = "/home/dmlab/Devendra/cancer_class_mapping.json"
    OUTPUT_JSON = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/interpretability/top_genes_per_cancer.json"

    INPUT_DIM = 15703
    Z_DIM = 128
    TOP_K = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load mappings ---
    with open(GENE_INDEX_PATH, "r") as f:
        gene_index_map = json.load(f)  # {"0": "A1BG", ...}
    with open(CANCER_CLASS_PATH, "r") as f:
        cancer_class_map = json.load(f)  # {"0": "ACC", ...}

    # --- Load data ---
    print("Loading data...")
    with open(DNA_PATH, "rb") as f:
        dna_meth = pickle.load(f)
    with open(GENE_PATH, "rb") as f:
        gene_exp = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)

    # Handle one-hot labels
    if labels.dim() > 1 and labels.shape[1] > 1:
        NUM_CLASSES = labels.shape[1]
        labels = torch.argmax(labels, dim=1)
    else:
        NUM_CLASSES = len(torch.unique(labels))

    print(f"Data loaded: {dna_meth.shape[0]} samples, {NUM_CLASSES} classes")

    # --- Load model ---
    print("Loading model...")
    model = MSECNNVAE(input_dim=INPUT_DIM, z_dim=Z_DIM, num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # --- Initialize Integrated Gradients ---
    ig = IntegratedGradients(model, n_steps=30)

    # --- Aggregate attributions per cancer class ---
    # We'll sum importance scores for each class and count samples
    attr_sums = defaultdict(lambda: np.zeros(INPUT_DIM, dtype=np.float64))
    attr_counts = defaultdict(int)

    print("\nComputing Integrated Gradients for all samples (aggregating per cancer class)...")
    n_samples = dna_meth.shape[0]

    for i in tqdm(range(n_samples), desc="Processing samples"):
        sample_dna = dna_meth[i:i+1].float()
        sample_gene = gene_exp[i:i+1].float()
        class_idx = int(labels[i].item())

        # Compute attribution for the true class
        attr = ig.attribute(sample_dna, sample_gene, target_class=class_idx)
        attr_sums[class_idx] += attr[0]
        attr_counts[class_idx] += 1

    # --- Compute mean attribution per class and find top genes ---
    print("\nComputing top genes per cancer class...")
    results = {}

    for class_idx in sorted(attr_sums.keys()):
        mean_attr = attr_sums[class_idx] / attr_counts[class_idx]

        # Get top-k indices
        top_indices = mean_attr.argsort()[-TOP_K:][::-1]

        # Map indices to gene names
        top_genes = []
        for idx in top_indices:
            gene_name = gene_index_map.get(str(idx), f"UNKNOWN_{idx}")
            importance_score = float(mean_attr[idx])
            top_genes.append({
                "rank": len(top_genes) + 1,
                "gene_index": int(idx),
                "gene_name": gene_name,
                "importance_score": round(importance_score, 6)
            })

        cancer_name = cancer_class_map.get(str(class_idx), f"CLASS_{class_idx}")
        results[cancer_name] = {
            "class_index": class_idx,
            "num_samples": attr_counts[class_idx],
            "top_genes": top_genes
        }

        print(f"  {cancer_name} ({attr_counts[class_idx]} samples): top gene = {top_genes[0]['gene_name']}")

    # --- Save JSON ---
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_JSON}")
    print("Done!")


if __name__ == "__main__":
    main()
