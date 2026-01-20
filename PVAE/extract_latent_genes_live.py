import torch
import numpy as np
import pickle
import sys
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128.pt"

DNA_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl"
GENE_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl"
LABEL_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl"

INPUT_DIM = 15703
Z_DIM = 128
NUM_CLASSES = 28

TARGET_CLASS = 2          # Cancer subtype index
TOP_K = 100               
CALC_BATCH_SIZE = 64      # Prevents OOM
EPSILON = 1e-4            # Step for finite difference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your specific model class
from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE

# -----------------------------
# 1. LOAD DATA DIRECTLY (TOTAL COVERAGE)
# -----------------------------
print("Loading Tensors for 100% patient coverage...")
with open(DNA_PATH, "rb") as f:
    dna_raw = pickle.load(f).float()
with open(GENE_PATH, "rb") as f:
    gene_raw = pickle.load(f).float()
with open(LABEL_PATH, "rb") as f:
    labels_raw = pickle.load(f)

if labels_raw.dim() > 1:
    labels_raw = torch.argmax(labels_raw, dim=1)
labels_raw = labels_raw.long()

# Extract every patient belonging to the target class
target_indices = (labels_raw == TARGET_CLASS).nonzero(as_tuple=True)[0]
num_patients = len(target_indices)

if num_patients == 0:
    raise ValueError(f"No patients found for class {TARGET_CLASS}")

x_dna_all = dna_raw[target_indices]
x_gene_all = gene_raw[target_indices]

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
model = CNNVAE(input_dim=INPUT_DIM, z_dim=Z_DIM, num_classes=NUM_CLASSES)
# Note: map_location handles CPU/GPU switching
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE).eval()

# -----------------------------
# 3. COMPUTE DRIVER ATTRIBUTION
# -----------------------------
all_dna_attrs = []
all_gene_attrs = []

print(f"Analyzing {num_patients} patients in batches of {CALC_BATCH_SIZE}...")

for i in range(0, num_patients, CALC_BATCH_SIZE):
    end_idx = min(i + CALC_BATCH_SIZE, num_patients)
    b_dna = x_dna_all[i:end_idx].to(DEVICE)
    b_gene = x_gene_all[i:end_idx].to(DEVICE)
    
    # Step A: Get Gradient of Classifier w.r.t Latent Space (d_Class / d_Z)
    with torch.enable_grad():
        mu, logvar = model.encoder(b_dna, b_gene) # encoder handles 2-channel stack
        z = model.reparameterize(mu, logvar)
        z = z.detach().requires_grad_(True)
        
        logits = model.classifier_mlp(z)
        class_logit = logits[:, TARGET_CLASS].sum()
        grad_z = torch.autograd.grad(class_logit, z)[0] 

    # Step B: Finite Difference for Input Attribution
    # This identifies which input features drive the latent space toward the target class
    with torch.no_grad():
        # Baseline
        recon_dna_base, recon_gene_base = model.decoder(z)
        # Perturbed in direction of target class
        z_perturbed = z + (EPSILON * grad_z)
        recon_dna_pert, recon_gene_pert = model.decoder(z_perturbed)
        
        # Calculate Delta (Attribution)
        all_dna_attrs.append(((recon_dna_pert - recon_dna_base) / EPSILON).cpu())
        all_gene_attrs.append(((recon_gene_pert - recon_gene_base) / EPSILON).cpu())
        
    sys.stdout.write(f"\rProgress: {end_idx}/{num_patients}")
    sys.stdout.flush()

# -----------------------------
# 4. AGGREGATE & SAVE
# -----------------------------
dna_scores = torch.cat(all_dna_attrs, dim=0).mean(dim=0).abs()
gene_scores = torch.cat(all_gene_attrs, dim=0).mean(dim=0).abs()

# Combined magnitude across both genomic modalities
driver_score = gene_scores

np.save("driver_scores.npy", driver_score.numpy())
np.save("expr_scores.npy", gene_scores.numpy())
np.save("meth_scores.npy", dna_scores.numpy())

# -----------------------------
# 5. REPORT TOP GENES
# -----------------------------
# Attempt to load gene names
try:
    GENE_CSV = "/home/dmlab/Devendra/data/tcga_transcdr/GeneExpression_data_with_cancer_filtered_final.csv"
    with open(GENE_CSV, 'r') as f:
        header = f.readline().strip().split(',')
    gene_names = header[1:] if header[0] == '' else header
except:
    gene_names = [f"Gene_{i}" for i in range(INPUT_DIM)]

topk_vals, topk_idx = torch.topk(driver_score, TOP_K)

print(f"\n\nTop {TOP_K} Driver Genes for Class {TARGET_CLASS}:")
print(f"{'Rank':<6} {'Gene Name':<20} {'Driver Score':<15} {'Expr Score':<12} {'Meth Score':<12}")
print("-" * 75)

for rank, (idx, score) in enumerate(zip(topk_idx, topk_vals), 1):
    idx_int = int(idx.item())
    name = gene_names[idx_int] if idx_int < len(gene_names) else f"Idx_{idx_int}"
    print(f"{rank:<6} {name:<20} {score.item():<15.6f} {gene_scores[idx_int]:<12.6f} {dna_scores[idx_int]:<12.6f}")