import torch
import numpy as np
import pickle
import sys
import os

from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised

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

TARGET_CLASS = 16          # cancer subtype index
TOP_K = 100               # number of driver genes to report
BATCH_LIMIT = 500         # Limit patients to prevent OOM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cancer type mapping
CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT"
]

CANCER_TYPE_NAME = CANCER_TYPES[TARGET_CLASS] if TARGET_CLASS < len(CANCER_TYPES) else f"Class_{TARGET_CLASS}"

GENE_CSV = "/home/dmlab/Devendra/data/tcga_transcdr/GeneExpression_data_with_cancer_filtered_final.csv"

# -----------------------------
# LOAD GENE NAMES
# -----------------------------
print(f"Loading gene names from {GENE_CSV}...")
try:
    with open(GENE_CSV, 'r') as f:
        header = f.readline().strip()
    columns = header.split(',')
    # Adjust slicing based on whether the first column is an index/empty
    gene_names = columns[1:] if columns[0] == '' else columns
    
    # Pad gene names if fewer than INPUT_DIM
    if len(gene_names) < INPUT_DIM:
        print(f"Warning: Found {len(gene_names)} names, but Input Dim is {INPUT_DIM}. Padding with placeholders.")
        gene_names += [f"Gene_{i}" for i in range(len(gene_names), INPUT_DIM)]
    print(f"Loaded {len(gene_names)} gene names")
except FileNotFoundError:
    print("CSV not found. Using dummy gene names.")
    gene_names = [f"Gene_{i}" for i in range(INPUT_DIM)]

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading Tensors...")
with open(DNA_PATH, "rb") as f:
    dna = pickle.load(f).float()

with open(GENE_PATH, "rb") as f:
    gene = pickle.load(f).float()

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

if labels.dim() > 1:
    labels = torch.argmax(labels, dim=1)
labels = labels.long()

# Robust unpacking of dataloader return
print("Creating Dataloaders...")
loaders = return_dataloaders_supervised(
    dna, gene, labels, split_fractions=(1.0, 0.0)
)
# Handle whether it returns (train, val, test) or (train, val)
train_loader = loaders[0]

# -----------------------------
# LOAD MODEL
# -----------------------------
print(f"Loading Model from {MODEL_PATH}...")
model = CNNVAE(
    input_dim=INPUT_DIM,
    z_dim=Z_DIM,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# -----------------------------
# COLLECT PATIENTS
# -----------------------------
xs_dna, xs_gene = [], []

print(f"Collecting patients for class {TARGET_CLASS}...")
for batch in train_loader:
    # Handle different batch unpacking structures
    if len(batch) == 3:
        x_d, x_g, y = batch
    elif len(batch) == 2:
        # Assuming (data, label), where data might be a tuple or stacked
        data, y = batch
        x_d, x_g = data[0], data[1] # Adjust based on actual dataloader structure
    else:
        continue

    mask = (y == TARGET_CLASS)
    if mask.any():
        xs_dna.append(x_d[mask])
        xs_gene.append(x_g[mask])

if not xs_dna:
    raise ValueError(f"No patients found for class {TARGET_CLASS}")

x_dna = torch.cat(xs_dna).to(DEVICE)
x_gene = torch.cat(xs_gene).to(DEVICE)

if BATCH_LIMIT is not None and x_dna.shape[0] > BATCH_LIMIT:
    x_dna = x_dna[:BATCH_LIMIT]
    x_gene = x_gene[:BATCH_LIMIT]

print(f"\n{'='*60}")
print(f"ANALYZING CANCER TYPE: {CANCER_TYPE_NAME}")
print(f"Using {x_dna.shape[0]} patients")
print(f"{'='*60}")

# -----------------------------
# STEP 1: LATENT GRADIENT (Direction of Cancer)
# -----------------------------
print("Computing Latent Gradients...")

with torch.enable_grad():
    # Encode
    mu, logvar = model.encoder(x_dna, x_gene)
    z = model.reparameterize(mu, logvar)
    z = z.detach().requires_grad_(True) # Detach from encoder, track for classifier

    # Classify
    logits = model.classifier_mlp(z)
    class_logit = logits[:, TARGET_CLASS].sum()

    # Get gradient: "How should z change to maximize this cancer probability?"
    grad_z = torch.autograd.grad(
        class_logit,
        z,
        create_graph=False # No need for higher order derivatives
    )[0] # [N, Z_DIM]

# Normalize grad_z to treat it as a direction unit vector per patient? 
# Usually keeping magnitude is better as it reflects confidence, 
# but for attribution, we often look at the directional derivative.
# Let's keep magnitude.

# -----------------------------
# STEP 2: GENE ATTRIBUTION via FINITE DIFFERENCES
# -----------------------------
# We want JVP: (d_Decoder / d_z) * grad_z
# This is computationally equivalent to: (Dec(z + eps*grad_z) - Dec(z)) / eps
# This replaces the slow 15,000 loop with 2 forward passes.

print("Computing Gene Attribution (Finite Differences)...")
EPSILON = 1e-4

with torch.no_grad():
    # 1. Base reconstruction
    recon_dna_base, recon_gene_base = model.decoder(z)
    
    # 2. Perturbed reconstruction (move z in direction of cancer class)
    z_perturbed = z + (EPSILON * grad_z)
    recon_dna_pert, recon_gene_pert = model.decoder(z_perturbed)
    
    # 3. Calculate finite difference (Directional Derivative)
    # [N, INPUT_DIM]
    delta_dna = (recon_dna_pert - recon_dna_base) / EPSILON
    delta_gene = (recon_gene_pert - recon_gene_base) / EPSILON

# -----------------------------
# STEP 3: AGGREGATE
# -----------------------------

# Mean attribution score across all patients of this class
# High positive score = Gene value INCREASES as we move towards cancer class
# High negative score = Gene value DECREASES as we move towards cancer class
# We usually care about magnitude (drivers)
dna_attr_mean = delta_dna.mean(dim=0).abs()   # [INPUT_DIM]
gene_attr_mean = delta_gene.mean(dim=0).abs() # [INPUT_DIM]

# Combine scores (L2 norm of the two modalities)
driver_score = torch.sqrt(dna_attr_mean**2 + gene_attr_mean**2)

# Convert to numpy for saving
dna_scores_np = dna_attr_mean.cpu().numpy()
gene_scores_np = gene_attr_mean.cpu().numpy()
driver_scores_np = driver_score.cpu().numpy()

# -----------------------------
# SAVE RESULTS
# -----------------------------

np.save("driver_scores.npy", driver_scores_np)
np.save("expr_scores.npy", gene_scores_np)
np.save("meth_scores.npy", dna_scores_np)

# -----------------------------
# REPORT TOP GENES
# -----------------------------

topk = torch.topk(driver_score, TOP_K)

print(f"\nTop {TOP_K} Driver Genes for {CANCER_TYPE_NAME}:")
print(f"{'Rank':<6} {'Gene Name':<20} {'Gene Index':<12} {'Combined':<12} {'Expr Attr':<12} {'Meth Attr':<12}")
print("-" * 90)

for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), 1):
    idx_int = int(idx.item())
    
    # Safety check for gene names
    if idx_int < len(gene_names):
        gene_name = gene_names[idx_int]
    else:
        gene_name = f"Unknown_{idx_int}"
        
    print(
        f"{rank:<6} {gene_name:<20} {idx_int:<12} "
        f"{score.item():<12.6f} "
        f"{gene_scores_np[idx_int]:<12.6f} "
        f"{dna_scores_np[idx_int]:<12.6f}"
    )

print("\nProcessing Complete.")