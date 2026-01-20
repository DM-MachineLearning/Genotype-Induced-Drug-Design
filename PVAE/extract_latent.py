# Usage: python -m Genotype_Induced_Drug_Design.PVAE.extract_latent

import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset

from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE

MODEL_PATH = "Genotype_Induced_Drug_Design/PVAE/results/best_finetuned_model.pt"
OUTPUT_PATH = "Genotype_Induced_Drug_Design/PVAE/results/latent_mu.pkl"

BATCH_SIZE = 64
USE_SELF_ATTN = False          # must match best model
Z_DIM = 128
HL_BOTTLENECK = 256

HLS_DNA = (2048, 512, 256)
HLS_GENE = (2048, 512, 256)


def main():

    print("Starting latent extraction script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print("Loading DNA methylation data...")
    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_dm_tcga_tensor1.pkl", "rb") as f:
        dna_meth = pickle.load(f).float()
    print(f"DNA methylation data loaded. Shape: {dna_meth.shape}")

    print("Loading gene expression data...")
    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_eg_tcga_tensor2.pkl", "rb") as f:
        gene_exp = pickle.load(f).float()
    print(f"Gene expression data loaded. Shape: {gene_exp.shape}")

    dataset = TensorDataset(dna_meth, gene_exp)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"DataLoader created with {len(loader)} batches (batch size = {BATCH_SIZE})")

    
    print("Building PVAE model...")
    model = PVAE(
        input_dim_dna=dna_meth.shape[1],
        input_dim_gene=gene_exp.shape[1],
        hls_dna=HLS_DNA,
        hls_gene=HLS_GENE,
        hl_bottleneck=HL_BOTTLENECK,
        z_dim=Z_DIM,
        use_self_attn=USE_SELF_ATTN,
    )

    
    print(f"Loading trained model weights from:\n   {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode")

   
    print("Extracting latent variables (μ)...")
    latents = []

    with torch.no_grad():
        for batch_idx, (x_dna, x_gene) in enumerate(loader, start=1):
            x_dna = x_dna.to(device)
            x_gene = x_gene.to(device)

            mu, _ = model.encoder(x_dna, x_gene)
            latents.append(mu.cpu())

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"   Processed batch {batch_idx}/{len(loader)}")

    latents = torch.cat(latents, dim=0)

    print(f"Latent extraction complete. Latent shape: {latents.shape}")

   
    print(f"Saving latent vectors to:\n   {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(latents.numpy(), f)

    print("Latent vectors successfully saved!")
    print("Script finished successfully.")


if __name__ == "__main__":
    main()
