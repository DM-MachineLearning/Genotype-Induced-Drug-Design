# import torch
# import numpy as np
# from tqdm import tqdm
# from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE
# from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised
# import pickle


# def load_model_and_data(model_path, device):

#     # Load tensors
#     with open("/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl", "rb") as f:
#         dna = pickle.load(f)

#     with open("/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl", "rb") as f:
#         gene = pickle.load(f)

#     with open("/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl", "rb") as f:
#         labels = pickle.load(f)

#     if labels.dim() > 1:
#         labels = torch.argmax(labels, dim=1)

#     # Ensure numpy arrays are converted to torch tensors
#     if not torch.is_tensor(dna):
#         dna = torch.tensor(dna, dtype=torch.float32)
#     if not torch.is_tensor(gene):
#         gene = torch.tensor(gene, dtype=torch.float32)
#     if not torch.is_tensor(labels):
#         labels = torch.tensor(labels, dtype=torch.long)

#     train_loader, val_loader, test_loader = return_dataloaders_supervised(
#         dna.float(),
#         gene.float(),
#         labels.long(),
#         split_fractions=(0.8, 0.1)
#     )

#     model = CNNVAE(
#         input_dim=gene.shape[1],
#         z_dim=128,
#         num_classes=len(torch.unique(labels))
#     )

#     # Load checkpoint and handle common checkpoint dict formats
#     import os
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
#     print(f"Loading model from: {model_path}")
#     ckpt = torch.load(model_path, map_location=device)

#     # common keys used when saving checkpoints
#     possible_keys = [
#         "state_dict",
#         "model_state_dict",
#         "weights",
#         "params",
#         "model",
#     ]

#     state = None
#     if isinstance(ckpt, dict):
#         for k in possible_keys:
#             if k in ckpt:
#                 state = ckpt[k]
#                 print(f"Found checkpoint under key: {k}")
#                 break

#     if state is None:
#         # If no standard key is found, assume the entire checkpoint is the state_dict
#         state = ckpt

#     try:
#         model.load_state_dict(state)
#         print("Model loaded successfully with strict=True")
#     except RuntimeError as e:
#         try:
#             model.load_state_dict(state, strict=False)
#             print(f"Model loaded with strict=False (some keys may be missing)")
#         except Exception as e2:
#             # as a last resort, if the checkpoint was a full Module, use it
#             if isinstance(ckpt, torch.nn.Module):
#                 print("Using checkpoint as full module")
#                 model = ckpt
#             else:
#                 raise ValueError(f"Could not load model from checkpoint. Error: {str(e2)}")
    
#     model.to(device)
#     model.eval()
#     print(f"Model is on device: {next(model.parameters()).device}")

#     return model, test_loader



# @torch.no_grad()
# def evaluate(model, loader, device):
#     correct, total = 0, 0
#     for x_dna, x_gene, y in loader:
#         x_dna = x_dna.to(device)
#         x_gene = x_gene.to(device)
#         y = y.to(device)

#         _, _, _, _, logits = model.forward_with_classifier(x_dna, x_gene)
#         preds = torch.argmax(logits, dim=1)

#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     return correct / total



# @torch.no_grad()
# def joint_gene_knockout(
#     model,
#     dataloader,
#     device,
#     mode="zero"   # "zero" or "mean"
# ):
#     model.eval()

#     baseline_acc = evaluate(model, dataloader, device)
#     print(f"\nBaseline accuracy: {baseline_acc:.4f}")

#     num_genes = next(iter(dataloader))[1].shape[1]
#     importance = np.zeros(num_genes)

#     # Compute gene means (for mean knockout)
#     if mode == "mean":
#         gene_mean = torch.zeros(num_genes)
#         dna_mean = torch.zeros(num_genes)
#         n = 0
#         for x_dna, x_gene, _ in dataloader:
#             gene_mean += x_gene.sum(dim=0)
#             dna_mean += x_dna.sum(dim=0)
#             n += x_gene.size(0)

#         gene_mean /= n
#         dna_mean /= n
#         gene_mean = gene_mean.to(device)
#         dna_mean = dna_mean.to(device)

#     for g in tqdm(range(num_genes), desc="Knocking out genes"):
#         correct, total = 0, 0

#         for x_dna, x_gene, y in dataloader:
#             x_dna = x_dna.clone().to(device)
#             x_gene = x_gene.clone().to(device)
#             y = y.to(device)

#             if mode == "zero":
#                 x_dna[:, g] = 0.0
#                 x_gene[:, g] = 0.0
#             else:
#                 x_dna[:, g] = dna_mean[g]
#                 x_gene[:, g] = gene_mean[g]

#             _, _, _, _, logits = model.forward_with_classifier(x_dna, x_gene)
#             preds = torch.argmax(logits, dim=1)

#             correct += (preds == y).sum().item()
#             total += y.size(0)

#         acc = correct / total
#         importance[g] = baseline_acc - acc

#     return importance



# if __name__ == "__main__":

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_path = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128.pt"
    
#     model, test_loader = load_model_and_data(model_path, device)

#     importance = joint_gene_knockout(
#         model,
#         test_loader,
#         device,
#         mode="zero"   # or "zero"
#     )

#     np.save("gene_knockout_importance.npy", importance)

#     # Top genes
#     top = np.argsort(importance)[::-1][:20]
#     print("\nTop important genes:")
#     for i in top:
#         print(f"Gene {i}: ΔAcc = {importance[i]:.5f}")



import torch
import numpy as np
from tqdm import tqdm
import pickle
import os

from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised


# ============================================================
# LOAD MODEL + DATA
# ============================================================
def load_model_and_data(model_path, device):

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl", "rb") as f:
        dna = pickle.load(f)

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl", "rb") as f:
        gene = pickle.load(f)

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl", "rb") as f:
        labels = pickle.load(f)

    if labels.dim() > 1:
        labels = torch.argmax(labels, dim=1)

    dna = torch.tensor(dna, dtype=torch.float32)
    gene = torch.tensor(gene, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Move tensors to device so DataLoader yields GPU tensors and
    # we avoid repeated host->device copies during iteration.
    dna = dna.to(device)
    gene = gene.to(device)
    labels = labels.to(device)

    train_loader, val_loader, test_loader = return_dataloaders_supervised(
        dna, gene, labels, split_fractions=(0.8, 0.1)
    )

    model = CNNVAE(
        input_dim=gene.shape[1],
        z_dim=128,
        num_classes=len(torch.unique(labels))
    )

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt:
                ckpt = ckpt[k]
                break

    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("✅ Model loaded")
    return model, test_loader


# ============================================================
# PER-CANCER GENE KNOCKOUT
# ============================================================
@torch.no_grad()
def gene_knockout_per_cancer(model, dataloader, device, mode="zero"):

    all_labels = []
    for _, _, y in dataloader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels)

    cancer_types = torch.unique(all_labels)
    num_genes = next(iter(dataloader))[1].shape[1]

    importance = {
        int(c.item()): np.zeros(num_genes)
        for c in cancer_types
    }

    # Mean values if needed
    if mode == "mean":
        # Create accumulators on device to avoid extra transfers.
        gene_mean = torch.zeros(num_genes, device=device)
        dna_mean = torch.zeros(num_genes, device=device)
        n = 0
        for x_dna, x_gene, _ in dataloader:
            x_gene = x_gene.to(device)
            x_dna = x_dna.to(device)
            gene_mean += x_gene.sum(dim=0)
            dna_mean += x_dna.sum(dim=0)
            n += x_gene.size(0)
        if n > 0:
            gene_mean /= n
            dna_mean /= n

    for cancer in cancer_types:
        cancer = int(cancer)
        print(f"\n🧬 Processing cancer class {cancer}")

        # ---- baseline accuracy ----
        correct, total = 0, 0
        for x_dna, x_gene, y in dataloader:
            mask = (y == cancer)
            if mask.sum() == 0:
                continue

            x_dna = x_dna[mask].to(device)
            x_gene = x_gene[mask].to(device)
            y = y[mask].to(device)

            _, _, _, _, logits = model.forward_with_classifier(x_dna, x_gene)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

        if total == 0:
            print(f"  ⚠️ No samples for cancer {cancer}, skipping...")
            continue
        
        baseline_acc = correct / total
        print(f"  Baseline acc: {baseline_acc:.4f}")

        # ---- gene knockout ----
        for g in tqdm(range(num_genes), desc=f"Cancer {cancer}"):

            correct, total = 0, 0

            for x_dna, x_gene, y in dataloader:
                mask = (y == cancer)
                if mask.sum() == 0:
                    continue

                x_dna = x_dna[mask].clone().to(device)
                x_gene = x_gene[mask].clone().to(device)
                y = y[mask].to(device)

                if mode == "zero":
                    x_dna[:, g] = 0
                    x_gene[:, g] = 0
                elif mode == "mean":
                    x_dna[:, g] = dna_mean[g]
                    x_gene[:, g] = gene_mean[g]

                _, _, _, _, logits = model.forward_with_classifier(x_dna, x_gene)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

            if total > 0:
                acc = correct / total
                importance[cancer][g] = baseline_acc - acc

    return importance


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_path = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128.pt"

    model, test_loader = load_model_and_data(model_path, device)

    importance = gene_knockout_per_cancer(
        model,
        test_loader,
        device,
        mode="zero"
    )

    # Save results
    with open("gene_importance_per_cancer.pkl", "wb") as f:
        pickle.dump(importance, f)
    print("✅ Results saved to gene_importance_per_cancer.pkl")

    # Print top genes per cancer
    for cancer, scores in importance.items():
        print(f"\n🔥 Top genes for cancer {cancer}")
        top = np.argsort(scores)[::-1][:10]
        for g in top:
            print(f"  Gene {g} → ΔAcc = {scores[g]:.4f}")