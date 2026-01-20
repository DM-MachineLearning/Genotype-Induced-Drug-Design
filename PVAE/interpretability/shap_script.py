# python -m Genotype_Induced_Drug_Design.PVAE.interpretability.shap_script

import numpy as np
import torch
import shap
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from Genotype_Induced_Drug_Design.PVAE.MSE_CNN_VAE import MSECNNVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised

MODEL_PATH = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128_mask_off_mu.pt"
DATA_DIR = "/home/dmlab/Devendra/data/preprocessed_datasets/"
INPUT_DIM = 15703
Z_DIM = 128
BACKGROUND_SAMPLES = 100
TEST_SAMPLES = 10
TARGET_CLASS = 2  # Which class to analyze

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShapWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ShapWrapper, self).__init__()
        self.model = original_model

    def forward(self, input_dna, input_gene):
        _, _, _, _, class_logits = self.model.forward_with_classifier(input_dna, input_gene)
        return class_logits

def load_data():
    print("Loading data...")
    with open(f"{DATA_DIR}/methylation_tensor_tcga.pkl", "rb") as f:
        dna_meth = pickle.load(f)

    with open(f"{DATA_DIR}/gene_expression_tensor_tcga.pkl", "rb") as f:
        gene_exp = pickle.load(f)

    try:
        with open(f"{DATA_DIR}/cancer_tags_tensor_tcga.pkl", "rb") as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        print("Labels file not found, creating dummy labels.")
        labels = torch.randint(0, 2, (len(dna_meth),))

    if labels.dim() > 1 and labels.shape[1] > 1:
        num_classes = labels.shape[1]
        labels = torch.argmax(labels, dim=1)
    else:
        num_classes = len(torch.unique(labels))

    dna_meth = dna_meth.to(dtype=torch.float32)
    gene_exp = gene_exp.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)

    return dna_meth, gene_exp, labels, num_classes

def main():
    dna_meth, gene_exp, labels, num_classes = load_data()

    train_loader, _, test_loader = return_dataloaders_supervised(
        dna_meth,
        gene_exp,
        labels,
        split_fractions=(0.8, 0.1)
    )

    print(f"Building model with input_dim={INPUT_DIM}, z_dim={Z_DIM}, classes={num_classes}")
    original_model = MSECNNVAE(
        input_dim=INPUT_DIM,
        z_dim=Z_DIM,
        num_classes=num_classes
    )

    print(f"Loading weights from {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        original_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Standard load failed, trying pickle load if entire object was saved: {e}")
        original_model = torch.load(MODEL_PATH, map_location=device)

    original_model.to(device)
    original_model.eval()

    model_shap = ShapWrapper(original_model).to(device)
    model_shap.eval()

    bg_dna, bg_gene, _ = next(iter(train_loader))
    bg_dna = bg_dna[:BACKGROUND_SAMPLES].to(device)
    bg_gene = bg_gene[:BACKGROUND_SAMPLES].to(device)

    test_dna, test_gene, test_labels = next(iter(test_loader))
    test_dna = test_dna[:TEST_SAMPLES].to(device)
    test_gene = test_gene[:TEST_SAMPLES].to(device)

    print(f"Background samples: {bg_dna.shape[0]}")
    print(f"Test samples to explain: {test_dna.shape[0]}")

    explainer = shap.DeepExplainer(model_shap, [bg_dna, bg_gene])

    print("Computing SHAP values (this may take time)...")
    shap_values = explainer.shap_values([test_dna, test_gene], check_additivity=True)

    print("SHAP values computed.")

    # DeepExplainer with multi-class output returns:
    # shap_values[class_idx][input_idx] with shape (samples, features)
    # OR shap_values[input_idx] with shape (samples, features, num_classes)
    
    class_idx = TARGET_CLASS
    
    if isinstance(shap_values, list) and isinstance(shap_values[0], list):
        # Structure: shap_values[num_classes][num_inputs]
        print(f"Output is nested list: {len(shap_values)} classes x {len(shap_values[0])} inputs")
        dna_shap_vals = shap_values[class_idx][0]  # (samples, features) for DNA
        gene_shap_vals = shap_values[class_idx][1]  # (samples, features) for Gene
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        # Structure: shap_values[num_inputs] with shape (samples, features, num_classes)
        print(f"Output is list of 2 arrays (one per input)")
        if shap_values[0].ndim == 3:
            dna_shap_vals = shap_values[0][:, :, class_idx]
            gene_shap_vals = shap_values[1][:, :, class_idx]
        else:
            # Single class or already 2D
            dna_shap_vals = shap_values[0]
            gene_shap_vals = shap_values[1]
    else:
        raise ValueError(f"Unexpected shap_values structure: {type(shap_values)}")

    if torch.is_tensor(dna_shap_vals):
        dna_shap_vals = dna_shap_vals.cpu().detach().numpy()
        gene_shap_vals = gene_shap_vals.cpu().detach().numpy()
    elif not isinstance(dna_shap_vals, np.ndarray):
        dna_shap_vals = np.array(dna_shap_vals)
        gene_shap_vals = np.array(gene_shap_vals)

    test_dna_np = test_dna.cpu().detach().numpy()
    test_gene_np = test_gene.cpu().detach().numpy()

    print(dna_shap_vals.shape, test_dna_np.shape)
    print(gene_shap_vals.shape, test_gene_np.shape)

    print("\n--- Generating Plots ---")

    plt.figure()
    plt.title(f"SHAP Summary: DNA Methylation (Class {class_idx})")
    shap.summary_plot(dna_shap_vals, test_dna_np, feature_names=None, show=False)
    plt.savefig("shap_summary_dna_class_idx_2_mask_off_mu_mse_updated.png")
    print("Saved shap_summary_dna_class_idx_2_mask_off_mu_mse_updated.png")
    plt.close()

    plt.figure()
    plt.title(f"SHAP Summary: Gene Expression (Class {class_idx})")
    shap.summary_plot(gene_shap_vals, test_gene_np, feature_names=None, show=False)
    plt.savefig("shap_summary_gene_class_idx_2_mask_off_mu_mse_updated.png")
    print("Saved shap_summary_gene_class_idx_2_mask_off_mu_mse_updated.png")
    plt.close()

    with open("shap_values_output_class_idx_2_mask_off_mu_mse_updated.pkl", "wb") as f:
        pickle.dump(shap_values, f)
    print("Saved raw shap values to shap_values_output_class_idx_2_mask_off_mu_mse_updated.pkl")

if __name__ == "__main__":
    main()