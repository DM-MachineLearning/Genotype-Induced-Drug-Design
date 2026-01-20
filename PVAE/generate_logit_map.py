# Usage : python -m Genotype_Induced_Drug_Design.PVAE.generate_logit_map

import torch
import pickle
import numpy as np
from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE 

LABEL_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl"
MODEL_PATH = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128_mask_off.pt"
INPUT_DIM = 15703
NUM_CLASSES = 28  

def get_representative_samples():
    print(f"Loading labels from {LABEL_PATH}...")
    with open(LABEL_PATH, "rb") as f:
        labels = pickle.load(f)

    if labels.dim() > 1 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)
    
    representatives = {}       # { Class_Index : Sample_Index_in_Dataset }
    

    print("Scanning dataset for representative samples...")
    for i in range(len(labels)):
        current_class = labels[i].item()
        if current_class not in representatives:
            representatives[current_class] = i
            
        if len(representatives) == NUM_CLASSES:
            break
            
    sorted_reps = dict(sorted(representatives.items()))
    return sorted_reps, labels

def main():
    # 1. Find Representative Indices
    rep_map, all_labels = get_representative_samples()
    
    print("\n--- Representative Samples Found ---")
    print(f"{'Logit/Class':<12} | {'Sample Index':<12}")
    print("-" * 30)
    for class_idx, sample_idx in rep_map.items():
        print(f"{class_idx:<12} | {sample_idx:<12}")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR MAPPING")
    print("="*60)
    print("Since your tensor files usually don't contain strings (e.g., 'BRCA'),")
    print("you must look up the representative Sample Indices above in your")
    print("original CSV/Excel/Clinical data file to see the string name.\n")
    
    print("Once you know that Sample", rep_map[0], "is 'ACC' (for example),")
    print("you can fill in the function below.\n")


    print("Verifying Model Alignment...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        state_dict = torch.load(MODEL_PATH, map_location=device)
        model = CNNVAE(input_dim=INPUT_DIM, z_dim=128, num_classes=NUM_CLASSES)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()



        with open("/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl", "rb") as f:
            dna = pickle.load(f)
        with open("/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl", "rb") as f:
            gene = pickle.load(f)

        print("\n--- Model Verification ---")
        for class_idx, sample_idx in rep_map.items():

            x_d = dna[sample_idx].unsqueeze(0).to(device).float()
            x_g = gene[sample_idx].unsqueeze(0).to(device).float()
            

            _, _, _, _, logits = model.forward_with_classifier(x_d, x_g)
            predicted_logit = torch.argmax(logits, dim=1).item()
            
            status = "MATCH" if predicted_logit == class_idx else "MISMATCH"
            print(f"Sample {sample_idx}: Label={class_idx} -> Pred={predicted_logit} [{status}]")

    except Exception as e:
        print(f"Could not verify with model (skipping): {e}")

if __name__ == "__main__":
    main()