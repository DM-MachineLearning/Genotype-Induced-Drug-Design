# Usage: python -m Genotype_Induced_Drug_Design.PVAE.data_to_latents

import pickle
import torch
from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE
from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE

def data_to_latents(path_eg, path_dm):

    hls_dna = (8192, 4096, 2048, 1024, 512, 256)   
    hls_gene = (8192, 4096, 2048, 1024, 512, 256)
    hl_bottleneck = 256
    z_dim = 128
    n_dna_meth = 15703
    n_genes = 15703

    state = torch.load("/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results/8192-4096-2048-1024-512-256_free_bits=0.1--lamb=5.pt", map_location="cpu")

    model = PVAE(n_dna_meth, n_genes, hls_dna, hls_gene, hl_bottleneck, z_dim, use_self_attn=False)  
    model.load_state_dict(state)
    model.eval()


    with open(path_eg, 'rb') as f:
        eg_data = pickle.load(f)

    with open(path_dm, 'rb') as f:
        dm_data = pickle.load(f)

    z = model.return_latent_var(dm_data, eg_data)

    return z




def data_to_latents(path_eg, path_dm, model_path):

    input_dim = 15703
    z_dim = 128
    num_classes = 28 

    model = CNNVAE(input_dim=input_dim, z_dim=z_dim, num_classes=num_classes)
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open(path_eg, 'rb') as f:
        eg_data = pickle.load(f)

    with open(path_dm, 'rb') as f:
        dm_data = pickle.load(f)

    eg_data = eg_data.to(dtype=torch.float32)
    dm_data = dm_data.to(dtype=torch.float32)

    z = model.return_latent_var(dm_data, eg_data)

    return z

if __name__ == "__main__":
    path_eg = "/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl"
    path_dm = "/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl"
    model_path = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_model_noisy_128.pt" 

    z = data_to_latents(path_eg, path_dm, model_path)

    print(z.shape)
    print(z[0, :5])

    save_path = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/latent_representations_tcga_cnn_vae_noisy_128.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(z, f)


