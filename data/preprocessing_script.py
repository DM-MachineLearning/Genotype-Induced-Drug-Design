# Usage : python -m Genotype_Induced_Drug_Design.data.preprocessing_script

import numpy as np
import pandas as pd
import torch
import pickle
from Genotype_Induced_Drug_Design.data.preprocessing_utils import only_common_rows_cols, save_data_to_tensor

input_gene_exp_path = '/home/dmlab/Devendra/data/tcga_transcdr/merged_outputs/merged_EG_with_cancer.csv'
input_methylation_path = '/home/dmlab/Devendra/data/tcga_transcdr/merged_outputs/merged_DM_with_cancer.csv'
output_gene_exp_path = '/home/dmlab/Devendra/data/preprocessed_datasets/filtered_gene_expression_tcga.csv'
output_methylation_path = '/home/dmlab/Devendra/data/preprocessed_datasets/filtered_methylation_tcga.csv'
output_gene_exp_tensor_path = '/home/dmlab/Devendra/data/preprocessed_datasets/gene_expression_tensor_tcga.pkl'
output_methylation_tensor_path = '/home/dmlab/Devendra/data/preprocessed_datasets/methylation_tensor_tcga.pkl'
output_cancer_tags_tensor_path = '/home/dmlab/Devendra/data/preprocessed_datasets/cancer_tags_tensor_tcga.pkl'

# Already have the merged files with cancer tags, if required use collect_and_merge from Genotype_Induced_Drug_Design.data.preprocessing_utils
# Make sure to transpose the merged dataframe - Final output should be of the form (n_samples, m_genes)

only_common_rows_cols(input_gene_exp_path, input_methylation_path, output_gene_exp_path, output_methylation_path)

df = pd.read_csv(output_gene_exp_path, index_col=0)
print(df.iloc[[0,1,2,3,4],[0,1,2,3,4,-1]])

save_data_to_tensor(output_gene_exp_path, output_methylation_path, output_gene_exp_tensor_path, output_methylation_tensor_path, output_cancer_tags_tensor_path)

with open(output_gene_exp_tensor_path, 'rb') as f:
    gene_exp_tensor = pickle.load(f)
with open(output_methylation_tensor_path, 'rb') as f:
    methylation_tensor = pickle.load(f)
with open(output_cancer_tags_tensor_path, 'rb') as f:
    cancer_tags_tensor = pickle.load(f)

print(f'Gene Expression Tensor Shape : {gene_exp_tensor.shape}')
print(f'Methylation Tensor Shape : {methylation_tensor.shape}')
print(f'Cancer Tags Tensor Shape : {cancer_tags_tensor.shape}')