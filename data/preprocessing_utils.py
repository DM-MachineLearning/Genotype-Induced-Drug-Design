import pandas as pd
import glob
import torch
import pickle
import os

def merge_csv_files(output_file, input_pattern="*.csv"):

    """Function to merge the individual csv files without the cancer tags
       output_file : path to save the merged csv file
       input_pattern : extension of the files to be merged (default : "*.csv")"""

    files = glob.glob(input_pattern)
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    merged = pd.concat(dfs, axis=1, join="inner")
    merged.to_csv(output_file)
    merged_output_df = pd.read_csv(output_file, index_col=0)

    print(f'Shape of the merged file : {merged_output_df.shape}')




def collect_and_merge(folder, suffix):     # suffix = "_DM_tcga_scale" or "_EG_tcga_scale" for DNA methylation and Gene Expression data respectively

    """Function to merge the individual csv files along with their cancer type as labels
       folder : folder path where the csv files are stored
       suffix : suffix of the csv files to be merged"""

    files = sorted(glob.glob(os.path.join(folder, f"*{suffix}.csv")))

    if not files:
        raise FileNotFoundError(f"No files matching *{suffix}.csv in {folder}")
    
    dfs = []
    sample_to_cancer = {}
    
    print(f"Reading files from {folder}...")

    for f in files:

        print(f"Reading {os.path.basename(f)}...")

        cancer = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f, index_col=0)
        for col in df.columns:
            if col in sample_to_cancer and sample_to_cancer[col] != cancer:
                print(f"Warning: sample {col} seen with cancers {sample_to_cancer[col]} and {cancer}; keeping first.")
            else:
                sample_to_cancer[col] = cancer
        dfs.append(df)
    
    print("Merging DataFrames...")
    merged = pd.concat(dfs, axis=1, join="inner")
    return merged, sample_to_cancer



def only_common_rows_cols(input_gene_exp_path, input_methylation_path, output_gene_exp_path, output_methylation_path):
    """Gene Expression Data : (n2_sample_ids, n1_genes)
       Methylation Data  : (m2_sample_ids, m1_genes)

       After filtering :-
        Gene Expression Data : (k_sample_ids, k_genes)
        Methylation Data  : (k_sample_ids, k_genes)
    """
    df1 = pd.read_csv(input_gene_exp_path, index_col=0)
    df2 = pd.read_csv(input_methylation_path, index_col=0)

    common_genes = df1.columns.intersection(df2.columns)
    common_sample_ids = df1.index.intersection(df2.index)

    df1_final = df1.loc[common_sample_ids, common_genes]
    df1_final.to_csv(output_gene_exp_path)
    print(f'Filtered Gene Expression Data Shape : {df1_final.shape}')

    df2_final = df2.loc[common_sample_ids, common_genes]
    df2_final.to_csv(output_methylation_path)
    print(f'Filtered Methylation Data Shape : {df2_final.shape}')



def save_data_to_tensor(filtered_gene_exp_path, filtered_methylation_path, output_gene_exp_tensor_path, output_methylation_tensor_path, cancer_tags_output_path):
    """
       Function to save the filtered data as torch tensors

       Gene Expression Data : (k_sample_ids, k_genes)
       Methylation Data  : (k_sample_ids, k_genes)

       Both the data contain the cancer tags as the last column for the corresponding patient/sample_ids

       Output -
        Gene Expression Tensor : torch.Tensor of shape (k_sample_ids, k_genes)
        Methylation Tensor : torch.Tensor of shape (k_sample_ids, k_genes)
        Cancer Tags Tensor : torch.Tensor of shape (k_sample_ids, C) where C is the number of unique cancer types

        Output is saved as pickle files at output paths.
    """
    df1 = pd.read_csv(filtered_gene_exp_path, index_col=0)
    df2 = pd.read_csv(filtered_methylation_path, index_col=0)

    cancer_tags = df1.iloc[:, -1].astype(str).values  
    cancer_tags_onehot = pd.get_dummies(cancer_tags)
    cancer_tags_tensor = torch.tensor(cancer_tags_onehot.values, dtype=torch.float32)

    with open(cancer_tags_output_path, 'wb') as f:
        pickle.dump(cancer_tags_tensor, f)
    print(f'Saved Cancer Tags Tensor of shape {cancer_tags_tensor.shape} to {cancer_tags_output_path}')

    gene_exp_tensor = torch.tensor(df1.drop(df1.columns[-1], axis=1).values, dtype=torch.float32)
    with open(output_gene_exp_tensor_path, 'wb') as f:
        pickle.dump(gene_exp_tensor, f)
    print(f'Saved Gene Expression Tensor of shape {gene_exp_tensor.shape} to {output_gene_exp_tensor_path}')

    methylation_tensor = torch.tensor(df2.drop(df2.columns[-1], axis=1).values, dtype=torch.float32)
    with open(output_methylation_tensor_path, 'wb') as f:
        pickle.dump(methylation_tensor, f)
    print(f'Saved Methylation Tensor of shape {methylation_tensor.shape} to {output_methylation_tensor_path}')