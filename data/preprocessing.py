import pandas as pd
import glob
import torch
import pickle
import os

def merge_csv_files(output_file, input_pattern="*.csv"):

    files = glob.glob(input_pattern)
    dfs = [pd.read_csv(f, index_col=0) for f in files]
    merged = pd.concat(dfs, axis=1, join="inner")
    merged.to_csv(output_file)
    merged_output_df = pd.read_csv(output_file, index_col=0)

    print(f'Shape of the merged file : {merged_output_df.shape}')



def collect_and_merge(folder, suffix):
    """
    """

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




def save_data_with_cancer_labels():
    """
    Merges per-cancer CSVs and appends cancer labels.
    """

    DM_DIR = "data/tcga_transcdr/processed_dm_tcga_scale_files"
    EG_DIR = "data/tcga_transcdr/processed_EG_tcga_scale_files"
    OUT_DIR = "data/tcga_transcdr/merged_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    def collect_and_merge(folder, suffix):
        files = sorted(glob.glob(os.path.join(folder, f"*{suffix}.csv")))
        if not files:
            raise FileNotFoundError(f"No files matching *{suffix}.csv in {folder}")

        dfs = []
        sample_to_cancer = {}

        for f in files:
            cancer = os.path.basename(f).split("_")[0]
            df = pd.read_csv(f, index_col=0)
            for col in df.columns:
                sample_to_cancer.setdefault(col, cancer)
            dfs.append(df)


        merged = pd.concat(dfs, axis=1, join="inner")
        return merged, sample_to_cancer

    merged_dm, meta_dm = collect_and_merge(DM_DIR, "_DM_tcga_scale")
    merged_eg, meta_eg = collect_and_merge(EG_DIR, "_EG_tcga_scale")

    merged_dm = merged_dm.T
    merged_dm["cancer"] = merged_dm.index.map(meta_dm)

    merged_eg = merged_eg.T
    merged_eg["cancer"] = merged_eg.index.map(meta_eg)

    merged_dm.to_csv(os.path.join(OUT_DIR, "merged_DM_with_cancer.csv"))
    merged_eg.to_csv(os.path.join(OUT_DIR, "merged_EG_with_cancer.csv"))

    print("Saved merged DM and EG files with cancer labels")




def only_common_rows_cols(input_gene_exp_path, input_methylation_path, output_gene_exp_path, output_methylation_path):
    """Gene Expression Data : (n1_genes, n2_sample_ids)
       Methylation Data  : (m1_genes, m2_sample_ids)

       After filtering :-
        Gene Expression Data : (k_genes, k_sample_ids)
        Methylation Data  : (k_genes, k_sample_ids)
    """
    df1 = pd.read_csv(input_gene_exp_path, index_col=0)
    df2 = pd.read_csv(input_methylation_path, index_col=0)

    common_columns = df1.columns.intersection(df2.columns)
    common_genes = df1.index.intersection(df2.index)

    df1_final = df1.loc[common_genes, common_columns]
    df1_final.to_csv(output_gene_exp_path)
    print(f'Filtered Gene Expression Data Shape : {df1_final.shape}')

    df2_final = df2.loc[common_genes, common_columns]
    df2_final.to_csv(output_methylation_path)
    print(f'Filtered Methylation Data Shape : {df2_final.shape}')



def save_data_to_tensor(filtered_gene_exp_path, filtered_methylation_path, output_gene_exp_tensor_path, output_methylation_tensor_path):
    """Gene Expression Data : (k_genes, k_sample_ids)
       Methylation Data  : (k_genes, k_sample_ids)

       Output -
        Gene Expression Tensor : torch.Tensor of shape (k_sample_ids, k_genes)
        Methylation Tensor : torch.Tensor of shape (k_sample_ids, k_genes)

        Output is saved as pickle files at output paths.
    """
    df1 = pd.read_csv(filtered_gene_exp_path, index_col=0)
    df2 = pd.read_csv(filtered_methylation_path, index_col=0)

    gene_exp_tensor = torch.tensor(df1.values.T, dtype=torch.float32)
    with open(output_gene_exp_tensor_path, 'wb') as f:
        pickle.dump(gene_exp_tensor, f)
    print(f'Saved Gene Expression Tensor of shape {gene_exp_tensor.shape} to {output_gene_exp_tensor_path}')

    methylation_tensor = torch.tensor(df2.values.T, dtype=torch.float32)
    with open(output_methylation_tensor_path, 'wb') as f:
        pickle.dump(methylation_tensor, f)
    print(f'Saved Methylation Tensor of shape {methylation_tensor.shape} to {output_methylation_tensor_path}')