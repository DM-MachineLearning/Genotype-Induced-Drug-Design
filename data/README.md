### TCGA Data Preprocessing

All preprocessing logic is consolidated in a single file: preprocessing.py.
The file merges per-cancer datasets, adds cancer labels, aligns genes and samples across modalities, and converts the data into PyTorch tensors.

## Overview
1. Per-cancer CSV files
2. Merge across cancers
3. Add cancer labels
4. Filter common genes & samples
5. Convert to PyTorch tensors

## Functions
# merge_csv_files
Utility function to merge multiple CSV files in a directory using an inner join.

# collect_and_merge
Reads per-cancer CSV files from a folder, merges them column-wise, and tracks sample-to-cancer mappings.

# save_data_with_cancer_labels
Merges per-cancer DM and EG datasets and appends a cancer label column for each sample.
Outputs labeled merged files to merged_outputs/.

# only_common_rows_cols
Filters gene expression and methylation datasets to retain only:
common genes
common sample IDs
Ensures both modalities are aligned.

# save_data_to_tensor
Converts filtered CSV files into PyTorch tensors and saves them as pickle files.