"""
Extract gene names from CSV header and create simple mapping files
"""

import json

# -----------------------------
# CONFIG
# -----------------------------

GENE_EXPR_CSV = "/home/dmlab/Devendra/data/tcga_transcdr/GeneExpression_data_with_cancer_filtered_final.csv"
OUTPUT_GENE_MAPPING = "/home/dmlab/Devendra/gene_index_mapping.json"
OUTPUT_CANCER_MAPPING = "/home/dmlab/Devendra/cancer_class_mapping.json"

# -----------------------------
# EXTRACT GENE NAMES FROM HEADER
# -----------------------------

print("Reading CSV header...")
with open(GENE_EXPR_CSV, 'r') as f:
    header = f.readline().strip()

# Split by comma and get gene names
columns = header.split(',')

# First column is usually blank or patient ID, rest are genes
gene_names = columns[1:] if columns[0] == '' or 'Unnamed' in columns[0] else columns

print(f"Found {len(gene_names)} genes")
print(f"First 10: {gene_names[:10]}")
print(f"Last 10: {gene_names[-10:]}")

# Create gene index mapping
gene_mapping = {idx: gene_name for idx, gene_name in enumerate(gene_names)}

# -----------------------------
# CREATE CANCER CLASS MAPPING
# -----------------------------

# Common TCGA cancer types (28 classes)
tcga_cancer_types = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT"
]

cancer_mapping = {idx: cancer_type for idx, cancer_type in enumerate(tcga_cancer_types)}

# -----------------------------
# SAVE MAPPINGS
# -----------------------------

print(f"\nSaving gene mapping to {OUTPUT_GENE_MAPPING}...")
with open(OUTPUT_GENE_MAPPING, "w") as f:
    json.dump(gene_mapping, f, indent=2)

print(f"Saving cancer mapping to {OUTPUT_CANCER_MAPPING}...")
with open(OUTPUT_CANCER_MAPPING, "w") as f:
    json.dump(cancer_mapping, f, indent=2)

# -----------------------------
# SUMMARY
# -----------------------------

print("\n" + "="*60)
print(f"Total genes: {len(gene_mapping)}")
print(f"Total cancer classes: {len(cancer_mapping)}")
print("\nFiles saved:")
print(f"  - {OUTPUT_GENE_MAPPING}")
print(f"  - {OUTPUT_CANCER_MAPPING}")
print("="*60)
