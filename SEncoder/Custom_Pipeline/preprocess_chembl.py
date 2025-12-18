"""
Tokenize ChEMBL SMILES into padded integer tensors for the Custom Pipeline VAE.

Usage:
    python preprocess_chembl.py --input_csv data/chembl/chembl_smiles.csv \\
        --output_dir data/chembl/tokenized --max_len 122 --limit 200000
"""

import argparse
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch

# Vocabulary (45 tokens) following the paper and Custom Pipeline defaults
VOCAB = [
    "<pad>", "<s>", "</s>",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", "[", "]", ":", "=", "@",
    "B", "C", "F", "H", "I", "N", "O", "P", "S",
    "b", "c", "n", "o", "s", "p",
    "Cl", "Br",
    "#", "+", "-", "/", "\\", ".",
    "@@", "%",
]

TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID["<pad>"]
START_ID = TOKEN_TO_ID["<s>"]
END_ID = TOKEN_TO_ID["</s>"]


def tokenize_smiles(smiles: str) -> Optional[List[str]]:
    """Tokenize a SMILES string using the defined vocabulary."""
    tokens: List[str] = []
    i = 0
    while i < len(smiles):
        # Multi-character tokens first
        if smiles.startswith("Cl", i):
            tokens.append("Cl")
            i += 2
            continue
        if smiles.startswith("Br", i):
            tokens.append("Br")
            i += 2
            continue
        if smiles.startswith("@@", i):
            tokens.append("@@")
            i += 2
            continue
        ch = smiles[i]
        if ch in TOKEN_TO_ID:
            tokens.append(ch)
            i += 1
            continue
        # Escape/backslash and forward slash
        if ch == "\\":
            tokens.append("\\")
            i += 1
            continue
        if ch == "/":
            tokens.append("/")
            i += 1
            continue
        # Unknown token -> drop molecule
        return None
    return tokens


def encode_smiles(smiles: str, max_len: int) -> Optional[List[int]]:
    tokens = tokenize_smiles(smiles)
    if tokens is None:
        return None
    if len(tokens) + 2 > max_len:  # account for start/end
        return None
    encoded = [START_ID] + [TOKEN_TO_ID[t] for t in tokens] + [END_ID]
    # Pad to max_len
    encoded += [PAD_ID] * (max_len - len(encoded))
    return encoded


def load_smiles(input_csv: str, limit: Optional[int]) -> List[str]:
    df = pd.read_csv(input_csv)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")
    smiles = df["smiles"].dropna().astype(str).tolist()
    if limit:
        smiles = smiles[:limit]
    return smiles


def split_data(data: List[List[int]], train_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * train_ratio)
    train = torch.tensor(data[:split], dtype=torch.long)
    val = torch.tensor(data[split:], dtype=torch.long)
    return train, val


def save_dataset(train: torch.Tensor, val: torch.Tensor, output_dir: str, max_len: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "vocab": VOCAB,
        "token_to_id": TOKEN_TO_ID,
        "pad_id": PAD_ID,
        "start_id": START_ID,
        "end_id": END_ID,
        "max_len": max_len,
        "train_size": train.shape[0],
        "val_size": val.shape[0],
    }
    torch.save({"data": train, "meta": meta}, os.path.join(output_dir, "train.pt"))
    torch.save({"data": val, "meta": meta}, os.path.join(output_dir, "val.pt"))
    print(f"[INFO] Saved train set: {train.shape}")
    print(f"[INFO] Saved val set:   {val.shape}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize ChEMBL SMILES for VAE pretraining.")
    parser.add_argument("--input_csv", type=str, default="data/chembl/chembl_smiles.csv",
                        help="CSV file with a 'smiles' column.")
    parser.add_argument("--output_dir", type=str, default="data/chembl/tokenized",
                        help="Directory to save tokenized tensors.")
    parser.add_argument("--max_len", type=int, default=122, help="Maximum sequence length (including start/end).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of molecules to process.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    smiles_list = load_smiles(args.input_csv, limit=args.limit)
    print(f"[INFO] Loaded {len(smiles_list)} SMILES rows.")

    encoded: List[List[int]] = []
    for smi in smiles_list:
        enc = encode_smiles(smi, max_len=args.max_len)
        if enc is not None:
            encoded.append(enc)
    print(f"[INFO] Encoded {len(encoded)} SMILES (dropped {len(smiles_list) - len(encoded)} too-long/unknown).")

    if not encoded:
        raise SystemExit("No SMILES were encoded; check vocab/max_len settings.")

    train, val = split_data(encoded, train_ratio=args.train_ratio, seed=args.seed)
    save_dataset(train, val, args.output_dir, max_len=args.max_len)


if __name__ == "__main__":
    main()

