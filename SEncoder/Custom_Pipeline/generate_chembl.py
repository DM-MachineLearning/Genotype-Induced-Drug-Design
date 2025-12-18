"""
Generate molecules from a trained Custom Pipeline Transformer VAE checkpoint.

Usage:
    python generate_chembl.py --checkpoint checkpoints/chembl_sanity/best.pt --num 32 --max_length 120
"""

import argparse
import os
from typing import Dict, List

import torch

from smiles_encoder import SMILESEncoder
from smiles_decoder import SMILESDecoder


def load_checkpoint(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device)
    return obj


def build_models(meta: Dict, args: argparse.Namespace, device: torch.device):
    encoder = SMILESEncoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        dropout=args.dropout,
    ).to(device)
    decoder = SMILESDecoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        dropout=args.dropout,
    ).to(device)
    return encoder, decoder


def decode_tokens(token_ids: List[int], meta: Dict) -> str:
    vocab = meta["vocab"]
    tokens = []
    for t in token_ids:
        if t == meta["end_id"]:
            break
        if t in (meta["start_id"], meta["pad_id"]):
            continue
        tokens.append(vocab[t])
    return "".join(tokens)


def sample_latent(num: int, latent_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num, latent_size, device=device)


def main():
    parser = argparse.ArgumentParser(description="Generate SMILES from trained Transformer VAE.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt).")
    parser.add_argument("--num", type=int, default=32, help="Number of molecules to generate.")
    parser.add_argument("--max_length", type=int, default=120, help="Generation length (<= max_len).")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=None, help="Optional top-k sampling (if None, greedy).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)
    meta = ckpt["meta"]

    encoder, decoder = build_models(meta, args, device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()

    latent = sample_latent(args.num, args.latent_size, device)
    with torch.no_grad():
        logits = decoder.generate_forward(latent, start_token=meta["start_id"], max_length=args.max_length)

    # Optional top-k sampling; otherwise decoder is greedy already
    if args.topk and args.topk > 0:
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=args.topk, dim=-1)
        # Sample within top-k for each position
        rng = torch.rand_like(topk_vals)
        choices = torch.argmax(rng / (topk_vals + 1e-9), dim=-1)
        token_ids = topk_idx.gather(-1, choices.unsqueeze(-1)).squeeze(-1)
    else:
        token_ids = torch.argmax(logits, dim=-1)

    smiles = [decode_tokens(seq.tolist(), meta) for seq in token_ids]
    print("[INFO] Generated SMILES:")
    for i, smi in enumerate(smiles):
        print(f"{i:03d}: {smi}")


if __name__ == "__main__":
    main()

