"""
Encoder-only pretraining for the SMILES Transformer encoder.

Workflow:
    1) Fetch SMILES:        python fetch_chembl.py --out_csv data/chembl/chembl_smiles.csv --limit 200000
    2) Tokenize SMILES:     python preprocess_chembl.py --input_csv data/chembl/chembl_smiles.csv --output_dir data/chembl/tokenized
    3) Pretrain encoder:    python pretrain_encoder.py --data_dir data/chembl/tokenized --checkpoint_dir checkpoints/encoder_pretrain
"""

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smiles_encoder import SMILESEncoder
from train_chembl import load_split


class SimpleDecoderHead(nn.Module):
    """
    Lightweight decoder head that uses the encoder's mean vector (mu)
    to reconstruct token logits for every position.
    """

    def __init__(self, latent_size: int, max_len: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.positional = nn.Embedding(max_len, latent_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(latent_size, vocab_size)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mu : torch.Tensor [batch_size, latent_size]

        Returns
        -------
        logits : torch.Tensor [batch_size, max_len, vocab_size]
        """
        batch_size, latent_size = mu.shape
        # Expand mu across sequence length and add positional embedding
        mu_expanded = mu.unsqueeze(1).expand(batch_size, self.max_len, latent_size)
        positions = torch.arange(self.max_len, device=mu.device).unsqueeze(0).expand(batch_size, -1)
        hidden = mu_expanded + self.positional(positions)
        hidden = self.dropout(hidden)
        logits = self.proj(hidden)
        return logits


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict]:
    train_obj = load_split(os.path.join(data_dir, "train.pt"))
    val_obj = load_split(os.path.join(data_dir, "val.pt"))
    meta = train_obj["meta"]

    train_ds = TensorDataset(train_obj["data"])
    val_ds = TensorDataset(val_obj["data"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, meta


def prepare_batch(batch: Tuple[torch.Tensor], device: torch.device) -> torch.Tensor:
    return batch[0].to(device)


def save_checkpoint(path: str, encoder: nn.Module, meta: Dict, step: int, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "meta": meta,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
    print(f"[INFO] Saved checkpoint to {path} (step={step}, val_loss={val_loss:.4f})")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, meta = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    encoder = SMILESEncoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        dropout=args.dropout,
    ).to(device)

    head = SimpleDecoderHead(
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        vocab_size=len(meta["vocab"]),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return max(0.0, (args.max_steps - step) / max(1, args.max_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ce_loss = nn.CrossEntropyLoss(ignore_index=meta["pad_id"])

    global_step = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        for batch in train_loader:
            global_step += 1
            tokens = prepare_batch(batch, device)

            mu, _ = encoder(tokens)
            logits = head(mu)

            loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step >= args.max_steps:
                break

        # Validation
        encoder.eval()
        head.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = prepare_batch(batch, device)
                mu, _ = encoder(tokens)
                logits = head(mu)
                loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        val_loss = val_loss / max(1, val_batches)
        print(f"[val] epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_encoder.pt"),
                encoder,
                meta,
                step=global_step,
                val_loss=val_loss,
            )

        if global_step >= args.max_steps:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encoder-only pretraining on ChEMBL tokenized SMILES.")
    parser.add_argument("--data_dir", type=str, default="data/chembl/tokenized", help="Directory with train.pt / val.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/encoder_pretrain", help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=5000, help="Stop after this many steps (across epochs)")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
