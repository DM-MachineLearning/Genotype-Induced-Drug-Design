import argparse
import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

from smiles_encoder import SMILESEncoder


def load_dataset(data_dir):
    train = torch.load(os.path.join(data_dir, "train.pt"), map_location="cpu")
    val = torch.load(os.path.join(data_dir, "val.pt"), map_location="cpu")
    return train, val


def build_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    encoder = SMILESEncoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=meta.get("embedding_dim", 512),
        n_layers=meta.get("n_layers", 8),
        latent_size=meta.get("latent_size", 512),
        max_len=meta["max_len"],
        dropout=0.0,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    return encoder, meta


@torch.no_grad()
def extract_latents(encoder, loader, device, label="train"):
    latents = []
    total = len(loader.dataset)
    batch_size = loader.batch_size

    start_time = time.time()
    last_print = start_time

    for i, (tokens,) in enumerate(loader):
        tokens = tokens.to(device)

        # Forward pass
        mu, _ = encoder(tokens)
        latents.append(mu.cpu())

        # Progress logging every ~5 seconds
        if time.time() - last_print > 5:
            processed = min((i + 1) * batch_size, total)
            elapsed = time.time() - start_time
            speed = processed / elapsed
            eta = (total - processed) / max(speed, 1e-6)

            print(
                f"[INFO][{label}] "
                f"{processed:,} / {total:,} "
                f"({100 * processed / total:.2f}%) | "
                f"{speed:,.0f} SMILES/s | "
                f"ETA {eta/60:.1f} min"
            )
            last_print = time.time()

    return torch.cat(latents, dim=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_obj, val_obj = load_dataset(args.data_dir)
    encoder, meta = build_encoder(args.checkpoint, device)

    train_loader = DataLoader(
        TensorDataset(train_obj["data"]),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        TensorDataset(val_obj["data"]),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    print("[INFO] Extracting train latents...")
    train_z = extract_latents(encoder, train_loader, device, label="train")

    print("[INFO] Extracting val latents...")
    val_z = extract_latents(encoder, val_loader, device, label="val")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(
        {"z": train_z, "meta": meta},
        os.path.join(args.output_dir, "train_latents.pt"),
    )

    torch.save(
        {"z": val_z, "meta": meta},
        os.path.join(args.output_dir, "val_latents.pt"),
    )

    print(f"[DONE] Saved train latents: {train_z.shape}")
    print(f"[DONE] Saved val latents:   {val_z.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract latent vectors from trained VAE encoder")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="latents")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)
