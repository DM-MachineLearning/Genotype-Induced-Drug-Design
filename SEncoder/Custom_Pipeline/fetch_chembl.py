"""
Download a subset of ChEMBL SMILES for quick pretraining experiments.

Usage (examples):
    python fetch_chembl.py --out_csv data/chembl/chembl_smiles.csv --limit 200000
    python fetch_chembl.py --out_csv data/chembl/chembl_smiles.csv --limit 50000 --only_organics

Notes:
- Uses chembl_webresource_client (install via `pip install chembl_webresource_client`).
- Canonicalization with RDKit is optional; if RDKit is available, SMILES are canonicalized
  and sanitized. Otherwise, raw SMILES strings are kept.
- The default cap (200k) fits comfortably in memory on a 16GB RAM machine.
"""

import argparse
import csv
import os
from typing import Iterable, List, Optional, Tuple

try:
    from chembl_webresource_client.new_client import new_client
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "chembl_webresource_client is required. Install with `pip install chembl_webresource_client`."
    ) from exc

try:
    from rdkit import Chem
except ImportError:
    Chem = None  # RDKit optional


def canonicalize(smiles: str, require_organic: bool, strict: bool = False) -> Optional[str]:
    """Canonicalize SMILES with RDKit when available; optionally filter to organics."""
    if Chem is None:
        return smiles  # Fallback to raw SMILES
    mol = Chem.MolFromSmiles(smiles, sanitize=not strict)
    if mol is None:
        return None
    if require_organic:
        # Filter out molecules containing atoms beyond organic subset
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in (1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53):
                return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def fetch_chembl_smiles(limit: int, batch: int = 5000) -> Iterable[str]:
    """Stream SMILES from ChEMBL with pagination to avoid large memory spikes."""
    molecule = new_client.molecule
    fields = ["molecule_chembl_id", "canonical_smiles"]
    fetched = 0
    offset = 0

    while fetched < limit:
        remaining = limit - fetched
        size = min(batch, remaining)
        results = molecule.filter(molecule_structures__isnull=False).only(fields)[offset : offset + size]
        if not results:
            break
        for row in results:
            smiles = row.get("canonical_smiles")
            if smiles:
                yield smiles
                fetched += 1
                if fetched >= limit:
                    break
        offset += size


def save_smiles(smiles_list: List[str], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smi in smiles_list:
            writer.writerow([smi])


def main():
    parser = argparse.ArgumentParser(description="Fetch a subset of ChEMBL SMILES.")
    parser.add_argument("--out_csv", type=str, default="data/chembl/chembl_smiles.csv",
                        help="Output CSV path (will create directories if needed).")
    parser.add_argument("--limit", type=int, default=200_000,
                        help="Max number of SMILES to download.")
    parser.add_argument("--only_organics", action="store_true",
                        help="Keep only organic molecules (C, H, N, O, F, P, S, Cl, Br, I, B).")
    parser.add_argument("--strict_rdkit", action="store_true",
                        help="Use strict RDKit sanitization (may drop more molecules).")
    parser.add_argument("--batch", type=int, default=5000,
                        help="Batch size for pagination.")
    args = parser.parse_args()

    print(f"[INFO] Fetching up to {args.limit} SMILES from ChEMBL...")
    raw_smiles = list(fetch_chembl_smiles(limit=args.limit, batch=args.batch))
    print(f"[INFO] Retrieved {len(raw_smiles)} SMILES rows.")

    if Chem is None:
        print("[WARN] RDKit not installed; skipping canonicalization and organic filtering.")
        processed = raw_smiles
    else:
        processed = []
        for smi in raw_smiles:
            can = canonicalize(smi, require_organic=args.only_organics, strict=args.strict_rdkit)
            if can:
                processed.append(can)
        print(f"[INFO] Canonicalized with RDKit; kept {len(processed)} molecules.")

    # Drop duplicates while preserving order
    seen = set()
    unique_smiles = []
    for smi in processed:
        if smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
    print(f"[INFO] Unique SMILES count: {len(unique_smiles)}")

    save_smiles(unique_smiles, args.out_csv)
    print(f"[INFO] Saved SMILES to {args.out_csv}")


if __name__ == "__main__":
    main()

