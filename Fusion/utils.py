import torch

def get_positive_negative_indices(pair_matrix: torch.Tensor):
    """
    pair_matrix: (N_cell_lines, N_drugs) binary tensor
    Returns:
        pos_pairs: (N_pos, 2)
        neg_pairs: (N_neg, 2)
    """
    # We find coordinates where the matrix is 1 (pos) or 0 (neg).
    pos_pairs = torch.nonzero(pair_matrix == 1, as_tuple=False)
    neg_pairs = torch.nonzero(pair_matrix == 0, as_tuple=False)

    return pos_pairs, neg_pairs


def build_pairwise_embeddings(G, S, pair_indices):
    """
    G: (N_cell_lines, d_gen)
    S: (N_drugs, d_smiles)
    pair_indices: (N_pairs, 2)

    Returns:
        z_gen, z_smiles: (N_pairs, d_input)
    """
    cell_idx = pair_indices[:, 0]
    drug_idx = pair_indices[:, 1]
    
    # Simple indexing to get the raw features for the pairs
    return G[cell_idx], S[drug_idx]

