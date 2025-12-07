from torch.utils.data import Dataset, DataLoader, random_split

class MultiOmicsDataset(Dataset):
    def __init__(self, dna_data, gene_data):
        """
        dna_data: tensor (N, input_dim_dna)
        gene_data: tensor (N, input_dim_gene)
        """
        self.dna = dna_data
        self.gene = gene_data

    def __len__(self):
        return len(self.dna)

    def __getitem__(self, idx):
        return self.dna[idx], self.gene[idx]
    

    
def return_dataloaders(X_dna_meth, X_gene_exp, split_fractions : tuple = (0.5, 0.3)):

    """ split_fractions (tuple) : fractional sizes of training and validation set """

    dataset = MultiOmicsDataset(X_dna_meth, X_gene_exp)

    train_size = int(split_fractions[0]*len(dataset))
    val_size = int(split_fractions[1]*len(dataset))
    test_size = int(len(dataset)) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader





