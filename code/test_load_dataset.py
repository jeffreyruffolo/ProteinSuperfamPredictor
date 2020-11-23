import torch
import matplotlib.pyplot as plt

from code.data import ProteinDataset

batch_size = 4
train_split = 0.9

h5_file = "data/dataset.h5"
dataset = ProteinDataset(h5_file)
train_split_length = int(len(dataset) * train_split)
torch.manual_seed(0)
train_dataset, validation_dataset = torch.utils.data.random_split(
    dataset,
    [train_split_length, len(dataset) - train_split_length])
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=ProteinDataset.merge_samples_to_minibatch)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size,
    collate_fn=ProteinDataset.merge_samples_to_minibatch)

# Loading and showing some distance matrices
for (seq_feats, dm_feats), superfams in train_loader:
    plt.imshow(dm_feats[0])
    plt.show()
    plt.close()
