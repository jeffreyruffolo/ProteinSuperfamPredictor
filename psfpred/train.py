import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from psfpred.dataset import ProteinDataset
from psfpred.model import Model

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = torch.device(device_type)

# To use TPU (not sure if this works)
# import torch_xla
# import torch_xla.core.xla_model as xm
# dev = xm.xla_device()

print("Using {} as device".format(dev))

out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/adam_10_2d"
# out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/adam_6_2d"
# out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/adam_2_2d"
# out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/sgd_10_2d"
# out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/adam_10_2d_noseq"
# out_dir = "/home-2/jruffol1@jhu.edu/code/ProteinSuperfamPredictor/training_runs/adam_10_2d_nodist"
os.makedirs(out_dir, exist_ok=True)

max_seq_len = 100
num_classes = 51
batch_size = 32

dataset = ProteinDataset("data/dataset.h5",
                         crop_size=max_seq_len,
                         num_classes=num_classes)

validation_split = 0.1
test_split = 0.2
train_split = 1 - validation_split - test_split
validation_split_length = int(len(dataset) * validation_split)
test_split_length = int(len(dataset) * test_split)
train_split_length = len(dataset) - validation_split_length - test_split_length
torch.manual_seed(0)
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_split_length, validation_split_length, test_split_length])
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=ProteinDataset.merge_samples_to_minibatch)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size,
    collate_fn=ProteinDataset.merge_samples_to_minibatch)
# test_loader = torch.utils.data.DataLoader(
#     validation_dataset,
#     batch_size=batch_size,
#     collate_fn=ProteinDataset.merge_samples_to_minibatch)

model = Model(max_seq_len=max_seq_len, num_classes=num_classes).to(dev)
# model = Model(max_seq_len=max_seq_len,
#               num_classes=num_classes,
#               ignore_seq=True).to(dev)
# model = Model(max_seq_len=max_seq_len,
#               num_classes=num_classes,
#               ignore_dist=True).to(dev)
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
loss_func = torch.nn.CrossEntropyLoss().to(dev)

epochs = 40
train_losses = []
validation_losses = []
for i in tqdm(range(epochs)):
    model.train()

    train_loss = 0
    for (seq_inputs, dist_inputs), superfam_labels in tqdm(train_loader):
        optimizer.zero_grad()

        seq_inputs = seq_inputs.to(dev)
        dist_inputs = dist_inputs.unsqueeze(1).to(dev)

        def handle_batch():
            outputs = model(seq_inputs, dist_inputs, dev=dev)
            loss = loss_func(outputs, superfam_labels.to(dev))

            loss.backward()
            optimizer.step()

            return loss.item()

        train_loss += handle_batch()

    validation_loss = 0
    with torch.no_grad():
        model.eval()
        for (seq_inputs,
             dist_inputs), superfam_labels in tqdm(validation_loader):
            seq_inputs = seq_inputs.to(dev)
            dist_inputs = dist_inputs.unsqueeze(1).to(dev)

            def handle_batch():
                outputs = model(seq_inputs, dist_inputs, dev=dev)
                loss = loss_func(outputs, superfam_labels.to(dev))

                return loss.item()

            validation_loss += handle_batch()

    scheduler.step(validation_loss / len(validation_loader))

    train_losses.append(train_loss / len(train_loader))
    validation_losses.append(validation_loss / len(validation_loader))

    print("train", train_losses[-1])
    print("validation", validation_losses[-1])

    plt.figure(dpi=500)
    plt.plot(train_losses, label="Train")
    plt.plot(validation_losses, label="Validation")
    plt.ylabel("CCE Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()
    np.savetxt(os.path.join(out_dir, "loss_data.csv"),
               np.array([
                   list(range(len(train_losses))), train_losses,
                   validation_losses
               ]).T,
               delimiter=",")

    torch.save(model, os.path.join(out_dir, "model.e{}.torch".format(i + 1)))

torch.save(model, os.path.join(out_dir, "model.torch"))
