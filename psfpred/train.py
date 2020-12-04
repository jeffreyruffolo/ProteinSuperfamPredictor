from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from psfpred.dataset import ProteinDataset
from psfpred.model import Model

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = torch.device(device_type)

# To use TPU (not sure if this works)
# import torch_xla
# import torch_xla.core.xla_model as xm
# dev = xm.xla_device()

max_seq_len = 100
num_classes = 51
batch_size = 64

dataset = ProteinDataset("data/dataset.h5",
                         crop_size=max_seq_len,
                         num_classes=num_classes)

train_split = 0.9
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

model = Model(max_seq_len=max_seq_len, num_classes=num_classes).to(dev)
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss().to(dev)

epochs = 50
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
            outputs = model(seq_inputs, dist_inputs)
            loss = loss_func(outputs, superfam_labels.to(dev))

            loss.backward()
            optimizer.step()

            return loss.item()

        train_loss += handle_batch()

        if train_loss > 300:
            break

    validation_loss = 0
    with torch.no_grad():
        model.eval()
        for (seq_inputs,
             dist_inputs), superfam_labels in tqdm(validation_loader):
            seq_inputs = seq_inputs.to(dev)
            dist_inputs = dist_inputs.unsqueeze(1).to(dev)

            def handle_batch():
                outputs = model(seq_inputs, dist_inputs)
                loss = loss_func(outputs, superfam_labels.to(dev))

                return loss.item()

            validation_loss += handle_batch()

            if validation_loss > 300:
                break

    train_losses.append(train_loss / len(train_loader))
    validation_losses.append(validation_loss / len(validation_loader))

    if (i + 1) % 2 == 0:
        plt.plot(train_losses, label="Train")
        plt.plot(validation_losses, label="Validation")
        plt.xlabel("CCE Loss")
        plt.ylabel("Epoch")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()

        torch.save(model, "model.e{}.torch".format(i + 1))

plt.plot(train_losses, label="Train")
plt.plot(validation_losses, label="Validation")
plt.xlabel("CCE Loss")
plt.ylabel("Epoch")
plt.legend()
plt.savefig("loss.png")
plt.close()

torch.save(model, "model.torch")