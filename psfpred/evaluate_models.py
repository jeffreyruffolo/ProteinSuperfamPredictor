import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from psfpred.dataset import ProteinDataset
from psfpred.model import Model

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = torch.device(device_type)

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
_, _, test_dataset = torch.utils.data.random_split(
    dataset, [train_split_length, validation_split_length, test_split_length])
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=ProteinDataset.merge_samples_to_minibatch)

model_10 = torch.load("training_runs/adam_10_2d/model.torch", map_location=dev)
model_6 = torch.load("training_runs/adam_6_2d/model.torch", map_location=dev)
model_2 = torch.load("training_runs/adam_2_2d/model.torch", map_location=dev)
model_sgd = torch.load("training_runs/sgd_10_2d/model.torch", map_location=dev)
model_seq = torch.load("training_runs/adam_10_2d_nodist/model.e12.torch",
                       map_location=dev)
model_dist = torch.load("training_runs/adam_10_2d_noseq/model.torch",
                        map_location=dev)

model_10.eval()
model_6.eval()
model_2.eval()
model_sgd.eval()
model_seq.eval()
model_dist.eval()

superfam_labels = []
model_10_preds = []
model_6_preds = []
model_2_preds = []
model_sgd_preds = []
model_seq_preds = []
model_dist_preds = []

with torch.no_grad():
    for (seq_inputs, dist_inputs), superfam_labels_ in tqdm(test_loader):
        seq_inputs = seq_inputs.to(dev)
        dist_inputs = dist_inputs.unsqueeze(1).to(dev)

        model_10_preds.append(model_10(seq_inputs, dist_inputs).argmax(1))
        model_6_preds.append(model_6(seq_inputs, dist_inputs).argmax(1))
        model_2_preds.append(model_2(seq_inputs, dist_inputs).argmax(1))
        model_sgd_preds.append(model_sgd(seq_inputs, dist_inputs).argmax(1))
        model_seq_preds.append(
            model_seq(seq_inputs, dist_inputs, dev=dev).argmax(1))
        model_dist_preds.append(
            model_dist(seq_inputs, dist_inputs, dev=dev).argmax(1))

        superfam_labels.append(superfam_labels_)

superfam_labels = torch.cat(superfam_labels)
model_10_preds = torch.cat(model_10_preds)
model_6_preds = torch.cat(model_6_preds)
model_2_preds = torch.cat(model_2_preds)
model_sgd_preds = torch.cat(model_sgd_preds)
model_seq_preds = torch.cat(model_seq_preds)
model_dist_preds = torch.cat(model_dist_preds)

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
model_names = [
    "Model-10", "Model-6", "Model-2", "Model-SGD", "Model-Seq", "Model-Dist"
]
model_preds = [
    model_10_preds, model_6_preds, model_2_preds, model_sgd_preds,
    model_seq_preds, model_dist_preds
]

res = np.concatenate([
    np.array([
        "Label", "Model-10", "Model-6", "Model-2", "Model-SGD", "Model-Seq",
        "Model-Dist"
    ])[None, :],
    torch.stack([
        superfam_labels, model_10_preds, model_6_preds, model_2_preds,
        model_sgd_preds, model_seq_preds, model_dist_preds
    ]).numpy().T
])
np.savetxt(os.path.join(out_dir, "results.csv"), res, delimiter=",", fmt="%s")

for i in range(len(model_names)):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(confusion_matrix(model_preds[i], superfam_labels)[1:, 1:])
    fig.colorbar()
    plt.savefig(os.path.join(out_dir, "{}_50_cm.png".format(model_names[i])))
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(confusion_matrix(model_preds[i], superfam_labels))
    fig.colorbar()
    plt.savefig(os.path.join(out_dir, "{}_51_cm.png".format(model_names[i])))
    plt.close()
