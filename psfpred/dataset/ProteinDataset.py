import h5py
import torch
import torch.nn.functional as F


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, filename, crop_size=100, num_classes=51):
        super(ProteinDataset, self).__init__()

        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins = len(self.h5file['sequence_len'])
        self.crop_size = crop_size
        self.num_classes = num_classes

        superfam_list = list(
            c.decode("utf-8") for c in self.h5file['superfam'])
        self.superfam_freq_dict = {
            sf: superfam_list.count(sf)
            for sf in set(superfam_list)
        }
        if num_classes > 0:
            top_n_superfams = [
                sf for sf, _ in sorted(self.superfam_freq_dict.items(),
                                       key=lambda item: item[1],
                                       reverse=True)[:num_classes - 1]
            ]
            self.superfam_dict = {
                sf: i + 1
                for i, sf in enumerate(top_n_superfams)
            }
        else:
            self.superfam_dict = {
                sf: i
                for i, sf in enumerate(set(superfam_list))
            }

    def __len__(self):
        return self.num_proteins

    def __getitem__(self, index):
        id = self.h5file['id'][index]
        sequence_len = self.h5file['sequence_len'][index]

        superfam = self.h5file['superfam'][index]
        # superfam = torch.tensor(self.superfam_dict[superfam]).long()
        # superfam = self.superfam_dict[superfam]
        superfam = self.superfam_dict.setdefault(superfam.decode("utf-8"), 0)

        sequence = self.h5file['sequence'][index, :sequence_len]
        sequence = F.one_hot(torch.tensor(sequence).long(), 21)

        dist_mat = self.h5file['dist_mat'][index, :sequence_len, :sequence_len]
        dist_mat = torch.Tensor(dist_mat).type(dtype=torch.float)

        if self.crop_size > 0:
            crop_start = torch.randint(low=0,
                                       high=max(sequence_len - self.crop_size,
                                                1),
                                       size=(1, )).item()
            crop_end = min(sequence_len, crop_start + self.crop_size)
            sequence = sequence[crop_start:crop_end]
            dist_mat = dist_mat[crop_start:crop_end, crop_start:crop_end]

        return id, superfam, sequence, dist_mat

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return ProteinBatch(zip(*samples)).data()


def pad_data_to_same_shape(tensor_list, pad_value=0):
    shapes = torch.Tensor([_.shape for _ in tensor_list])
    target_shape = torch.max(shapes.transpose(0, 1), dim=1)[0].int()

    padded_dataset_shape = [len(tensor_list)] + list(target_shape)
    padded_dataset = torch.Tensor(*padded_dataset_shape).type_as(
        tensor_list[0])

    for i, data in enumerate(tensor_list):
        # Get how much padding is needed per dimension
        padding = reversed(target_shape - torch.Tensor(list(data.shape)).int())

        # Add 0 every other index to indicate only right padding
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        padding = padding.view(1, -1)[0].tolist()

        padded_data = F.pad(data, padding, value=pad_value)
        padded_dataset[i] = padded_data

    return padded_dataset


class ProteinBatch:
    def __init__(self, batch_data):
        (self.id, self.superfam, self.sequence, self.dist_mat) = batch_data

    def data(self):
        return self.features(), self.labels()

    def features(self):
        """Gets the one-hot encoding of the sequences with a feature that
        delimits the chains"""
        padded_sequence = pad_data_to_same_shape(
            self.sequence, pad_value=0).float().transpose(1, 2).contiguous()
        padded_dist_mat = pad_data_to_same_shape(self.dist_mat, pad_value=-999)

        return padded_sequence, padded_dist_mat

    def labels(self):
        return torch.tensor(list(self.superfam))


def h5_antibody_dataloader(filename, batch_size=1, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(dict(collate_fn=ProteinDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return torch.data.DataLoader(ProteinDataset(filename), **kwargs)
