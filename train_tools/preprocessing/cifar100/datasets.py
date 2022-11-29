import torch.utils.data as data
from torchvision.datasets import CIFAR100

import numpy as np


class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, download=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 100

        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        base_dataset = CIFAR100(
            self.root, self.train, self.transform, None, self.download
        )

        data = base_dataset.data
        targets = np.array(base_dataset.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)
