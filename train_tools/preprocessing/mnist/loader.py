import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import sys, os

from .datasets import MNIST_truncated


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def _data_transforms_cifar10():
    MNIST_MEAN = [0.1307]
    MNIST_STD = [0.3081]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD),]
    )

    return train_transform, valid_transform


def get_all_targets_mnist(root, train=True):
    dataset = MNIST_truncated(root=root, train=train)
    all_targets = dataset.targets
    return all_targets


def get_dataloader_mnist(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar10()

    if train:
        dataset = MNIST_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )
        dataset.targets

    else:
        dataset = MNIST_truncated(
            root, dataidxs, train=False, transform=valid_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader
