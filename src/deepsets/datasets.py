from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from .settings import DATA_ROOT

from IPython import embed

MNIST_TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class MNISTSummation(Dataset):
    def __init__(self, min_len: int, max_len: int, dataset_len: int, train: bool = True, transform: Compose = None):
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_len = dataset_len
        self.train = train
        self.transform = transform

        self.mnist = MNIST(DATA_ROOT, train=self.train, transform=self.transform, download=True)
        # print(self.mnist.__dict__)
        mnist_len = self.mnist.__len__()
        mnist_items_range = np.arange(0, mnist_len)

        items_len_range = np.arange(self.min_len, self.max_len + 1)
        items_len = np.random.choice(items_len_range, size=self.dataset_len, replace=True)
        self.mnist_items = []
        for i in range(self.dataset_len):
            label = np.random.choice(np.arange(0,10))
            idx = self.mnist.targets == label
            # print(mnist_items_range[idx])
            # self.mnist_items.append(np.random.choice(mnist_items_range, size=items_len[i], replace=True))
            self.mnist_items.append(np.random.choice(mnist_items_range[idx], size=items_len[i], replace=True))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        mnist_items = self.mnist_items[item]

        # the_sum = 0
        images1 = []
        for mi in mnist_items:
            img, target = self.mnist.__getitem__(mi)
            # the_sum += target
            images1.append(img)

        mnist_items_range = np.arange(0, self.mnist.__len__())
        idx = self.mnist.targets == target
        mnist_items = np.random.choice(mnist_items_range[idx], size=len(mnist_items), replace=True)

        images2 = []
        for mi in mnist_items:
            img, target = self.mnist.__getitem__(mi)
            # the_sum += target
            images2.append(img)

        # return torch.stack(images, dim=0), torch.FloatTensor([the_sum])
        # print(len(images1),len(images2))
        return torch.stack(images1, dim=0), torch.stack(images2, dim=0), torch.LongTensor([target])
