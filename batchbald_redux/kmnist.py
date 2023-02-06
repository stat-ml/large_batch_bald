# Cell

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Cell
import numpy as np
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


def create_KMNIST_dataset():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    train_dataset = datasets.KMNIST("data", train=True, download=True, transform=transform)

    test_dataset = datasets.KMNIST("data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)