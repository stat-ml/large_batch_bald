import random

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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
            self.transformer = \
                lambda _, data_label: \
                (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
    
def get_targets(dataset):
    """
    Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets.
    """
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat(
                [get_targets(sub_dataset) for sub_dataset in dataset.datasets]
            )

    return torch.as_tensor(dataset.targets)

def create_CIFAR10_dataset():
    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(*stats, inplace=True)
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
       
    train_dataset = datasets.CIFAR10(
        "data", train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR10(
        "data", train=False, transform=test_transforms
    )

    return train_dataset, test_dataset