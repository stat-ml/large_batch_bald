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
    
def apply_noise(idx, sample):
    data, target = sample
    return data + dataset_noise[idx], target
    
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

def create_CIFAR100_dataset():
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(*stats,inplace=True)
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    train_dataset = datasets.CIFAR100(
        "data", train=True, download=True, transform=train_transforms
    )
    test_dataset = datasets.CIFAR100(
        "data", train=False, transform=test_transforms
    )
    
    return train_dataset, test_dataset

def create_MNIST_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset

def create_EMNIST_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dataset = datasets.EMNIST(
        "emnist_data", split="balanced", train=True, download=True, 
        transform=transform
    )
    test_dataset = datasets.EMNIST(
        "emnist_data", split="balanced", train=False, transform=transform
    )

    return train_dataset, test_dataset

def create_FashionMNIST_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.286), (0.353))
    ])

    train_dataset = datasets.FashionMNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        "data", train=False, transform=transform
    )

    return train_dataset, test_dataset

def create_KMNIST_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = datasets.KMNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.KMNIST(
        "data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset

def create_SVHN_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    train_dataset = datasets.SVHN(
        "data", split='train', download=True, transform=transform
    )
    test_dataset = datasets.SVHN(
        "data", split='test', download=True, transform=transform
    )

    return train_dataset, test_dataset

def create_repeated_CIFAR10_dataset(*, num_repetitions=3, add_noise=True):
    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
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

    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty( \
            (len(train_dataset), 32, 32), dtype=torch.float32 \
        ).normal_(0.0, 0.1)

        train_dataset = TransformedDataset(
            train_dataset, transformer=apply_noise
        )

    test_dataset = datasets.CIFAR10(
        "data", train=False, transform=test_transforms
    )

    return train_dataset, test_dataset

def create_repeated_CIFAR100_dataset(*, num_repetitions=3, add_noise=True):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(*stats,inplace=True)
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(*stats)
    ])
    
    train_dataset = datasets.CIFAR100(
        "data", train=True, download=True, transform=train_transforms
    )
    
    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty( \
            (len(train_dataset), 32, 32), dtype=torch.float32 \
        ).normal_(0.0, 0.1)

        train_dataset = TransformedDataset(
            train_dataset, transformer=apply_noise
        )

    test_dataset = datasets.CIFAR100(
        "data", train=False, download=True, transform=test_transforms
    )

    return train_dataset, test_dataset

def create_repeated_MNIST_dataset(*, num_repetitions=3, add_noise=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )

    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty( \
            (len(train_dataset), 28, 28), dtype=torch.float32 \
        ).normal_(0.0, 0.1)

        train_dataset = TransformedDataset(
            train_dataset, transformer=apply_noise
        )

    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset