import torch
from batchbald_redux import consistent_mc_dropout
from torch import nn as nn
from torch.nn import functional as F

class CNN_MC_RMNIST(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
    
class CNN_ENS_RMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input
    
class CNN_MC_EMNIST(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=47):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(512, num_classes)

    def mc_forward_impl(self, input:  torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(self.conv2_drop(self.conv2(input)))
        input = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(input)), 2))
        input = input.view(-1, 128 * 4 * 4)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input
    
class CNN_ENS_EMNIST(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_drop = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, input: torch.Tensor):
        input = F.relu(self.max_pool2d(self.conv1_drop(self.conv1(input))))
        input = F.relu(self.conv2_drop(self.conv2(input)))
        input = F.relu(self.max_pool2d(self.conv3_drop(self.conv3(input))))
        input = input.view(-1, 128 * 4 * 4)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input
    
# model = nn.Sequential( # (batch_size, 3, 32, 32)
#         nn.Conv2d(3, 32, kernel_size=3), # (batch_size, 32, 30, 30)
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.25),
#         nn.Conv2d(32, 32, kernel_size=3), # (batch_size, 32, 28, 28)
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.25),
#         nn.MaxPool2d(kernel_size=2), # (batch_size, 64, 14, 14)
#         nn.Conv2d(32, 64, kernel_size=3), # (batch_size, 64, 12, 12)
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.25),
#         nn.Conv2d(64, 64, kernel_size=3), # (batch_size, 64, 10, 10)
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.25),
#         nn.MaxPool2d(kernel_size=2), # (batch_size, 64, 5, 5)
#         nn.Flatten(), # (batch_size, 64*5*5)
#         nn.Dropout(p=0.5),
#         nn.Linear(64*5*5, 10), # (batch_size, )
#         )
    
class CNN_ENS_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv1_drop = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_drop = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4_drop = nn.Dropout(p=0.25)
        
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64*5*5, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, input: torch.Tensor):
        input = self.conv1_drop(self.relu(self.conv1(input)))
        input = self.max_pool2d(self.conv2_drop(self.relu(self.conv2(input))))
        input = self.conv3_drop(self.relu(self.conv3(input)))
        input = self.max_pool2d(self.conv4_drop(self.relu(self.conv4(input))))
        input = input.view(-1, 64 * 5 * 5)
        input = self.fc1_drop(self.relu(input))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input