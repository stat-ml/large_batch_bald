from tqdm import tqdm
import os
import time
import numpy as np
import random
import pandas as pd
import csv
import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import copy
from argparse import ArgumentParser

from dataclasses import dataclass
from typing import List

import torch
from torch.nn import functional as F
import torchvision

from batchbald_redux import (
    progr_active_learning as active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
    emnist,
    fmnist,
    cifar10,
    repeated_cifar10,
    cifar100,
    svhn
)

from cnn_models import (
    CNN_MC_RMNIST,
    CNN_ENS_RMNIST,
    CNN_MC_EMNIST,
    CNN_ENS_EMNIST,
    CNN_ENS_CIFAR10
)

from utils import init_glorot, plot_graph
# from pytorch_resnet_cifar10.resnet import resnet20

parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MNIST')
# parser.add_argument('--dataset_names', nargs='+', type=str, default=['CIFAR100', 'CIFAR10', 'EMNIST', 'FMNIST', 'SVHN', 'RMNIST', 'MNIST'])
parser.add_argument('--model_name', type=str, default='CNN_ENS_RMNIST')
parser.add_argument('--optimizer_name', type=str, default='Adam')
parser.add_argument('--uns_type', type=str, default='ENS')
parser.add_argument('--algs', nargs='+', type=str, default=['PLBB', 'PBALD', 'Rand', 'LBB', 'BALD', 'BB'])
parser.add_argument('--random_seeds', nargs='+', type=int, default=[42, 227, 346, 684, 920]) # 42, 227, 346, 684, 920 
parser.add_argument('--num_models', type=int, default=5) # 5, 10
parser.add_argument('--num_init_samples', type=int, default=200)
parser.add_argument('--max_train_samples', type=int, default=10000)
parser.add_argument('--acq_batch_size', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=50) # 64
parser.add_argument('--pool_batch_size', type=int, default=100) #128
parser.add_argument('--test_batch_size', type=int, default=5000) # 250 # 512
parser.add_argument('--num_train_inference_samples', type=int, default=100)
parser.add_argument('--num_test_inference_samples', type=int, default=5)
parser.add_argument('--num_samples', type=int, default=100000)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--training_iterations', type=int, default=24576) # 4096*6
parser.add_argument('--cuda_number', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--val_size', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience_threshold', type=int, default=10)
args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
optimizer_name = args.optimizer_name
uns_type = args.uns_type
num_initial_samples = args.num_init_samples

if dataset_name == 'MNIST':
    train_dataset, test_dataset = repeated_mnist.create_repeated_MNIST_dataset(num_repetitions=1, add_noise=False)
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    repeated_mnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'RMNIST':
    train_dataset, test_dataset = repeated_mnist.create_repeated_MNIST_dataset(num_repetitions=4, add_noise=False)
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    repeated_mnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'EMNIST':
    train_dataset, test_dataset = emnist.create_EMNIST_dataset()
    num_classes = 47
    initial_samples = active_learning.get_balanced_sample_indices(
    emnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'FMNIST':
    train_dataset, test_dataset = fmnist.create_FashionMNIST_dataset()
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    fmnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'CIFAR10':
    train_dataset, test_dataset = cifar10.create_CIFAR10_dataset()
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    cifar10.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'RCIFAR10':
    train_dataset, test_dataset = repeated_cifar10.create_repeated_CIFAR10_dataset(num_repetitions=4, add_noise=False)
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    repeated_cifar10.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
#     initial_samples = active_learning.get_unbalanced_sample_indices(
#     repeated_cifar10.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes,
#         not_used_classes=[3, 5, 7]
# )
#     print("initial_samples.get_targets:", set(initial_samples.get_targets))
elif dataset_name == 'CIFAR100':
    train_dataset, test_dataset = cifar100.create_CIFAR100_dataset()
    num_classes = 100
    initial_samples = active_learning.get_balanced_sample_indices(
    cifar100.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
elif dataset_name == 'SVHN':
    train_dataset, test_dataset = svhn.create_SVHN_dataset()
    num_classes = 10
    initial_samples = active_learning.get_balanced_sample_indices(
    svhn.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
)
algs = args.algs

random_seeds = args.random_seeds

max_training_samples = args.max_train_samples
acquisition_batch_size = args.acq_batch_size
num_train_inference_samples = args.num_train_inference_samples
num_test_inference_samples = args.num_test_inference_samples
num_samples = args.num_samples

test_batch_size = args.test_batch_size
batch_size = args.train_batch_size
scoring_batch_size = args.pool_batch_size
training_iterations = args.training_iterations

T = args.num_models
cuda_number = args.cuda_number

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

epochs = args.num_epochs

config = {
    'model': model_name,
    'max_training_samples': max_training_samples,
    'number of models': T,
    'train batch size': batch_size,
    'pool batch size': scoring_batch_size,
    'num_initial_samples': num_initial_samples,
    'train size': len(train_dataset),
    'test size': len(test_dataset),
    'pool size': len(train_dataset) - num_initial_samples
} 
PATH = 'results/' + uns_type + '/' + dataset_name + '/' + str(acquisition_batch_size)
try:
    os.makedirs(PATH)    
except FileExistsError:
    pass

with open(PATH + '/config.json', 'w') as f:
    json.dump(config, f)
    
@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]
        
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
#         x = F.log_softmax(x, dim=1)

        return x

for random_seed in random_seeds:
    for alg in algs:
#         print("random_seed:", random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        
        print("curr alg:", alg)
        use_cuda = torch.cuda.is_available()

#         print(f"use_cuda: {use_cuda}")

        device = "cuda:" + str(cuda_number) if use_cuda else "cpu"

        kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {} # "num_workers": 1

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

        active_learning_data = active_learning.ActiveLearningData(train_dataset)

        # Split off the initial samples first.
        active_learning_data.acquire(initial_samples)
        
#         active_learning_data.extract_dataset_from_pool(40000)

        train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset,
            shuffle=True,
#             sampler=active_learning.RandomFixedLengthSampler(active_learning_data.training_dataset, training_iterations),
            batch_size=batch_size,
            **kwargs,
        )
    
        val_indeces = random.sample(range(0, len(active_learning_data.pool_dataset)), args.val_size)
        active_learning_data.val_acquire(val_indeces)
#         val_set = torch.utils.data.Subset(active_learning_data.pool_dataset, val_indeces)
        val_loader = torch.utils.data.DataLoader(
            active_learning_data.val_dataset,
            batch_size=batch_size,
            **kwargs,
        )
        
#         active_learning_data.pool_dataset = active_learning_data.remove_from_pool(val_indeces)

        pool_loader = torch.utils.data.DataLoader(
            active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs
        )
#         print("pool data:", len(pool_loader.dataset))

#         def append_dropout(model, rate=0.2):
#             for name, module in model.named_children():
#                 if len(list(module.children())) > 0:
#                     append_dropout(module)
#                 if isinstance(module, torch.nn.ReLU):
#                     new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate))
#                     setattr(model, name, new)



        # Run experiment
        test_accs = []
        test_loss = []
        added_indices = []

        pbar = tqdm(initial=len(active_learning_data.training_dataset), total=max_training_samples, desc="Training Set Size")

        while True:
            if uns_type == 'MC':
                T = 1
                if model_name == 'CNN_MC_EMNIST':
                    model = CNN_MC_EMNIST(num_classes).to(device=device)
                elif model_name == 'CNN_MC_RMNIST':
                    model = CNN_MC_RMNIST(num_classes).to(device=device)
            models = []
            for _ in range(T):
                if model_name == 'CNN_ENS_CIFAR10':
                    model = CNN_ENS_CIFAR10(num_classes).to(device=device)
                elif model_name == 'CNN_ENS_EMNIST':
                    model = CNN_ENS_EMNIST(num_classes).to(device=device)
                elif model_name == 'CNN_ENS_RMNIST':
                    model = CNN_ENS_RMNIST(num_classes).to(device=device)
                elif model_name == 'ResNet-18':
#                     model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=device)
                    model = Model(num_classes).to(device=device)
#                     append_dropout(model) ###
#                     model = torchvision.models.resnet152(pretrained=False, num_classes=num_classes).to(device=device)
                elif model_name == 'ResNet-20':
                    model = resnet20().to(device=device)
                elif model_name == 'VGG-16':
                    model = torchvision.models.vgg16(pretrained=False, num_classes=num_classes).to(device=device)
                elif model_name == 'DenseNet-121':
#                     model = torchvision.models.densenet121(growth_rate=12, pretrained=False, num_classes=num_classes, drop_rate=0.2).to(device=device)
                    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes, drop_rate=args.dropout_rate).to(device=device)
                model.apply(init_glorot) ###
                if optimizer_name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # 0.001
#                     optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=1e-4)
#                     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=epochs, 
#                                                 steps_per_epoch=len(train_loader))
                elif optimizer_name == 'SGD':
#                     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
#                     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#                     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005) #

                    milestones = [25, 40]

                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
                    )
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=milestones, gamma=0.1
                    )
                
                patience = 0
                prev_loss = 10e8
                prev_acc = 0
                best_loss = 10e8
                best_acc = 0
    
#                 model.train()
                
                for epoch in range(epochs):
#                     print("epoch:", epoch)
#                     while patience < args.patience_threshold:
                        model.train()
                        correct = 0
                        epoch_loss = 0.0

                        # Train
                        for data, target in tqdm(train_loader, desc="Training", leave=False):
                            data = data.to(device=device)
                            target = target.to(device=device)

                            optimizer.zero_grad()
                            if model_name == 'ResNet-18' or model_name == 'ResNet-20' or model_name == 'VGG-16' or model_name == 'DenseNet-121':
                                if uns_type == 'MC':
                                    prediction = torch.log_softmax(model(data, 1).squeeze(1), dim=1)
#                                     prediction = model(data, 1).squeeze(1)
                                elif uns_type == 'ENS':
                                    prediction = torch.log_softmax(model(data), dim=1)
#                                     prediction = model(data)
                            else:
                                if uns_type == 'MC':
                                    prediction = model(data, 1).squeeze(1)
                                elif uns_type == 'ENS':
                                    prediction = model(data)

                            loss = F.nll_loss(prediction, target)

                            loss.backward()
                            optimizer.step()
    #                         print("train prediction raw:", prediction.shape)
#                             print("loss.item():", loss.item())
#                             print("len(train_loader.dataset):", len(train_loader.dataset))
                            epoch_loss += loss.item()

                            prediction = prediction.max(1)[1] # max indeces
                            correct += prediction.eq(target.view_as(prediction)).sum().item()

    #                         print("train prediction.max(1)[1]:", prediction.shape)
    #                         print("train target:", target)
                        scheduler.step() # important to place outside dataloader cycle
                        train_percentage_correct = 100.0 * correct / len(train_loader.dataset)
                        epoch_loss /= len(train_loader.dataset)
                        print("epoch_loss:", epoch_loss)

                        print(
                            "Train set: Accuracy: {}/{} ({:.2f}%)".format(
                                correct, len(train_loader.dataset), train_percentage_correct
                            )
                        )

                        val_loss = 0
                        correct = 0
                        with torch.no_grad():
                            for data, target in tqdm(val_loader, desc="val", leave=False):
                                data = data.to(device=device)
                                target = target.to(device=device)
    #                             print("model(data):", model(data))

                                if model_name == 'ResNet-18' or model_name == 'ResNet-20' or model_name == 'VGG-16' or model_name == 'DenseNet-121':
                                    if uns_type == 'MC':
                                        prediction = torch.log_softmax(model(data, 1).squeeze(1), dim=1)
                                    elif uns_type == 'ENS':
                                        prediction = torch.log_softmax(model(data), dim=1)
#                                         prediction = model(data)
                                else:
                                    if uns_type == 'MC':
                                        prediction = model(data, 1).squeeze(1)
                                    elif uns_type == 'ENS':
                                        prediction = model(data)
    #                             print("test prediction raw:", prediction.shape)
                                loss = F.nll_loss(prediction, target)
        #                         print("train prediction raw:", prediction.shape)
                                val_loss += loss.item()

                                prediction = prediction.max(1)[1]  # max indeces
    #                             print("prediction.max():", prediction.max())
    #                             print("prediction.min():", prediction.min())
                                correct += prediction.eq(target.view_as(prediction)).sum().item()
    #                             print("test prediction prediction.max(1)[1]:", prediction.shape)
    #                             print("test target:", target)

                            percentage_correct = 100.0 * correct / len(val_loader.dataset)

                            val_loss /= len(val_loader.dataset)
#                             if val_loss < best_loss:
                            if percentage_correct > best_acc:
                                best_acc = copy.copy(percentage_correct)
                                best_loss = copy.copy(val_loss)
                                best_model = copy.deepcopy(model)

#                             print("prev_loss:", prev_loss)
                            print("prev_acc:", prev_acc)
                            print("val_loss:", val_loss)
#                             if prev_loss < val_loss:
                            if prev_acc > percentage_correct:
                                patience += 1

#                             prev_loss = copy.copy(val_loss)
                            prev_acc = copy.copy(percentage_correct)

                        print("patience:", patience)
                        print(
                            "Val set: Accuracy: {}/{} ({:.2f}%)".format(
                                correct, len(val_loader.dataset), percentage_correct
                            )
                        )
                print("best val accuracy:", best_acc)
                models.append(best_model)

            # Test
            for model in models:
                model = model.eval()
                
            if len(models) == 1:
                model = models[0]

            loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc="Testing", leave=False):
                    data = data.to(device=device)
                    target = target.to(device=device)
                    
                    if uns_type == 'MC':
                        prediction = torch.logsumexp(model(data, num_test_inference_samples), dim=1) - math.log(num_test_inference_samples)
                    elif uns_type == 'ENS':
                        ens_test_output = []
                        for model in models:
                            if model_name == 'ResNet-18' or model_name == 'ResNet-20' or model_name == 'VGG-16' or model_name == 'DenseNet-121':
                                ens_test_output.append(torch.log_softmax(model(data), dim=1))
#                                 ens_test_output.append(model(data))
                            else:
                                ens_test_output.append(model(data))
                        ens_test_output = torch.stack(ens_test_output, dim=1)

                        prediction = torch.logsumexp(ens_test_output, dim=1) - math.log(T)

                    loss += F.nll_loss(prediction, target, reduction="sum")

                    prediction = prediction.max(1)[1]
                    correct += prediction.eq(target.view_as(prediction)).sum().item()

            loss /= len(test_loader.dataset)
            test_loss.append(loss)

            percentage_correct = 100.0 * correct / len(test_loader.dataset)
            test_accs.append(percentage_correct)

            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    loss, correct, len(test_loader.dataset), percentage_correct
                )
            )
            
            filename = PATH + "/" + alg + str(random_seed) + ".csv"
            file_exists = os.path.isfile(filename)

            with open(filename, 'a+', newline='') as csvfile:
                fieldnames = ['Number of samples', 'Test accuracy', 'Test loss', 'Time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                if len(active_learning_data.training_dataset) != num_initial_samples: 
                    period = 0 # period = "{:.6f}".format(end - start)
                else:
                    period = 0

                writer.writerow({'Number of samples': len(active_learning_data.training_dataset),
                                 'Test accuracy': percentage_correct,
                                 'Test loss': "{:.6f}".format(loss.item()),
                                 'Time': period,
                })
                csvfile.close()

            if len(active_learning_data.training_dataset) >= max_training_samples:
                break

            # Acquire pool predictions # = pred from selected pool
            N = len(active_learning_data.pool_dataset)
            if uns_type == 'MC':
                logits_N_K_C = torch.empty((N, num_train_inference_samples, num_classes), dtype=torch.double, pin_memory=use_cuda)
            elif uns_type == 'ENS':
                logits_N_K_C = torch.empty((N, T, num_classes), dtype=torch.double, pin_memory=use_cuda)
        #     print("logits_N_K_C.size():", logits_N_K_C.size())

            with torch.no_grad():
                model.eval()

                for i, (data, _) in enumerate(tqdm(pool_loader, desc="Evaluating Acquisition Set", leave=False)):
                    data = data.to(device=device)

                    lower = i * pool_loader.batch_size
                    upper = min(lower + pool_loader.batch_size, N)
                    if uns_type == 'MC':
                        logits_N_K_C[lower:upper].copy_(model(data, num_train_inference_samples).double(), non_blocking=True)
                    elif uns_type == 'ENS':
                        ens_pool_output = []
                        for model in models:
                            if model_name == 'ResNet-18' or model_name == 'ResNet-20' or model_name == 'VGG-16' or model_name == 'DenseNet-121':
                                ens_pool_output.append(torch.log_softmax(model(data), dim=1))
#                                 ens_pool_output.append(model(data))
                            else:    
                                ens_pool_output.append(model(data))
                        ens_pool_output = torch.stack(ens_pool_output, axis=1)

                        logits_N_K_C[lower:upper].copy_(ens_pool_output.double(), non_blocking=True)

            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_counter()
                if alg == 'BB':
                    candidate_batch = batchbald.get_batchbald_batch(
                        logits_N_K_C, acquisition_batch_size, num_samples, dtype=torch.double, device=device
                    )
                elif alg == 'LBB':
                    candidate_batch = batchbald.get_lbb_batch(
                        logits_N_K_C, acquisition_batch_size, dtype=torch.double, device=device
                    )
                elif alg == 'BALD':
                    candidate_batch = batchbald.get_bald_batch(
                        logits_N_K_C, acquisition_batch_size, dtype=torch.double, device=device
                    )
                elif alg == 'Rand':
#                     candidate_batch = batchbald.get_random_batch(
#                         logits_N_K_C, acquisition_batch_size, dtype=torch.double, device=device
#                     )
                    candiate_scores, candidate_indices = np.random.randn(acquisition_batch_size), active_learning_data.get_random_pool_indices(acquisition_batch_size)
#                     np.random.choice(active_learning_data.pool_dataset.indices, acquisition_batch_size)
                    candidate_batch = CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())
                    
                elif alg == 'PLBB': 
                    candidate_batch = batchbald.get_powerlbb_batch(
                        logits_N_K_C, acquisition_batch_size, dtype=torch.double, device=device, alpha=5
                    )
                elif alg == 'PBALD': 
                    candidate_batch = batchbald.get_powerbald_batch(
                        logits_N_K_C, acquisition_batch_size, dtype=torch.double, device=device
                    )
                end = time.perf_counter()
                print("acquisition time (sec.):", end - start)
            if dataset_name == 'CIFAR10':
                targets = cifar10.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'FMNIST':
                targets = fmnist.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'EMNIST':
                targets = emnist.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'CIFAR100':
                targets = cifar100.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'SVHN':
                targets = svhn.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'MNIST' or dataset_name == 'RMNIST':
                targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
            elif dataset_name == 'RCIFAR10':
                targets = repeated_cifar10.get_targets(active_learning_data.pool_dataset)    
            dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)

            print("Dataset indices: ", dataset_indices)
            print("Scores: ", candidate_batch.scores)
            print("Labels: ", targets[candidate_batch.indices])

            active_learning_data.acquire(candidate_batch.indices)
            added_indices.append(dataset_indices)
            pbar.update(len(dataset_indices))
#     plot_graph(algs, uns_type, dataset_name, acquisition_batch_size, random_seed, num_initial_samples, max_training_samples)