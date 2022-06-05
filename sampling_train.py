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
from argparse import ArgumentParser

from dataclasses import dataclass
from typing import List

import torch
from torch.nn import functional as F
import torchvision

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
    emnist,
    fmnist,
    cifar10,
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
parser.add_argument('--num_models', type=int, default=10)
parser.add_argument('--num_init_samples', type=int, default=200)
parser.add_argument('--max_train_samples', type=int, default=10000)
parser.add_argument('--acq_batch_size', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=64) # 64
parser.add_argument('--pool_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=512)
parser.add_argument('--num_train_inference_samples', type=int, default=100)
parser.add_argument('--num_test_inference_samples', type=int, default=5)
parser.add_argument('--num_samples', type=int, default=100000)
parser.add_argument('--training_iterations', type=int, default=24576) # 4096*6
parser.add_argument('--cuda_number', type=int, default=0)
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

        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

        active_learning_data = active_learning.ActiveLearningData(train_dataset)

        # Split off the initial samples first.
        active_learning_data.acquire(initial_samples)
        
#         active_learning_data.extract_dataset_from_pool(40000)

        train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset,
            sampler=active_learning.RandomFixedLengthSampler(active_learning_data.training_dataset, training_iterations),
            batch_size=batch_size,
            **kwargs,
        )

        pool_loader = torch.utils.data.DataLoader(
            active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs
        )

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
                    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device=device)
                elif model_name == 'ResNet-20':
                    model = resnet20().to(device=device)
                model.apply(init_glorot)
                if optimizer_name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters())
                elif optimizer_name == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

                model.train()

                # Train
                for data, target in tqdm(train_loader, desc="Training", leave=False):
                    data = data.to(device=device)
                    target = target.to(device=device)

                    optimizer.zero_grad()
                    if model_name == 'ResNet-18' or model_name == 'ResNet-20':
                        if uns_type == 'MC':
                            prediction = torch.log_softmax(model(data, 1).squeeze(1), dim=1)
                        elif uns_type == 'ENS':
                            prediction = torch.log_softmax(model(data), dim=1)
                    else:
                        if uns_type == 'MC':
                            prediction = model(data, 1).squeeze(1)
                        elif uns_type == 'ENS':
                            prediction = model(data)
                  
                    loss = F.nll_loss(prediction, target)

                    loss.backward()
                    optimizer.step()

                models.append(model)

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
                            if model_name == 'ResNet-18' or model_name == 'ResNet-20':  
                                ens_test_output.append(torch.log_softmax(model(data), dim=1))
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
                    period = "{:.6f}".format(end - start)
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
                            if model_name == 'ResNet-18' or model_name == 'ResNet-20':  
                                ens_pool_output.append(torch.log_softmax(model(data), dim=1))
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
            dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)

            print("Dataset indices: ", dataset_indices)
            print("Scores: ", candidate_batch.scores)
            print("Labels: ", targets[candidate_batch.indices])

            active_learning_data.acquire(candidate_batch.indices)
            added_indices.append(dataset_indices)
            pbar.update(len(dataset_indices))
#     plot_graph(algs, uns_type, dataset_name, acquisition_batch_size, random_seed, num_initial_samples, max_training_samples)