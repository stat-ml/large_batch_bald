# AUTOGENERATED! DO NOT EDIT! File to edit: 01_batchbald.ipynb (unless otherwise specified).

# __all__ = ['compute_conditional_entropy', 'compute_entropy', 'CandidateBatch', 'get_batchbald_batch', 'get_bald_batch']
__all__ = ['compute_conditional_entropy', 'compute_entropy', 'CandidateBatch', 'get_batchbald_batch', 'get_bald_batch', 'get_lbb_batch']

# Cell
import math
from dataclasses import dataclass
from typing import List

import torch
from toma import toma
from tqdm.auto import tqdm

from batchbald_redux import joint_entropy

# Cell
import numpy as np
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N

# Cell


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape
    # new probs that will be added to 
    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C) # from ordinary BALD

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    ) # same joint_entr class but with dynamic properties

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum() # they are easier, all sums

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)

# Cell


def get_bald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())
#####
# def compute_entropy_vec(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
#     N, K, C = log_probs_N_K_C.shape

#     entropies_N = torch.empty(N, dtype=torch.double)

#     pbar = tqdm(total=N, desc="Entropy", leave=False)

#     @toma.execute.chunked(log_probs_N_K_C, 1024)
#     def compute(log_probs_n_K_C, start: int, end: int):
# #         print("log_probs_n_K_C.shape:", log_probs_n_K_C.shape)
#         mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
# #         nats_n_C = torch.zeros(N, C) #?
#         n = log_probs_n_K_C.shape[0]
#         nats_n = torch.zeros(n) # N

# #         for i in range(N-1):
#         for i in range(n):
# #             for j in range(i+1, N):
#             for j in range(n): # N can be bigger than batch_size -> error # one of possible solut-s -- tensor view
# #                 print("i, j:", (i, j))
# #                 print("eq:", torch.exp(mean_log_probs_n_C[i][:]) * torch.exp(mean_log_probs_n_C[j][:])*\
# #                 (mean_log_probs_n_C[i][:] + mean_log_probs_n_C[j][:]) - torch.exp(mean_log_probs_n_C[i][:])*\
# #                 mean_log_probs_n_C[i][:] - torch.exp(mean_log_probs_n_C[j][:]) * mean_log_probs_n_C[j][:])
# #                 nats_n_C
#                 if i != j:
# #                     print("i, j:", (i, j))
# #                     print("mean_log_probs_n_C.shape:", mean_log_probs_n_C.shape)
#                     nats_n[i] += torch.sum(torch.exp(mean_log_probs_n_C[i][:]) * torch.exp(mean_log_probs_n_C[j][:])*\
#                     (mean_log_probs_n_C[i][:] + mean_log_probs_n_C[j][:]) - torch.exp(mean_log_probs_n_C[i][:])*\
#                     mean_log_probs_n_C[i][:] - torch.exp(mean_log_probs_n_C[j][:]) * mean_log_probs_n_C[j][:])
# #                 print("nats_n_C:", nats_n_C)

# #                 print("nats_n:", nats_n) ###
    
# #         print("nats_n_C.shape:", nats_n_C.shape)
# #         print("nats_n_C:", nats_n_C)
# #         print("mean_log_probs_n_C:", mean_log_probs_n_C)
#         entropies_N[start:end].copy_(nats_n)
# #         -torch.sum(nats_n_C, dim=1)

# #         print("entropies_N:", entropies_N) ###
    
#         pbar.update(end - start)

#     pbar.close()

#     return entropies_N
#####
# tensor way

# def compute_entropy_vec(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
#     N, K, C = log_probs_N_K_C.shape

#     entropies_N = torch.empty(N, dtype=torch.double)

#     pbar = tqdm(total=N, desc="Entropy", leave=False)

#     @toma.execute.chunked(log_probs_N_K_C, 1024)
#     def compute(log_probs_n_K_C, start: int, end: int):
#         mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
#         n = log_probs_n_K_C.shape[0]
# #         print("1 matmul:", torch.matmul(torch.exp(mean_log_probs_n_C), torch.exp(mean_log_probs_n_C).t()).shape)
# #         print("2 *:", torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[0]).fill_diagonal_(0.0).shape)
# #         print("3 matmul:", mean_log_probs_n_C.shape)
#         a = torch.matmul(torch.matmul(torch.exp(mean_log_probs_n_C), torch.exp(mean_log_probs_n_C).t())*torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[0]).fill_diagonal_(0.0), mean_log_probs_n_C).fill_diagonal_(0.0)
#         b = torch.matmul(torch.matmul(torch.exp(mean_log_probs_n_C), torch.exp(mean_log_probs_n_C).t())*torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[0]).fill_diagonal_(0.0), mean_log_probs_n_C)*torch.eye(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[1])
# #         c = torch.matmul(torch.exp(mean_log_probs_n_C), mean_log_probs_n_C.t())*torch.eye(mean_log_probs_n_C.shape[0])*n
# #         d = torch.matmul(torch.exp(mean_log_probs_n_C), mean_log_probs_n_C.t())*torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[0]).fill_diagonal_(0.0)
#         c = torch.exp(mean_log_probs_n_C) * mean_log_probs_n_C * torch.eye(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[1])*(n-1) #*n #*(n-1) #?
#         d = torch.exp(mean_log_probs_n_C) * mean_log_probs_n_C * torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[1]).fill_diagonal_(0.0)
        
# #         print((a * b).shape)
# #         print(c.shape)
# #         print(d.shape)
# #         nats_n = (torch.matmul(a, b.t()) - c - d).sum(dim=1)
#         nats_n = (a + b - c - d).sum(dim=1)

#         pbar.update(end - start)

#     pbar.close()

#     return entropies_N
#####
# def compute_entropy_vec(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
#     N, K, C = log_probs_N_K_C.shape

#     entropies_N = torch.empty(N, dtype=torch.double)

#     pbar = tqdm(total=N, desc="Entropy", leave=False)

#     @toma.execute.chunked(log_probs_N_K_C, 1024)
#     def compute(log_probs_n_K_C, start: int, end: int):
#         mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K) # change
#         n = log_probs_n_K_C.shape[0]
#         a = torch.matmul(torch.exp(mean_log_probs_n_C), torch.exp(mean_log_probs_n_C).t())*torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[0]).fill_diagonal_(0.0)
#         b = (a * torch.log(a)).sum(dim=1)
#         c = (torch.matmul(a, mean_log_probs_n_C) * torch.eye(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[1])).sum(dim=1)
#         d = (torch.matmul(a, mean_log_probs_n_C) * torch.ones(mean_log_probs_n_C.shape[0], mean_log_probs_n_C.shape[1]).fill_diagonal_(0.0)).sum(dim=1)

# #         print("a.shape:", a.shape)
# #         print("b.shape:", b.shape)
# #         print("c.shape:", c.shape)
# #         print("d.shape:", d.shape)
#         nats_n = (b - c - d) #.sum(dim=1)

#         pbar.update(end - start)

#     pbar.close()

#     return entropies_N
#####
# def compute_entropy_vec(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
#     N, K, C = log_probs_N_K_C.shape

#     entropies_N = torch.empty(N, dtype=torch.double)

#     pbar = tqdm(total=N, desc="Entropy", leave=False)

#     @toma.execute.chunked(log_probs_N_K_C, 1024)
#     def compute(log_probs_n_K_C, start: int, end: int):
#         mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
# #         nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
#         n = log_probs_n_K_C.shape[0]
# #         print("1:", torch.exp(log_probs_n_K_C[:, :, :, None]).shape) # .transpose(2, 3)
# #         print("2:", torch.exp(log_probs_n_K_C)[:, :, None, :].shape) # .transpose(1, 2) #.transpose(1, 3)
# #         print("1:", torch.exp(log_probs_n_K_C[:, :, None, :]).shape)
# #         print("2:", torch.exp(log_probs_n_K_C)[:, None, :, :].transpose(2, 3).shape)

#         # mb to use expand func as in joint prob file
#         a = torch.matmul(torch.exp(log_probs_n_K_C[:, :, :, None]), torch.exp(log_probs_n_K_C)[:, :, None, :]) # .transpose(1, 3)
# #         a = torch.matmul(torch.exp(log_probs_n_K_C).permute(2, 0, 1), torch.exp(log_probs_n_K_C).permute(2, 1, 0))
    
#         print("a.shape:", a.shape)
#         a = a.sum(dim=(1, 2)) / K
#         print("a.shape:", a.shape)
#         a = a.masked_fill_(torch.eye(n, C).byte(), 0.0)
#         b = (a * torch.log(a)).sum(dim=1) # already log probs
#         c = (torch.matmul(a, mean_log_probs_n_C.t()) * torch.eye(n, n)).sum(dim=1) # n, C
#         d = torch.matmul(a, mean_log_probs_n_C.t())
#         d = d.masked_fill_(torch.eye(n, n).byte(), 0.0).sum(dim=1) # n, C

# #         print("a.shape:", a.shape)
# #         print("b.shape:", b.shape)
# #         print("c.shape:", c.shape)
# #         print("d.shape:", d.shape)
#         nats_n = b - c - d #.sum(dim=1)

#         pbar.update(end - start)

#     pbar.close()

#     return entropies_N
#####
def compute_entropy_vec(log_probs_N_K_C: torch.Tensor, device) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
#         nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
        n = log_probs_n_K_C.shape[0]
#         print("1:", torch.exp(log_probs_n_K_C[:, :, :, None]).shape) # .transpose(2, 3)
#         print("2:", torch.exp(log_probs_n_K_C)[:, :, None, :].shape) # .transpose(1, 2) #.transpose(1, 3)
#         print("1:", torch.exp(log_probs_n_K_C[:, :, None, :]).shape)
#         print("2:", torch.exp(log_probs_n_K_C)[:, None, :, :].transpose(2, 3).shape)

        # mb to use expand func as in joint prob file
        a = torch.matmul(torch.exp(log_probs_n_K_C).permute(2, 1, 0)[:, :, :, None].to(device), torch.exp(log_probs_n_K_C).permute(2, 1, 0)[:, :, None, :].to(device)) # .transpose(1, 3) 
#         print("a.is_cuda:", a.is_cuda)
        # .permute(2, 1, 0) # .permute(2, 1, 0)
        # .permute(2, 1, 0, 3) # .permute(3, 1, 2, 0)
#         a = torch.matmul(torch.exp(log_probs_n_K_C).permute(2, 0, 1), torch.exp(log_probs_n_K_C).permute(2, 1, 0))
    
#         print("a.shape:", a.shape)
#         a = a.permute(3, 2, 1, 0) # before: C, K, n, n

#         print("a.shape:", a.shape)
        zero_diag_mask = (torch.ones(n) - torch.eye(n)).repeat(C, K, 1, 1) #.view(n, n, K, C) #.permute(3, 2, 1, 0)
#         print("torch.ones(n) - torch.eye(n):", (torch.ones(n) - torch.eye(n)).shape)
#         print("torch.ones(n, n) - torch.eye(n, n):", (torch.ones(n, n) - torch.eye(n, n)).shape)
#         zero_diag_mask = torch.ones(n, n, K, C).diag_embed(0, dim1=0, dim2=1)
#         print("zero_diag_mask.shape:", zero_diag_mask.shape)
#         print("zero_diag_mask:", zero_diag_mask)
        a = a * zero_diag_mask.to(device)
#         print("a:", a)
#         print("a.shape:", a.shape)
        a = a.sum(dim=(1, 2)) / K
#         print("a.shape:", a.shape)
#         print("a:", a)
        a = a.t()
    
#         a = ((a * zero_diag_mask).sum(dim=(1, 2)) / K).t() ###
    
#         print("a.shape:", a.shape)
#         a = a.masked_fill_(torch.eye(n, C).byte(), 0.0) # fix
#         log_a = torch.log(a)
#         log_a[log_a == '-inf'] = 0.0

#         b = (a * torch.log(1.0 + a)).sum(dim=1) # logsumexp mb
    
#         b = b*(n-1) #? # should be equivalent to n sums as in c and d

#         c = (torch.matmul(a, mean_log_probs_n_C.t()) * torch.eye(n, n)).sum(dim=1) # n, C ###
#         c = c*(n-1) # because on each n-1 jth will be n-1 ith ###
#         d = torch.matmul(a, mean_log_probs_n_C.t()) ###
        
#         print("d:", d)
#         print("d.masked_fill_(torch.eye(n, n).bool(), 0.0):", d.masked_fill_(torch.eye(n, n).bool(), 0.0))

#         d = d.masked_fill_(torch.eye(n, n).bool(), 0.0).sum(dim=1) # n, C ###

#         print("a.shape:", a.shape)
#         print("b.shape:", b.shape)
#         print("c.shape:", c.shape)
#         print("d.shape:", d.shape)

#         nats_n = b - c - d
    
        c = torch.matmul(torch.exp(mean_log_probs_n_C).permute(1, 0)[:, :, None].to(device), torch.exp(mean_log_probs_n_C).permute(1, 0)[:, None, :].to(device)) #was without exp
        zero_diag_mask2 = (torch.ones(n) - torch.eye(n)).repeat(C, 1, 1)
#         print("zero_diag_mask2.shape:", zero_diag_mask2.shape)
#         print("zero_diag_mask2:", zero_diag_mask2)
#         print("c.is_cuda:", c.is_cuda)
        c = c * zero_diag_mask2.to(device)
        c = c.sum(dim=1)
        
#         print("c.shape:", c.shape)
        c = c.t()
    
#         с = (c * zero_diag_mask2).sum(dim=1).t() ###
    
#         c = (a * c).sum(dim=1)
#         print("c.shape:", c.shape)
#         nats_n = b - c
    
        nats_n = (a * (torch.log(1.0 + a) - torch.log(1.0 + c))).sum(dim=1) # was just c
    
#         print("nats_n:", nats_n)
#         print("a:", a)
#         print("torch.log(1.0 + a):", torch.log(1.0 + a))
#         print("b:", b)
#         print("c:", c)
#         print("torch.log(1.0 + c):", torch.log(1.0 + c))
        
#         print("nats_n:", nats_n)
#         print("d:", d)
#         print("end-start:", end-start)
#         print("nats_n.shape:", nats_n.shape)
        entropies_N[start:end].copy_(-nats_n) # sign

        pbar.update(end - start)

    pbar.close()

    return entropies_N
#####
# def compute_entropy_vec(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
#     N, K, C = log_probs_N_K_C.shape

#     entropies_N = torch.empty(N, dtype=torch.double)

#     pbar = tqdm(total=N, desc="Entropy", leave=False)

#     @toma.execute.chunked(log_probs_N_K_C, 1024)
#     def compute(log_probs_n_K_C, start: int, end: int):
#         mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
# #         nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)
#         n = log_probs_n_K_C.shape[0]
# #         print("1:", torch.exp(log_probs_n_K_C[:, :, :, None]).shape) # .transpose(2, 3)
# #         print("2:", torch.exp(log_probs_n_K_C)[:, :, None, :].shape) # .transpose(1, 2) #.transpose(1, 3)
# #         print("1:", torch.exp(log_probs_n_K_C[:, :, None, :]).shape)
# #         print("2:", torch.exp(log_probs_n_K_C)[:, None, :, :].transpose(2, 3).shape)

#         # mb to use expand func as in joint prob file
# #         a = torch.matmul(torch.exp(log_probs_n_K_C[:, :, :, None]), torch.exp(log_probs_n_K_C)[:, :, None, :]) # .transpose(1, 3)
#         a = torch.matmul(torch.exp(log_probs_n_K_C).permute(2, 0, 1), torch.exp(log_probs_n_K_C).permute(2, 1, 0))
    
# #         print("a.shape:", a.shape)
#         a = a.permute(2, 1, 0).sum(dim=2)
# #         a = a.sum(dim=(1, 2)) / K
# #         print("a.shape:", a.shape)
#         a = a.masked_fill_(torch.eye(n, n).byte(), 0.0) # n, C
#         b = (a * torch.log(a)).sum(dim=1)
#         c = (torch.matmul(a, mean_log_probs_n_C.t()) * torch.eye(n, n)).sum(dim=1) # n, C
#         d = torch.matmul(a, mean_log_probs_n_C.t())
#         d = d.masked_fill_(torch.eye(n, n).byte(), 0.0).sum(dim=1) # n, C

# #         print("a.shape:", a.shape)
# #         print("b.shape:", b.shape)
# #         print("c.shape:", c.shape)
# #         print("d.shape:", d.shape)
#         nats_n = b - c - d #.sum(dim=1)

#         pbar.update(end - start)

#     pbar.close()

#     return entropies_N

def get_lbb_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    scores_N -= compute_entropy_vec(log_probs_N_K_C, device)
#     print("scores_N:", scores_N)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

#########################

def get_random_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape
    
    batch_size = min(batch_size, N)
    
    candidate_indices = []
    candidate_scores = []

    candiate_scores, candidate_indices = torch.rand(batch_size), torch.multinomial(torch.range(0, N, 1), num_samples=batch_size, replacement=False)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

#########################
# ens setting #

# def init_glorot(model):
#     for module in model.modules():
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.xavier_uniform_(module.weight)
            
# def ens(model, T):
#     models = []
#     for _ in range(T):
#         model = make_model() #re-init model
#         model.apply(init_glorot)
#         _, model = train()
#         models.append(model)
    
#     otput = []
#     for model in models:
#         output.append(torch.softmax(model(batch), dim=-1))
#     output = torch.stack(output, dim=0)
#     output = output.mean(dim=0)
    
#     return output

def get_powerlbb_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None, alpha=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    scores_N -= compute_entropy_vec(log_probs_N_K_C, device)
    scores_N[scores_N < 0] = 0.0
#     print("scores_N:", scores_N)
#     print("scores_N min orig:", torch.min(scores_N))
    
    scores_N = torch.pow(scores_N, alpha) # 5
#     print("scores_N:", scores_N)
#     print("min:", torch.min(scores_N))
#     print("sum scores_N:", torch.sum(scores_N))
    scores_N /= torch.sum(scores_N)
#     print("dist:", scores_N)
#     print("max:", torch.max(scores_N))
#     print("min:", torch.min(scores_N))

    candidate_indices = torch.multinomial(scores_N, batch_size, replacement=False)
#     print("candidate_indices:", candidate_indices)
    candiate_scores = scores_N[candidate_indices]
#     print("candiate_scores:", candiate_scores)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

def get_powerbald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)
    
    scores_N = torch.pow(scores_N, 5)
    scores_N /= torch.sum(scores_N)
    candidate_indices = torch.multinomial(scores_N, batch_size, replacement=False)
    candiate_scores = scores_N[candidate_indices]

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())