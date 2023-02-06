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

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

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

def get_entropy_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N += compute_entropy(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

def get_variance_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N += torch.var(log_probs_N_K_C, dim=1).sum(dim=1)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

def compute_entropy_vec(log_probs_N_K_C: torch.Tensor, device) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        n = log_probs_n_K_C.shape[0]
        a = torch.matmul(torch.exp(log_probs_n_K_C).permute(2, 1, 0)[:, :, :, None].to(device), torch.exp(log_probs_n_K_C).permute(2, 1, 0)[:, :, None, :].to(device))
        zero_diag_mask = (torch.ones(n) - torch.eye(n)).repeat(C, K, 1, 1)
        a = a * zero_diag_mask.to(device)
        a = a.sum(dim=(1, 2)) / K
        a = a.t()
    
        c = torch.matmul(torch.exp(mean_log_probs_n_C).permute(1, 0)[:, :, None].to(device), torch.exp(mean_log_probs_n_C).permute(1, 0)[:, None, :].to(device))
        zero_diag_mask2 = (torch.ones(n) - torch.eye(n)).repeat(C, 1, 1)
        c = c * zero_diag_mask2.to(device)
        c = c.sum(dim=1)

        c = c.t()
    
        nats_n = (a * (torch.log(1.0 + a) - torch.log(1.0 + c))).sum(dim=1)
    
        entropies_N[start:end].copy_(-nats_n)

        pbar.update(end - start)

    pbar.close()

    return entropies_N

def get_lbb_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    scores_N -= compute_entropy_vec(log_probs_N_K_C, device)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())

def get_powerlbb_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None, alpha=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    scores_N -= compute_entropy_vec(log_probs_N_K_C, device)
    scores_N[scores_N < 0] = 0.0
    
    scores_N = torch.pow(scores_N, alpha)
    scores_N /= torch.sum(scores_N)

    candidate_indices = torch.multinomial(scores_N, batch_size, replacement=False)
    candiate_scores = scores_N[candidate_indices]

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