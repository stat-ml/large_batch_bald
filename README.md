# Large Batch BALD (LBB)
> Scalable implementation of Batch BALD.


We present the Large BatchBALD algorithm, which gives a well-grounded approximation to the BatchBALD method that aims to achieve comparable quality while being more computationally efficient.

The code of original BatchBALD is available on https://github.com/BlackHC/batchbald_redux, 
and the website on https://blackhc.github.io/batchbald_redux.

## Install BatchBALD

`pip install batchbald_redux`

## Motivation

BatchBALD calculates mutual information between model output and model parameters but in a batch sense, that is, considering inter-variable correlation and taking a batch of outputs as a joint random variable. While accounting nicely for the correlation between observation, BatchBALD criterion is often computationally expensive,especially for large batches.

We approximate BatchBALD acquisition function as follows:

$$I_{\text{LBB}}(y_{1:b}; \theta) := \sum \limits_{i=1}^b I(y_i, \theta) - \sum \limits_{i=1}^b \sum \limits_{j \neq i}^b I(y_i; y_j),$$

where $I$ - is a mutual information function.

In addition, we looked into PowerBALD modification and added our own implementation for Large BatchBALD.

## Please cite us

At the moment our paper is not published yet.

## How to use

Available datasets: ['CIFAR100', 'CIFAR10', 'EMNIST', 'FMNIST', 'SVHN', 'RMNIST', 'MNIST']

Available acquisition methods: ['PLBB', 'PBALD', 'Rand', 'LBB', 'BALD', 'BB']. These are shorts for: Power Large BatchBALD, PowerBALD, Random, Large BatchBALD, BALD and BatchBALD.

We provide a simple example experiment that uses this package [here](example_link). 
