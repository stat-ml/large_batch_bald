# Scalable Batch Acquisition for Deep Bayesian Active Learning
This is a PyTorch implementation of the [SDM 2023](https://www.siam.org/conferences/cm/conference/sdm23) paper [Scalable Batch Acquisition for Deep Bayesian Active Learning](https://arxiv.org/abs/2301.05490). Our work present a novel bayesian active learning algorithm called Large BatchBALD (and its stochastic extension Power Large BatchBALD), which gives a well-grounded approximation to the [BatchBALD](https://arxiv.org/abs/1906.08158) method and aims to achieve comparable quality while being more computationally efficient.

## Install
`pip install batchbald_redux`

## Train examples
FMNIST with MC-dropout:
```sh
python sampling_train.py --dataset_name='FMNIST' --model_name='CNN_MC_RMNIST' --uns_type='MC' --algs PLBB PBALD Rand LBB BALD BB MaxProb --random_seeds 42 227 346 684 920 --acq_batch_size=10 --num_init_samples=20 --max_train_samples=500
```

RCIFAR-100 with deep ensembles:
```sh
python epochs_train.py --dataset='RCIFAR10' --model_name='ResNet-18' --optimizer_name='SGD' --uns_type='ENS' --algs PLBB PBALD Rand LBB BALD MaxProb --random_seeds 42 227 346 684 920 --acq_batch_size=100 --train_batch_size=100 --num_init_samples=2000 --max_train_samples=10000 --num_epochs=50
```

## Experimental setup
Uncertainty estimation options: MC-dropout and deep ensembles. 

Available active learning algorithms: 
- [BALD](https://arxiv.org/abs/1112.5745)
- [PowerBALD](https://arxiv.org/abs/2101.03552)
- [BatchBALD](https://arxiv.org/abs/1906.08158)
- Large BatchBALD (ours)
- Power Large BatchBALD (ours)
- [MaxProb](https://arxiv.org/abs/cmp-lg/9407020)
- Entropy sampling
- Random sampling

Datasets:
- MNIST-based: MNIST, RMNIST, FMNIST, EMNIST, KMNIST
- CIFAR-based: CIFAR-10, CIFAR-100, RCIFAR-10, RCIFAR-100 
- Others: SVHN, AG News (text)

## Citation
```
@article{rubashevskii2023sbadbal,
  title={Scalable Batch Acquisition for Deep Bayesian Active Learning},
  author={Rubashevskii, Aleksandr and Kotova, Daria and Panov, Maxim},
  journal={arXiv preprint arXiv:2301.05490},
  year={2023}
}
```

Big thanks to [**batchbald_redux**](https://github.com/BlackHC/batchbald_redux), our code is partially borrowing from them.
