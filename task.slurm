#!/bin/bash
#
#SBATCH -o process_train_%j
#SBATCH --ntasks=1
#SBATCH --partition gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aleksandr.rubashevskii@skoltech.ru
#
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=20G
#SBATCH --output=/beegfs/home/a.rubashevskii/parallel_%j.log
#
NAME=ALEXRUBASHEVSKY.scf

nvidia-smi

python train.py --dataset='CIFAR10' --model_name='CNN_ENS_CIFAR10' --uns_type='ENS' --algs PLBB PBALD Rand LBB BALD BB --random_seeds 920 --acq_batch_size=10 --num_init_samples=1000 --max_train_samples=10000
