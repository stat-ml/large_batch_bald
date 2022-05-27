#!/bin/bash
#
#SBATCH -o emnist_%j
#SBATCH --ntasks=1
#SBATCH --partition gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=d.kotova@skoltech.ru
#
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=20G
#SBATCH --output=/beegfs/home/d.kotova/emnist_%j.log
#
NAME=DKOTOVA.scf

nvidia-smi

python train.py --dataset='EMNIST' --model_name='CNN_ENS_EMNIST' --uns_type='ENS' --algs PLBB PBALD Rand LBB BALD --random_seeds 31 42 227 --acq_batch_size=20 --num_init_samples=50 --max_train_samples=3000 --num_models=10 
