#!/bin/bash
#
#SBATCH -o test_run_%j
#SBATCH --ntasks=1
#SBATCH --partition gpu_devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=d.kotova@skoltech.ru
#
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=20G
#SBATCH --output=/beegfs/home/d.kotova/parallel_%j.log
#
NAME=DKOTOVA.scf

nvidia-smi

python train.py --dataset='FMNIST' --model_name='CNN_ENS_RMNIST' --uns_type='ENS' --algs PLBB PBALD Rand LBB BALD BB --random_seeds 920 31 --acq_batch_size=10 --num_init_samples=100 --max_train_samples=5000 --num_models=10 
