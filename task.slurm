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

python newpipe_train.py --dataset='RCIFAR10' --model_name='ResNet-18' --optimizer_name='SGD' --uns_type='ENS' --algs BALD --random_seeds 920 --num_models=5 --acq_batch_size=200 --train_batch_size=200 --num_init_samples=2000 --max_train_samples=10000 --num_epochs=50
