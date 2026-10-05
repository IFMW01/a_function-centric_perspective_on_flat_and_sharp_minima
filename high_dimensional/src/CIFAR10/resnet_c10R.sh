#!/bin/sh
#SBATCH --job-name=C10_ResNet_random
#SBATCH --mail-type=ALL
#SBATCH --signal=USR2
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --array=0-10
seed=$SLURM_ARRAY_TASK_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

#Make sure to change rand_prob, it ranges from 0.0-->1.0.

python ./src/main_sharp.py --dataset  CIFAR10R --model_name "ResNet18" --seed $seed --num_epochs 100 --save_name "ResNet18_Base_R100" --sharpness True --rand_prob 1.0
echo Done



