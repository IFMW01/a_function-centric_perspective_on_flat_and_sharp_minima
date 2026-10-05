#!/bin/sh
#SBATCH --job-name=C10_ResNet_wd
#SBATCH --mail-type=ALL
#SBATCH --signal=USR2
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --array=0-10
seed=$SLURM_ARRAY_TASK_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8


python ./src/main_sharp.py --dataset CIFAR10 --model_name "ResNet18" --seed $seed --num_epochs 100 --save_name "ResNet18_wd" --sharpness True --weight_decay 5e-4
python ./src/main_sharp.py --dataset CIFAR10-C --model_name "ResNet18" --seed $seed --num_epochs 100 --save_name "ResNet18_wd" --sharpness True --weight_decay 5e-4 --corrupt True
echo Done

