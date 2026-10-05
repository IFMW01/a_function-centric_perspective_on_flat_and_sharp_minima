#!/bin/sh
#SBATCH --job-name=SAM_vis
#SBATCH --mail-user=k20039204@kcl.ac.uk 
#SBATCH --mail-type=ALL
#SBATCH --signal=USR2
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=15:00:00
#SBATCH --output=./logs/%j.out
seed=$SLURM_ARRAY_TASK_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
deactivate

. /scratch/users/k20039204/second-project/bin/activate

cd /scratch/prj/inf_func_transition_nnt/sharpness_diversity

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet18_Base/0/ResNet18_Base.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet18_Base_SAM/0/ResNet18_Base_SAM.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot 

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet18_wd/0/ResNet18_wd.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot 

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet18_wd_SAM/0/ResNet18_wd_SAM.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot 

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet/ResNet18_aug/0/ResNet18_aug.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --raw_data 

! python ./visualization/plot_surface.py --cuda --model resnet18 --x=-1:1:51 --y=-1:1:51 --model_file ./models/ResNet/ResNet18_aug_SAM/0/ResNet18_aug_SAM.pth --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --raw_data

echo Done

