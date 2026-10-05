#SBATCH --job-name=significance
#SBATCH --mail-type=ALL
#SBATCH --signal=USR2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
export CUBLAS_WORKSPACE_CONFIG=:4096:8
deactivate

cd /scratch/prj/inf_func_transition_nnt/sharpness_diversity/
python significance.py
echo Done



