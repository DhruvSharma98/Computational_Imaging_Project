#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=varnetk0_lr_0_1
#SBATCH --time=09:35:00
#SBATCH --mail-user=dhruv.sharma@fau.de
#SBATCH --mail-type=ALL
#
# do not export environment variables
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python/3.8-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate modl

python train.py