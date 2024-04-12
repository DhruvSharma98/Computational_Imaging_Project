#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=zz_final_k1_200_1e5
#SBATCH --time=09:50:00
#SBATCH --mail-user=dhruv.sharma@fau.de
#SBATCH --mail-type=ALL
#
# do not export environment variables
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python/3.8-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate ssdu

python train.py