#!/bin/bash
#
#SBATCH --job-name=scheduler
#SBATCH --output=scheduler_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:4

cd ../roadr
source /home/guests/nhigueraruiz/miniconda3/etc/profile.d/conda.sh
conda activate tnormrl
srun bash scheduler.sh