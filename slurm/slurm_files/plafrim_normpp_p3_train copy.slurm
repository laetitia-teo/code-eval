#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[22-25]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --job-name=normpp_p3_train
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/normpp_p3_train-%j.out
#SBATCH --error=./out/normpp_p3_train-%j.out

cd /projets/flowers2/laetitia/code-eval
conda deactivate
conda activate aces

srun python experiments/compute_quality.py --config-name plafrim_normpp_p3_train