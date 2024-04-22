#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[22-25]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --job-name=less
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/%x-%j.out
#SBATCH --error=./out/%x-%j.out

cd /projets/flowers2/laetitia/code-eval
conda deactivate
conda activate aces

srun python experiments/compute_quality.py --config-name less_plafrim_1