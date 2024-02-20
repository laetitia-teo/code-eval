#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[22-25]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --job-name={run_id}
#SBATCH --hint=nomultithread
#SBATCH --time={hours}:00:00
#SBATCH --output=./out/{run_id}.out
#SBATCH --error=./out/{run_id}.out

{script}