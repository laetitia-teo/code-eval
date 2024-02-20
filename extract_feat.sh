#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=sft3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0,1,2,3,4,5   
#SBATCH --output=./out/out_finetune_llama3b-%A_%a.out
#SBATCH --error=./out/out_finetune_llama3b-%A_%a.out
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.0.1
cd $SCRATCH/evaluate_model 

python extract_feat.py -z $SCRATCH/evaluate_model/ -m ${SLURM_ARRAY_TASK_ID} -f full
