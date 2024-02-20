#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=sft3b-low_fitness
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/finetune-coder-low_fitness.out
#SBATCH --error=./out/finetune-coder-low_fitness.out

module load python/3.11.5
conda deactivate
module purge
module load cuda/12.1.0
conda activate aces

cd $WORK/evaluate_model 

python test_finetuned_rework.py -z $WORK/evaluate_model/ -p 'puzzles_low_fitness_archivetrain.json' -e 2 -c 16 -b 8 -s "1" -t "train_eval"