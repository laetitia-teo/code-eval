import os
import argparse
import subprocess
from datetime import datetime


# TODO do this with hydra as well, share config files
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='a100')  # set to v100 for a v100 partition
parser.add_argument('--config', '-c', default='base_config')
parser.add_argument('--run-id', '-r', default=None)
args = parser.parse_args()

# detect cluster
cwd = os.getcwd()
if 'gpfs' in cwd:
    cluster = 'jz'
elif 'projets' in cwd:
    cluster = 'plafrim'
else:
    cluster = 'cleps'  # TODO add gcp

# create run id
if args.run_id is not None:
    run_id = str(datetime.now()).replace(' ', '_')

# create and run slurm file
match cluster, args.gpu:
    case 'jz', 'a100':
        template = open('slurm/slurm_templates/jz_template.slurm').read()
    case 'jz', 'v100':
        template = open('slurm/slurm_templates/jz_template_v100.slurm').read()
    case 'plafrim', _:
        template = open('slurm/slurm_templates/plafrim_template.slurm').read()
    case 'cleps', _:
        raise NotImplementedError
    case _:
        raise NotImplementedError

script = f"""
module load python/3.11.5
conda deactivate
module purge
module load cuda/12.1.0
conda activate aces

cd $WORK/code-eval

srun python run.py --config-name {args.config} run_id={run_id}
"""

slurm_file = template.format(
    run_id=run_id,
    hours=20,
    script=script
)

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

slurmfile_path = f'slurm/slurm_files/run_{args.config}.slurm'
with open(slurmfile_path, 'w') as f:
    f.write(slurm_file)

if not os.path.exists('out'):
    os.makedirs('out')

subprocess.call(f'sbatch {slurmfile_path}', shell=True)
