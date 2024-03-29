"""
Performs several times an experiemnts where we arbitrarily split in 2 a dataset, 
and do finetuning and eval on each one.
"""

import os
import json
import pathlib
import subprocess
from argparse import ArgumentParser

from quality_metrics.common import load_dataset, save_dataset

import numpy as np


parser = ArgumentParser()
parser.add_argument('-d', '--data-path', default='data/p3_pb_dataset.json')
parser.add_argument('-s', '--seed', type=int, default=None)
parser.add_argument('--gpu', '-g', default='a100')  # for the expe
parser.add_argument('--quality-key', '-q', default='pp_diff', help='Adjust this to match the'
                    ' field of the quality score, in case there are several')
parser.add_argument('--run', '-r', action='store_true', default=False, help='Whether to run the'
                    ' experiemnt after creating the datasets')

test_dataset_path = 'data/P3_test_emb_wizard3B.json'

args = parser.parse_args()

if args.seed is None:
    seed = np.random.randint(2**31)
else:
    seed = args.seed

os.chdir(str(pathlib.Path(__file__).parent.parent))  # make sure we are at root of project
print(f'current directory: {os.getcwd()}')

print(f'seed = {seed}')

# split dataset and save
entire_dataset = load_dataset(args.data_path)
# entire_dataset = [p for p in entire_dataset if args.quality_key in p.quality]

indices_first = np.random.choice(len(entire_dataset), len(entire_dataset) // 2, replace=False)
indices_second = [i for i in range(len(entire_dataset)) if i not in indices_first]

first_dataset = [entire_dataset[i] for i in indices_first]
second_dataset = [entire_dataset[i] for i in indices_second]
dataset_name = pathlib.Path(args.data_path).stem

first_dataset_path = f'data/first_{dataset_name}_{seed}.json'
second_dataset_path = f'data/second_{dataset_name}_{seed}.json'

save_dataset(first_dataset, first_dataset_path)
save_dataset(second_dataset, second_dataset_path)

# create config files
with open('conf/config_template.yaml', 'r') as f:
    config_template = f.read()

config_first = config_template.format(
    test_path=test_dataset_path,
    train_path=first_dataset_path,
    quality_key='null',
)
config_second = config_template.format(
    test_path=test_dataset_path,
    train_path=second_dataset_path,
    quality_key='null',
)
conf_name_first = f'conf_first_{dataset_name}_{seed}'
conf_name_second = f'conf_second_{dataset_name}_{seed}'

with open(f'conf/{conf_name_first}.yaml', 'w') as f:
    f.write(config_first)
with open(f'conf/{conf_name_second}.yaml', 'w') as f:
    f.write(config_second)
    
# run experiment
if args.run:
    run_id_first = f'split_expe_first_{seed}'
    run_id_second = f'split_expe_second_{seed}'
    command_first = (f'python cluster_run.py -g {args.gpu} --config {conf_name_first} '
                    f'--run-id {run_id_first}')
    command_second = (f'python cluster_run.py -g {args.gpu} --config {conf_name_second} '
                    f'--run-id {run_id_second}')

    subprocess.call(command_first, shell=True)
    subprocess.call(command_second, shell=True)

