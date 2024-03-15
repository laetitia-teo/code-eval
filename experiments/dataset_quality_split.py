"""
This script prepares an experiment where the dataset is split in 2 according to quality.
"""

import os
import json
import pathlib
import subprocess
from collections import defaultdict
from argparse import ArgumentParser

from quality_metrics.common import load_dataset, save_dataset

import numpy as np


parser = ArgumentParser()
parser.add_argument('-d', '--data-path', default='data/dataset_quality_puzzles_train_1.json')
parser.add_argument('--gpu', '-g', default='a100')  # for the expe
parser.add_argument('--keep-data-source', '-k', default=True, action='store_true', 
                    help='Whether to split the different data sources in 2 (True) or whether'
                    ' to split the whole dataset in two directly (False)')
parser.add_argument('--quality-key', '-q', default='pp_diff', help='Adjust this to match the'
                    ' field of the quality score, in case there are several')
parser.add_argument('--test-dataset-path', '-t', default='data/P3_test_emb_wizard3B.json', 
                    help='Path of the test dataset')
parser.add_argument('--run', '-r', action='store_true', default=False, help='Whether to run the'
                    ' experiemnt after creating the datasets')

args = parser.parse_args()

os.chdir(str(pathlib.Path(__file__).parent.parent))  # make sure we are at root of project
print(f'current directory: {os.getcwd()}')

# split dataset and save
# with open(args.data_path, 'r') as f:
#     entire_dataset = json.load(f)
entire_dataset = load_dataset(args.data_path)
entire_dataset = [p for p in entire_dataset if args.quality_key in p.quality]

low_quality_dataset = []
high_quality_dataset = []

def quality_process_fn(quality):  # TODO generalize this to other stuff
    return np.mean(quality)

if args.keep_data_source:
    # split data acording to its source
    data_origins = defaultdict(list)
    for p in entire_dataset:
        data_origins[p.origin].append(p)

    for data in data_origins.values():
        sorted_set = sorted(data, key=lambda x: quality_process_fn(x.quality[args.quality_key]))
        low_quality_dataset += sorted_set[:(len(sorted_set) // 2)]
        high_quality_dataset += sorted_set[(len(sorted_set) // 2):]
else:
    sorted_train_set = sorted(entire_dataset, key=lambda x: quality_process_fn(x.quality[args.quality_key]))
    low_quality_dataset += sorted_train_set[:(len(sorted_train_set) // 2)]
    high_quality_dataset += sorted_train_set[(len(sorted_train_set) // 2):]

dataset_name = pathlib.Path(args.data_path).stem
low_quality_dataset_path = f'data/low_quality_{dataset_name}_{args.quality_key}.json'
high_quality_dataset_path = f'data/high_quality_{dataset_name}_{args.quality_key}.json'

save_dataset(low_quality_dataset, low_quality_dataset_path)
save_dataset(high_quality_dataset, high_quality_dataset_path)

# with open(low_quality_dataset_path, 'w') as f:
#     json.dump(low_quality_dataset, f)
# with open(high_quality_dataset_path, 'w') as f:
#     json.dump(high_quality_dataset, f)

with open('conf/config_template.yaml', 'r') as f:
    config_template = f.read()

config_low_quality = config_template.format(
    test_path=args.test_dataset_path,
    train_path=low_quality_dataset_path,
)
config_high_quality = config_template.format(
    test_path=args.test_dataset_path,
    train_path=high_quality_dataset_path,
)
conf_name_low_quality = f'conf_low_quality_{dataset_name}_{args.quality_key}'
conf_name_high_quality = f'conf_high_quality_{dataset_name}_{args.quality_key}'

with open(f'conf/{conf_name_low_quality}.yaml', 'w') as f:
    f.write(config_low_quality)
with open(f'conf/{conf_name_high_quality}.yaml', 'w') as f:
    f.write(config_high_quality)

# run experiment
if args.run:
    run_id_low_quality = f'split_expe_low_quality_{args.quality_key}'
    run_id_high_quality = f'split_expe_high_quality_{args.quality_key}'
    command_low_quality = (f'python cluster_run.py -g {args.gpu} --config {conf_name_low_quality} '
                    f'--run-id {run_id_low_quality}')
    command_high_quality = (f'python cluster_run.py -g {args.gpu} --config {conf_name_high_quality} '
                    f'--run-id {run_id_high_quality}')

    subprocess.call(command_low_quality, shell=True)
    subprocess.call(command_high_quality, shell=True)
