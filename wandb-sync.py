# util for syncing wandb runs online
# all runs more recent than the
import os
import json
import subprocess
from datetime import datetime
from argparse import ArgumentParser

CP_ROOT="/gpfsscratch/rech/imi/uqv82bm/evaluate_model/"

parser = ArgumentParser()
parser.add_argument('--date', '-d', type=str, default='yesterday', help='format for date is either yyyymmdd or mmdd or'
                                                                        '"yesterday" (default) in which case the date'
                                                                        ' 24 hours ago is used.')
parser.add_argument('--name', '-n', type=str, default='', help='If given, will look for occurences of the run'
                                                               'name in run_id and only sync if the name is '
                                                               'found')

args = parser.parse_args()
if args.date == 'today':
    dt = datetime.now()
    date_id = f'{dt.year:04d}{dt.month:02d}{(dt.day):02d}'
elif args.date == 'yesterday':
    dt = datetime.now()
    date_id = f'{dt.year:04d}{dt.month:02d}{(dt.day - 1):02d}'
elif len(args.date) == 4:
    dt = datetime.now()
    date_id = f'{dt.year:04d}{args.date}'
elif len(args.date) == 8:
    date_id = args.date
else:
    raise ValueError(f"Invalid format for date {args.date}, should be yyyymmdd or mmdd")

wandb_runs = os.listdir("/gpfsscratch/rech/imi/uqv82bm/evaluate_model/wandb/")
os.chdir("/gpfsscratch/rech/imi/uqv82bm/evaluate_model/wandb")
runs_synced = 0
for run in wandb_runs:
    if 'offline-run-' in run:
        date_id_run = run.split('-')[2]
        if date_id_run >= date_id:
            if not args.name:
                exitcode = subprocess.call(['wandb', 'sync', run])
                if exitcode == 0:
                    runs_synced += 1
            else:
                with open(os.path.join(run, 'files', 'wandb-metadata.json')) as f:
                    conf = json.load(f)
                run_args = conf['args']
                run_name = ''
                for arg in run_args:
                    if 'logging.run_id' in arg:
                        run_name = arg.split('=')[1]
                if args.name in run_name:
                    exitcode = subprocess.call(['wandb', 'sync', run])
                    if exitcode == 0:
                        runs_synced += 1

print(f'Successfully synced {runs_synced} runs')