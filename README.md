# Evaluation scripts for code models 

### Create random splits and finetune models on them

```bash
python experiments/dataset_random_split.py
```

Add the `-r` (`--run`) flag to run the experiment immediately (on a cluster).

### Create quality-based splits and finetune models on them

```bash
python experiments/dataset_quality_split.py
```

Args:

```
-d --data-path path to the desired dataset
-g --gpu preferred gpu (one of {a100, v100})
-q --quality-key what quality key to use for the split
-r --run whether to run the experiment with slurm immediately after the split
```

Use the `-q` flag to change the key of the quality metric we use to split (we assume higher is better). 

### Train a model

```bash
python run.py --config-name {conf}
```

