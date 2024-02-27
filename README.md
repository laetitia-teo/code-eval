# Evaluation scripts for code models 

### Create random splits and finetune models on them

```bash
python dataset_random_split.py
```

Add the `-r` (`--run`) flag to run the experiment immediately (on a cluster).

### Create quality-based splits and finetune models on them

```bash
python dataset_quality_split.py
```

Use the `-q` flag to change the key of the quality metric we use to split (we assume higher is better). 

