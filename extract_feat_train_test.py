import argparse
import os
import json
import pickle 
from tqdm import tqdm

parser = argparse.ArgumentParser(description="aaaaaaaaaaaaaaaaaaaaa")
parser.add_argument("-z", "--base_path", type=str, help="path to git project evaluate_model",default="/media/data/flowers/evaluate_model/")
parser.add_argument("-m", "--arg_model_id", type=int, help=" model")

args = parser.parse_args()

os.environ['HF_DATASETS_CACHE'] = args.base_path+"hf/datasets"
os.environ['TRANSFORMERS_CACHE'] = args.base_path+"hf/models"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

import numpy as np
import torch
from transformers import pipeline
list_name_model=["WizardLM/WizardCoder-1B-V1.0","WizardLM/WizardCoder-3B-V1.0","WizardLM/WizardCoder-Python-7B-V1.0","WizardLM/WizardCoder-Python-13B-V1.0",args.base_path+"hf/models/"+"WizardCoder-15B-V1.0"]
model_id=list_name_model[args.arg_model_id]

print("\n=============================\n")
print(model_id)
print("\n=============================\n")

from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16
)
feature_extractor = pipeline("feature-extraction", model=model_id,device_map="auto",quantization_config=quantization_config)
name_model=model_id.split("/")[-1]
# list_all_seed=[]
with torch.no_grad():
    text_test = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
    len_base_prompt = feature_extractor(text_test,return_tensors = "pt").size(1)
#"evaluate_model/run_saved/P3_test.json"
#"evaluate_model/run_saved/P3_train.json"
    list_name_path=["P3_train","P3_test"]
    list_path=["run_saved/P3_train.json","run_saved/P3_test.json"]
    # list_all_embeddings = []
    for idx_path,path in enumerate(list_path):
        with open(args.base_path+path, 'r') as f:
            data = json.load(f)
        # list_emb_path_i=[]
        list_puzzl= [puzz["program_str"] for puzz in data]
        for idx_puzz,puz in enumerate(tqdm(list_puzzl)):
            text= "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"

            new_text=text.format(instruction=puz)
            feat_out=feature_extractor(new_text,return_tensors = "pt")[0][len_base_prompt:]
            out= feat_out.numpy().mean(axis=0) 
            data[idx_puzz]["emb_features"]=out.tolist()
            # list_emb_path_i.append([out])
        # list_emb_path_i=np.array(list_emb_path_i)
        # list_all_embeddings.append(list_emb_path_i)
    # list_all_seed.append(list_all_embeddings)
        with open(args.base_path+"save_feat/"+list_name_path[idx_path]+name_model+'.json', 'w') as f:
            json.dump(data, f)