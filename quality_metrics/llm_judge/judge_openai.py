from quality_metrics.common import (
    Problem,
)
from quality_metrics.llm_judge.prompt_judge import(
    OpenCodeInterpreter_1, OpenCodeInterpreter_2,
    yes_finetuning, yes_education
)
from quality_metrics.llm_judge.judge_base import Rank_puzzle
from openai import AzureOpenAI,OpenAI

from quality_metrics.common import get_completion, get_multiple_completions, chunks
from tqdm import tqdm

import copy
import torch 
import numpy as np

from quality_metrics.llm_judge.utils_judge import return_proba_yes

# base openai class        
class OpenAI_Rank(Rank_puzzle):
    def __init__(self,puzzle_dict, prompt_instruction: str= None, cfg_openai_client: dict = None,
                 model_id="gpt-3.5-turbo-0125",mode_rank="absolute",n_generation=1,temperature=0, max_workers=20, bs=50,) -> None:
            
        """
        Args:
        - puzzle_dict: a dictionary of puzzles to rank
        - prompt_instruction: the prompt to use for the ranking
        - exllama2: whether to use exllama2
        - model_id: the model_id to use
        - revision: the revision to use
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle
        - azure : whether to use azure or openai
        - openai_key: the openai_key to use
        - cfg_openai_client: the configuration of the openai client
        {"openai_key: str,
        "azure": bool,
        "api_version": "2024-02-15-preview"
        "azure_endpoint" =,... } 
        kwargs:
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"
        - max_workers: the number of workers to use for batch call
        """
        self.max_workers = max_workers
        
     
        self.temperature = temperature
        self.model_id = model_id
        cfg_openai_client = copy.deepcopy(cfg_openai_client)
        if "azure_endpoint" in cfg_openai_client:
            self.model_id = cfg_openai_client["model_id"]
            del cfg_openai_client["model_id"]
            self.cfg_openai_client = cfg_openai_client
            self.azure = True
        else:
            if "api_key" in cfg_openai_client:
                self.openai_key= cfg_openai_client["api_key"]
            else:
                self.openai_key= None
            self.azure = False


        super().__init__(puzzle_dict=puzzle_dict,prompt_instruction=prompt_instruction,mode_rank=mode_rank,n_generation=n_generation,bs=bs)

    def init_model(self):
        self.cfg: dict = {
        "temperature": self.temperature,
        # "top_p": 1.,
        # TODO: rename config option?
        "model": self.model_id,
        "logprobs": False,
        # "top_logprobs": 5,
        "max_tokens": 200,
        }

        max_retries=10
        timeout=10
        if self.azure:
            self.client = AzureOpenAI(**self.cfg_openai_client,max_retries=max_retries, timeout=timeout)
        else:
            self.client = OpenAI(api_key=self.openai_key,max_retries=max_retries, timeout=timeout)

    def generate(self,text):
        """
        if logprobs is True, return the whole completion object c.f get_completion
        else only return test
        """
        out = get_completion(self.client, text, self.cfg)
        return out
    
    def multiple_generation(self,list_text):
        """
        return batch version of generate
        """
        list_out = get_multiple_completions(self.client, list_text, self.cfg, max_workers=self.max_workers)
        return list_out



class OpenCodeInterpreter(OpenAI_Rank):
    def __init__(self,Opencode_mode="1",**kwargs) -> None:
        """ 
        Opencode_mode: "1" or "2" see prompt
        prompt_instruction not used
        """
        if Opencode_mode=="1":
            self.prompt_1= OpenCodeInterpreter_1
        elif Opencode_mode=="2":
            self.prompt_1= OpenCodeInterpreter_2
        super().__init__(**kwargs)
        

    def absolute_grade_one_code(self, code):
        """return the absolute_grade int between 0 and 5"""

        input_single = self.prompt_1.format(query=code)
        n_try=3
        grade=-1
        while n_try>0:
            try:
                out = self.generate(input_single)
                grade=eval(out.split("[")[1].split("]")[0])
                assert grade in [1,2,3,4,5]
                return grade
            except:
                try:
                    grade = eval(out.split("\n")[0].split(":")[1].strip())
                    assert grade in [1,2,3,4,5]
                    return grade
                except:
                    pass
                print(f"Error in the generation of the grade")
                print( out)
                n_try-=1
        return grade
    

    def absolute_grade(self,list_codes):
        """return the list_puzzles int between 0 and 5"""
        from concurrent.futures import ThreadPoolExecutor

        assert isinstance(list_codes, list)
        completions = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sub_batch in chunks(list_codes, self.max_workers):
                for _,message_list in enumerate(sub_batch):
                    kwargs = {"code":message_list}
                    
                    future = executor.submit(
                        self.absolute_grade_one_code,**kwargs
                    )
                    completions.append(future)
        # Retrieve the results from the futures
        grades = [future.result() for future in completions]
        return grades
    


class Yes_model(OpenAI_Rank):
    def __init__(self,yes_mode="finetuning",**kwargs) -> None:
        """ 
        Opencode_mode: "1" or "2" see prompt
        prompt_instruction not used
        """
        self.yes_mode = yes_mode
        self.soft = torch.nn.Softmax(dim=1)
        super().__init__(**kwargs)

        self.cfg['logprobs'] = True
        self.cfg['top_logprobs'] = 5


    def generate(self,text):
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            out_yes = self.model(**inputs)
            k=10
            yes_logits=self.soft(out_yes.logits[:,-1]).cpu().detach() #logits associated with the token "yes"
            values,indices=torch.topk(yes_logits, k)
            list_token=self.tokenizer.batch_decode(indices.T)
            flag_no = False
            if "Yes" in list_token:
                idx = list_token.index("Yes")
                proba_yes = values[[0],idx].item()
            elif "No" in list_token:
                idx = list_token.index("No")
                flag_no = True
                proba_No = values[[0],idx].item()
                if "no" in list_token:
                    idx_no = list_token.index("no")
                    proba_no = values[[0],idx_no].item()
                    if proba_no>proba_No:
                        idx = idx_no
            else:
                print("No yes or no token found")
                return -1
            proba_yes = values[[0],idx].item()
            if flag_no: # if the token "no" is selected, we need to invert the probability
                proba_yes = 1-proba_yes

            proba_yes=values[[0],idx].item()
        return proba_yes
    
    def absolute_grade(self,list_text):
        """return the absolute_grade float between 0 and 10"""
        if self.yes_mode=="education":
            yes_prompt = yes_education
        elif self.yes_mode=="finetuning":
            yes_prompt = yes_finetuning
        else:
            raise ValueError(f"Invalid yes_mode: {self.yes_mode}")
        list_text = [yes_prompt.format(datapoint=txt) for txt in list_text]

        list_completion = self.multiple_generation(list_text) # remove [0] when main loop is batchable

        values = []
        list_words = []
        for completion in list_completion:
            values.append([np.exp(tok.logprob) for tok in completion.logprobs.content[0].top_logprobs])
            list_words.append([tok.token for tok in completion.logprobs.content[0].top_logprobs])

        list_proba_yes=[]
        # values,list_token
        for idx in range(len(list_words)):
            list_proba_yes.append(return_proba_yes(values[idx],list_words[idx]))
        return list_proba_yes
