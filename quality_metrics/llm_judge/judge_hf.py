
from tqdm import tqdm
from typing import Tuple 
import torch 
import numpy as np

from quality_metrics.llm_judge.judge_base import Rank_puzzle
from quality_metrics.llm_judge.prompt_judge import OpenCodeInterpreter_1, OpenCodeInterpreter_2,yes_finetuning,yes_education
from quality_metrics.llm_judge.utils_judge import return_proba_yes
from quality_metrics.llm_judge.utils_hf import (return_prompt_format,
    extract_single_rating_autoj,
    extract_pairwise_result_autoj,
    build_autoj_input
)
from quality_metrics.common import (
    QualityMetric,
    Problem,
    create_model_and_tokenizer,
)


# [Yes_model, Auto_j_Rank, JudgeLM

# HF model
class HF_Rank(Rank_puzzle):
    def __init__(self, puzzle_dict,prompt_instruction,mode_rank="pairwise",exllama2=False,model_id=None,
                 revision="main",n_generation=4,bs=2,temperature=0.0001,top_p=1.,max_new_tokens=1024,load_in_4bit=False,load_in_8bit=False
) -> None:
        """
        Args:
        - puzzle_dict: a dictionary of puzzles to rank
        - prompt_instruction: the prompt to use for the ranking
        - exllama2: whether to use exllama2
        - model_id: the model_id to use
        - revision: the revision to use
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle

        kwargs:
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"

        """
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens=max_new_tokens
        self.exllama2 = exllama2        
        self.model_id = model_id
        self.revision = revision
        super().__init__(puzzle_dict=puzzle_dict,prompt_instruction=prompt_instruction,mode_rank=mode_rank,n_generation=n_generation,bs=bs)

    def init_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        path_model=self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(path_model,revision = self.revision)
        self.tokenizer.padding_side="left"

        if self.exllama2:
            gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto",quantization_config=gptq_config,revision = self.revision)
            # from auto_gptq import exllama_set_max_input_length
            # print("self.bs: ",self.bs)
            # self.model = exllama_set_max_input_length(self.model, max_input_length=self.bs*1024)

        else:
            kwargs = {}
            if self.load_in_4bit or self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                if self.load_in_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    kwargs["quantization_config"]=bnb_config
                if self.load_in_8bit:

                    kwargs["load_in_8bit"]=True

            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto",revision = self.revision,**kwargs)
        

    def prompt_format(self, text):
        """
        return the prompt format for the model system,user,...
        """
        return return_prompt_format(self.model_id, text)
    
    def generate(self,list_text): 
        #TODO: text -> list_text
        with torch.inference_mode():
            inputs = self.tokenizer(list_text, return_tensors="pt",padding=True).to("cuda")
            len_sequence = inputs.input_ids.shape[1]
            out_tok = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature = self.temperature, top_p=self.top_p)
            list_out = self.tokenizer.batch_decode(out_tok[:,len_sequence:], skip_special_tokens=True) # only keep completion
        return list_out
    


class Yes_model(HF_Rank):
    def __init__(self, puzzle_dict,mode_rank="absolute",prompt_instruction=None,exllama2=False,model_id="/home/flowers/work/hf/deepseek-coder-1.3b-instruct",yes_mode="finetuning",n_generation=1,bs=2,debug=False) -> None:
        """
        yes_mode = ["finetuning","education"] #prompt to use for the ranking
        """
        self.debug = debug
        self.exllama2 = exllama2
        self.yes_mode = yes_mode # "finetuning" or "education"
        self.soft = torch.nn.Softmax(dim=-1)
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,exllama2=exllama2,n_generation=n_generation,bs=bs)
        
        
    def generate(self,list_text: list[str]):
        assert isinstance(list_text,list)
        with torch.inference_mode():
            inputs = self.tokenizer(list_text, return_tensors="pt",padding=True).to("cuda")
            out_yes = self.model(**inputs)
            # out = self.tokenizer.decode(out_tok[0])
            k=10
            yes_logits=self.soft(out_yes.logits[:,-1]).cpu().detach() #logits associated with the token "yes"
            values,indices=torch.topk(yes_logits, k)
            list_words=self.tokenizer.batch_decode(indices.flatten())
            list_words=np.array(list_words).reshape(values.shape).tolist()
            values = values.tolist()
            list_proba_yes=[]
            # values,list_token
            for idx in range(len(list_words)):
                if self.debug:
                    print("-----")
                    for j in range(len(list_words[idx])):
                        print(f"list_words[idx][j]: {list_words[idx][j]}, values[idx][j]: {values[idx][j]}")
                list_proba_yes.append(return_proba_yes(values[idx],list_words[idx]))
        return list_proba_yes
    
    def absolute_grade(self,list_text: list[str]):
        """return the absolute_grade float between 0 and 10"""
        assert isinstance(list_text,list) 
        # query = self.prompt_instruction
        # yes_education,yes_finetuning
        if self.yes_mode=="education":
            yes_prompt = yes_education
        elif self.yes_mode=="finetuning":
            yes_prompt = yes_finetuning
        else:
            raise ValueError(f"Invalid yes_mode: {self.yes_mode}")
        for idx in range(len(list_text)):
            list_text[idx] = self.prompt_format(yes_prompt.format(datapoint=list_text[idx]))

        out = self.generate(list_text) # remove [0] when main loop is batchable
        return out




    
# TODO: evaluate with GAIR/autoj-13b-GPTQ-4bits 

class Auto_j_Rank(HF_Rank):
    def __init__(self, puzzle_dict,mode_rank="pairwise",prompt_instruction=None,exllama2=True,model_id="GAIR/autoj-13b-GPTQ-4bits",n_generation=1,bs=2) -> None:
        self.exllama2 = exllama2
        assert "autoj" in model_id # only for autoj model (quantized or not)
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,n_generation=n_generation,bs=bs,exllama2=exllama2)
        self.exllama2 = exllama2
        
    def pairwise_ranking(self, list_puzzle) -> list[int]:
        """return the winner (puzzle1 or puzzle2)"""
        query = self.prompt_instruction
        list_prompt=[]
        for i in range(len(list_puzzle)):
            resp1 = list_puzzle[i][0]
            resp2 = list_puzzle[i][1]
            inputs_pairwise = build_autoj_input(prompt=query, 
                        resp1 = resp1,  resp2 = resp2, 
                        protocol = "pairwise_tie") # for pairwise response comparison 
            list_prompt.append(inputs_pairwise)
        out = self.generate(list_prompt)
        results_pairwise=[]
        for i in range(len(out)):
            results_pairwise.append(extract_pairwise_result_autoj(out[i]))
        return results_pairwise
    
    def absolute_grade(self,list_puzzle) -> list[int]:
        """return the absolute_grade float between 0 and 10"""
        query = self.prompt_instruction
        list_prompt=[]
        for i in range(len(list_puzzle)):
            resp1 = list_puzzle[i]
            inputs_single = build_autoj_input(prompt=query, 
                        resp1 = resp1, resp2=None, 
                        protocol = "single") # for single response evaluation 
            list_prompt.append(inputs_single)
        out = self.generate(list_prompt)

        results_single=[]
        for i in range(len(out)):
            results_single.append(extract_single_rating_autoj(out[i]))

        return results_single

# JudgeLM: Fine-tuned Large Language Models are Scalable Judges
# paper: https://arxiv.org/abs/2310.17631
# code: https://github.com/baaivision/JudgeLM
# WIP 
    
from quality_metrics.llm_judge.utils_hf import (
    KeywordStoppingCriteria,conv_judge_pair,
    conv_judge_pair_w_reference,
    parse_score_JudgeLM,
    translate_score_to_win_list_JudgeLM,
    return_judgeLM_prompt
)
from quality_metrics.llm_judge.prompt_judge import example_puzzle

class JudgeLM(HF_Rank):
    def __init__(self, puzzle_dict,prompt_instruction,example_puzzle=example_puzzle,fast_eval=True,mode_rank="pairwise",exllama2=False,model_id="/home/flowers/work/hf/JudgeLM-7B-v1.0",n_generation=1,bs=2,load_in_4bit=False,load_in_8bit=False) -> None:
        """
        example_puzzle: puzzle use for absolute grading grading 
        """
        self.example_puzzle=example_puzzle
        self.exllama2 = exllama2
        self.explanation = []
        if not "JudgeLM" in model_id:
            print("Warning: LLM model used should be a part of 'JudgeLM' model")
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,n_generation=n_generation,bs=bs,exllama2=exllama2,load_in_4bit=load_in_4bit,load_in_8bit=load_in_8bit)
        self.exllama2 = exllama2
        self.fast_eval = fast_eval # if True, just give score without explanation, if False, give explanation

    def generate(self,list_prompt: list[str],eos_string="\n"):
        assert isinstance(list_prompt,list)
        with torch.inference_mode():
            do_sample = False if self.temperature < 1e-4 else True
            inputs = self.tokenizer(list_prompt,return_tensors="pt",padding=True)
            len_prompt = inputs.input_ids.shape[1]
            inputs=inputs.to("cuda")
            stopping_criteria = KeywordStoppingCriteria(eos_string, self.tokenizer, len_prompt)
            output_ids = self.model.generate(
                        **inputs,
                        do_sample=do_sample,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        
                        stopping_criteria=[stopping_criteria]
                    )
            outputs =self.tokenizer.batch_decode(output_ids[:, len_prompt:], skip_special_tokens=True)
            if eos_string:
                if not self.fast_eval:
                    for i in range(len(list_prompt)):

                        self.explanation.append({"prompt":list_prompt[i],"output":outputs[i],"full" : list_prompt[i]+"\n"+outputs[i]})
                    outputs = [out[: out.find(eos_string)].strip() for out in outputs]
        return outputs

    def pairwise_ranking(self, list_puzzle) -> list[int]:
        """return the winner (puzzle1 or puzzle2)"""
        references = None
        conv = conv_judge_pair.copy(None) if references is None else conv_judge_pair_w_reference.copy(None)
        if self.fast_eval:
            conv.sep = "\n"
        query = self.prompt_instruction
        if query == None:
            query = "Grade the quality of those two answer."
        questions= [{"question_body": query, "answer1_body": list_puzzle[i], "answer2_body": list_puzzle[i][1]} for i in range(len(list_puzzle))]
        list_prompt=[]
        for question in questions:
            data_sample = return_judgeLM_prompt(conv,question)
            list_prompt.append(data_sample)

        out = self.generate(list_prompt,eos_string=conv.sep)
        results_pairwise_score=[]
        for i in range(len(out)):
            results_pairwise_score.append(parse_score_JudgeLM(out[i])) # for absolute ranking just get that with a ref puzzles.
        results_pairwise = translate_score_to_win_list_JudgeLM(results_pairwise_score)
        return results_pairwise
    
    def absolute_grade(self,list_puzzle) -> list[int]:

    # return [str(8**88)[i:i+8] for i in range(0,80,8)]

        """return the absolute_grade float between 0 and 10"""
        query = self.prompt_instruction
        references = None
        conv = conv_judge_pair.copy(None) if references is None else conv_judge_pair_w_reference.copy(None)
        if self.fast_eval:
            conv.sep = "\n"
        query = self.prompt_instruction
        if query == None:
            query = "Grade the quality of those two answer."
        questions= [{"question_body": query, "answer1_body": list_puzzle[i], "answer2_body": self.example_puzzle} for i in range(len(list_puzzle))]
        list_prompt=[]
        for question in questions:
            data_sample = return_judgeLM_prompt(conv,question)
            list_prompt.append(data_sample)

        out = self.generate(list_prompt,eos_string=conv.sep)
        results_absolute_score=[]
        for i in range(len(out)):
            results_absolute_score.append(parse_score_JudgeLM(out[i])[0]) # for absolute ranking just get that with a ref puzzles.
        return results_absolute_score



# # TODO: finish openchat ranking -> rename prometheus

# class Open_chat(HF_Rank):
#     def __init__(self, puzzle_dict,mode_rank="absolute",prompt_instruction=None,exllama2=True,model_id="TheBloke/openchat-3.5-1210-GPTQ",revision="gptq-4bit-32g-actorder_True", n_generation=4) -> None:
#         self.exllama2 = exllama2
#         super().__init__(model_id,puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,revision=revision, n_generation = n_generation)
    
#     def absolute_grade(self,puzzle):
#         """return the absolute_grade float between 0 and 5"""
#         query = self.prompt_instruction
#         resp1 = puzzle
#         instruct = instruction_openchat.format(orig_instruction=query,orig_response=resp1,orig_reference_answer="...",orig_criteria="...",
#                                     orig_score1_description="...",orig_score2_description="...",orig_score3_description="...",
#                                     orig_score4_description="...",orig_score5_description="...")
#         input_single = prompt_openchat.format(instruct=instruct)
#         out = self.generate(input_single)
#         raise NotImplementedError # need to extract the score
#         return  