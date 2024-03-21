def return_prompt_format(model_id, text):
    """
    return the prompt with the correct format for the given model_id

    list model_compatible:['deepseek-coder', 'openchat']
    """
    
    if "deepseek-coder" in model_id:
        prompt_model=prompt_deepseek_coder
    elif "openchat" in model_id:
        prompt_model = prompt_openchat
    else: 
        raise ValueError(f"Model {model_id} not supported")
    
    return prompt_model.format(instruction=text)



prompt_deepseek_coder="""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
### Instruction:
{instruction}
### Response:
"""

prompt_openchat= """GPT4 Correct User: {instruction}<|end_of_turn|>GPT4 Correct Assistant:"""



# utils for GAIR/autoj-13b-GPTQ-4bits
    
PROMPT_INPUT_SYSTEM: str = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'

PROMPT_INPUT_WO_SYSTEM: str = "[INST] {input} [/INST]"

PROMPT_INPUT_FOR_SCENARIO_CLS: str = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"

single = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

pairwise_tie = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""

protocol_mapping = {
    "pairwise_tie": pairwise_tie,
    "single": single,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = protocol_mapping[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg, )

def extract_pairwise_result_autoj(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'): pred_label = 0
        elif pred_rest.startswith('response 2'): pred_label = 1
        elif pred_rest.startswith('tie'): pred_label = 2
    return pred_label

def extract_single_rating_autoj(score_output):
    pred_score =0.
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [["):pos2].strip())
    else: 
        print("Warning: Rating not found in the output")
    return pred_score


# utils for JudgeLM: Fine-tuned Large Language Models are Scalable Judges
# paper: https://arxiv.org/abs/2310.17631
# code: https://github.com/baaivision/JudgeLM

# https://github.com/baaivision/JudgeLM/blob/main/judgelm/llm_judge/common.py
from typing import Optional
import dataclasses
from enum import auto, Enum
import json
from typing import List, Tuple, Any

import torch
from transformers import StoppingCriteria

def parse_score_JudgeLM(review,mode='7b'):
    """
    anwser format:
    for 13b mode 
        ### Response:
        Assistant 1: 7
        Assistant 2: 8

    """

    try:
        if mode == '7b':
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
        elif mode == '13b':
            score1, score2 = -1, -1
            for line in review.split('\n'):
                if "Assistant 1:" in line:
                    score1 = int(line.split(":")[1])
                elif "Assistant 2:" in line:
                    score2 = int(line.split(":")[1])
                if score1 != -1 and score2 != -1:
                    return [score1, score2]
            return [score1, score2]
        else:
            print("mode not supported")
            # print("review: ", review)
            # raise Exception('Invalid score pair.')
            raise Exception()
            pass
    except Exception as e:
        # print(f'{e}\nContent: {review}\n'
        #              'You must manually fix the score pair.')
        return [-1, -1]

def translate_score_to_win_list_JudgeLM(score_list, T=0.0):
    win_list = []
    for i in range(len(score_list)):
        if score_list[i][0] - score_list[i][1] > T:
            win_list.append(1)
        elif score_list[i][1] - score_list[i][0] > T:
            win_list.append(0)
        else:
            win_list.append(0.5)
    return win_list

def translate_score_to_win_list(score_list, T=0.0):
    win_list = []
    for i in range(len(score_list)):
        if score_list[i][0] - score_list[i][1] > T:
            win_list.append(1)
        elif score_list[i][1] - score_list[i][0] > T:
            win_list.append(-1)
        else:
            win_list.append(0)
    return win_list

def generate_question_template(domain, question1, question2):
    Q = ("Human", "Provide a question in [" + domain + "] domain just like {" + question1 + "}, your provided question must be different from the questions that we have mentioned in this conversation.")
    A = ("Assistant", "Certainly! Here's another question in a [" + domain + "] domain: {" + question2 + "}")
    return (Q, A)

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

# change from original repo stopping batch mode
class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword, tokenizer, len_prompt):
        self.keyword = keyword
        self.tokenizer = tokenizer
        self.start_len = None
        self.len_prompt = len_prompt

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.len_prompt
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            for out_text in outputs:
                # print("keyword= ",self.keyword, "out_test= ",out_text)
                if not(self.keyword in out_text): # if one of the output does not contain the keyword, do not stop
                    return False
            return True
        return False
    
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, list_keywords, tokenizer, len_prompt):
        self.list_keywords = list_keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.len_prompt = len_prompt

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.len_prompt
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            for out_text in outputs:
                # print("keyword= ",self.keyword, "out_test= ",out_text)
                for key in self.list_keywords:
                    if not(key in out_text):
                        return False
            return True
        return False
    
class KeywordsStoppingCriteriaOrdered(StoppingCriteria):
    def __init__(self, list_keywords, tokenizer, len_prompt):
        self.list_keywords = list_keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.len_prompt = len_prompt

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.len_prompt
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            for out_text in outputs:
                # print("keyword= ",self.keyword, "out_test= ",out_text)
                out_raw= out_text
                for key in self.list_keywords:
                    idx_key=out_raw.find(key)
                    out_raw=out_raw[idx_key:]
                    if not(key in out_text):
                        return False
            return True
        return False
# create num to words dict
num2words = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five",
             6:"six", 7:"seven", 8: "eight", 9: 'nine', 10: 'ten', \
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    prompt: str
    prompt_template: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    appendix: str = "### Response: "

    skip_next: bool = False
    conv_id: Any = None
    answer_format: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self, answer_num):
        if answer_num is not None:
            prompt = self.prompt\
                .replace("two", num2words[int(answer_num)])\
                .replace("for Assistant 1 and 2", "for Assistant 1")
            
            plug_in_after_str = "for Assistant 1"
            plug_in_pos = prompt.find(plug_in_after_str) + len(plug_in_after_str)

            new_answer = ""
            for i in range(int(answer_num)-2):
                new_answer += f", {i+2}"
            new_answer += f" and {int(answer_num)}"
            prompt = prompt[:plug_in_pos] + new_answer + prompt[plug_in_pos:]
        else:
            prompt = self.prompt
        return Conversation(
            system=self.system,
            roles=self.roles,
            prompt=prompt,
            prompt_template=self.prompt_template,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            appendix=self.appendix,
            conv_id=self.conv_id,
            answer_format=self.answer_format)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "prompt": self.prompt,
            "prompt_template": self.prompt_template,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "appendix": self.appendix,
            "conv_id": self.conv_id,
        }

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


conv_judge_pair = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:"
)

conv_judge_pair_w_reference = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:"
)

conv_judge_multi = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n{answer_more}[System]\n{prompt}\n\n",
    answer_format = "[The Start of Assistant {answer_id}'s Answer]\n{answer}\n\n[The End of Assistant {answer_id}'s Answer]\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:"
)

conv_judge_multi_w_reference = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n{answer_more}[System]\n{prompt}\n\n",
    answer_format = "[The Start of Assistant {answer_id}'s Answer]\n{answer}\n\n[The End of Assistant {answer_id}'s Answer]\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:"
)

conv_judge_single = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:" #problem here it should not have 10 ???" ### Response:10"
)


conv_judge_single_w_reference = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:10"
)

conv_judge_vqa_single_answer = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants' answers in comparing with the reference answer and determine if they match meaningfully. Please rate the helpfulness, relevance, accuracy, level of details of the Assistants' answers. To accomplish the task, you must : 1. Focus on the meaningful match between the reference answer and the Assistants' answers.\n 2. Consider synonyms or paraphrases as valid matches.\n 3. Evaluate the correctness of the Assistants' answers compared to the reference answer.\n 4. If there are multiple reference answers, the Assistants' answer is considered correct as long as it is close to any of the answers.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:10"
)

def return_judgeLM_prompt(conv: Conversation, question, references: str = None)-> str :
    """"""
    template = conv.prompt_template
    if references is None:
        data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                            answer_1=question['answer1_body'],
                                                            answer_2=question['answer2_body'],
                                                            prompt=conv.prompt) + conv.appendix
    else:

        data_sample = conv.system + '\n' + template.format(question=question['question_body'],
                                                            reference=references['reference']['text'],
                                                            answer_1=question['answer1_body'],
                                                            answer_2=question['answer2_body'],
                                                            prompt=conv.prompt) + conv.appendix + "\n\n"
    return data_sample
