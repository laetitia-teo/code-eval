# Prompt absolute ranking of a puzzle

example_puzzle = """```python
def f(li: List[int]):
    return all([li[i] != li[i + 1] for i in range(10)]) and len(set(li)) == 3

def g():
    return list(range(3)) * 10
assert f(g()) == True
```
"""



prompt_instruction_finetuning = """You are an Large Language model (LLM) expert who want to finetuned a model on code. You want to create Python Programming Puzzles to improve a Large Language model in programming.
Python Programming Puzzles (P3) consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in solving a challenging SAT problem `f` with a function `g` such that `f(g())` evaluates to `True`.

Rules:
- f and g should be distinct (if the solution `g` is similar to `f` the information is useless for the model)
- Make sure the problem is challenging for an advance LLM

```python
def f(solution, args=...) -> bool:
    # Python code to test the solution returned by g.
    # This function is a test unit and must return True if the solution is correct, False otherwise.
s
def g(args=...) -> solution:
    # Python code to generate a solution for the problem.
    # The solution should generalize to all possible args.
    return solution

assert f(g()) == True
```
Your Task:
Create a new Python Programming Puzzle that is challenging and educational for an advance LLM in code."""

## prompt from OpenCodeInterpreter https://arxiv.org/abs/2402.14658 

OpenCodeInterpreter_1="""Rate the following code queries on a scale of 1 to 5 based on their complexity, where 1 is the easiest and 5 is the most
difficult. Consider the complexity of the query
Query: [{query}]
You are obliged to choose only from the following list.
Scoring Criteria:
1 Point - Very Basic: The query involves simple operations or common issues
2 Points - Basic: The query involves fundamental programming concepts or commonly used functions
3 Points - Intermediate: The query requires some programming experience, possibly involving multiple steps
4 Points - Difficult: The query involves advanced programming skills, including complex logic, algorithms, or data
structures
5 Points - Very Difficult: The query requires extensive expertise, potentially involving innovative problem-solving
approaches or unique algorithm design
Please give the score first with the format: "Score: [SCORE]" (write the score between bracket) then explain why"""

OpenCodeInterpreter_2= """Rate the following code queries on a scale of 1 to 5 based on their complexity, where 1 is the easiest and 5 is the most
difficult. Consider the complexity of the query 
Query: [{query}]
You are obliged to choose only from the following list.
Scoring Criteria:
1 Point - Moderately Difficult: Involves understanding specific programming concepts or libraries, and may include
medium complexity algorithms or data structures like basic sorting algorithms or tree structures.
2 Points - Challenging: Requires handling more complex logic or algorithms such as advanced sorting algorithms,
recursive logic, or intermediate data structures like hash tables and heaps.
3 Points - Highly Challenging: Demands deeper knowledge in algorithms and data structures, potentially including
graph algorithms, dynamic programming, or complex string manipulation techniques.
4 Points - Advanced: Focuses on proficiency in programming and algorithm design, dealing with complex system
architecture issues, performance optimization, or solving advanced algorithmic challenges like NP-hard problems.
5 Points - Expert Level: The highest difficulty level, requiring innovative problem-solving approaches or unique
algorithm design, possibly involving interdisciplinary knowledge or the application of cutting-edge technologies.
Please give the score first with the format: "Score: [SCORE]" (write the score between bracket) then explain why"""

# yes 
yes_finetuning="""###
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal for fine-tuning a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge about advance programming skills.
OPTIONS:
- Yes
- No
"""

## TODO: change prompt_education to be more specific to the task
yes_education="""This is a educational datapoint to give to students during their exams:
###
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal to a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge of the world.
OPTIONS:
- Yes
- No
"""








# Prompt pairwise ranking of a puzzle


prompt_openchat = """GPT4 Correct User: {instruct}<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:"""
## pormpt prometheus / openchat
instruction_prometheus="""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback:
"""