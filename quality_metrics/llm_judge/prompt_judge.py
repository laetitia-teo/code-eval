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


Educ_advanced ="""You are a discerning and objective assistant tasked with evaluating the difficulty and complexity of Python Programming Puzzles for candidates applying for a Google software engineer position. Your role is crucial in ensuring the puzzle's difficulty is accurately gauged to maintain the integrity of our evaluation process.

Each puzzle should be rated on a scale from 1 to 10, where:
- A score of 1 signifies a puzzle that is extremely straightforward, involving elementary operations (e.g., simple addition).
- A score of 10 denotes a puzzle of maximal difficulty and complexity, requiring advanced algorithmic problem-solving and deep understanding of Python programming concepts.

We encourage you to utilize the full range of the scale to differentiate between the levels of difficulty effectively. Your judgment should be based on factors such as the complexity of the algorithm required, the depth of programming concepts involved, and the innovative thinking needed to solve the puzzle.

Please provide a score (an integer) that best represents the puzzle's difficulty. We rely on your expertise and judgment to use the entire scoring range appropriately to ensure a fair and accurate assessment.

You must only give a score with the following format: "Score: "

## Question:
{puzzle}

## Score:
"""

"""
You are a discerning and objective assistant tasked with evaluating the difficulty and complexity of Python Programming Puzzles for candidates applying for a Google software engineer position. Your role is crucial in ensuring the puzzle's difficulty is accurately gauged to maintain the integrity of our evaluation process.
Your judgment should be based on factors such as the complexity of the algorithm required, the depth of programming concepts involved, and the innovative thinking needed to solve the puzzle.
We rely on your expertise and judgment to use the entire scoring range appropriately to ensure a fair and accurate assessment.
You must only give a score with the following format: "Score: "
"""
pairwise_tie_educ_advanced = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. 
You are a discerning and objective assistant tasked with evaluating the difficulty and complexity of Python Programming Puzzles for candidates applying for a Google software engineer position. Your role is crucial in ensuring the puzzle's difficulty is accurately gauged to maintain the integrity of our evaluation process.
Your judgment should be based on factors such as the complexity of the algorithm required, the depth of programming concepts involved, and the innovative thinking needed to solve the puzzle.
We rely on your expertise and judgment to ensure a fair and accurate assessment.

Here is the data:

[BEGIN DATA]
***
[Puzzle 1]:
```
{puzzle1}
```
***
[Puzzle 2]:
```
{puzzle2}
```
***
[END DATA]

You Must format your response as follows:
- If Puzzle 1 is better than Puzzle 2, respond with "The decision is Puzzle 1".
- If Puzzle 2 is better than Puzzle 1, respond with "The decision is Puzzle 2".
- If both puzzles are equally good, respond with "The decision is Tie".
"""
pairwise_tie_educ_advanced_2="""You are a discerning and analytical assistant tasked with evaluating the relative difficulty and complexity of pairs of Python Programming Puzzles. These puzzles are designed for candidates applying for a Google software engineer position. Your insights are critical in determining which puzzle of each pair presents a greater challenge to the candidate, thereby helping to refine our selection process.

For each pair of puzzles, you should assess which one is more difficult and complex, considering factors such as the sophistication of the algorithm required, the depth of programming concepts involved, and the level of innovative thinking needed to solve the puzzle.

Please choose between Puzzle A and Puzzle B, indicating which one you believe represents a greater challenge. Additionally, if you think the difficulty is equal and represents a significant challenge in their own ways, you may indicate a tie. Use the scale provided to guide your decision:

- If Puzzle A is better than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B is better than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles are equally good, respond with "The decision is Tie.".

Your evaluation will ensure that our assessment process accurately reflects the varying levels of puzzle difficulty, aiding in the fair and precise selection of candidates.

## Puzzle A:
{puzzle_a}

## Puzzle B:
{puzzle_b}

Response Format:
### Decision:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""
pairwise_tie_educ_advanced_cot="""You are a discerning and analytical assistant tasked with evaluating the relative difficulty and complexity of pairs of Python Programming Puzzles. These puzzles are designed for candidates applying for a Google software engineer position. Your insights are critical in determining which puzzle of each pair presents a greater challenge to the candidate, thereby helping to refine our selection process.

For each pair of puzzles, you should assess which one is more difficult and complex, considering factors such as the sophistication of the algorithm required, the depth of programming concepts involved, and the level of innovative thinking needed to solve the puzzle.

Please choose between Puzzle A and Puzzle B, indicating which one you believe represents a greater challenge. Additionally, if you think the difficulty is equal and represents a significant challenge in their own ways, you may indicate a tie. Use the scale provided to guide your decision:

- If Puzzle A is better than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B is better than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles are equally good, respond with "The decision is Tie.".

Your evaluation will ensure that our assessment process accurately reflects the varying levels of puzzle difficulty, aiding in the fair and precise selection of candidates.

## Puzzle A:
{puzzle1}

## Puzzle B:
{puzzle2}

Response Format:
### Puzzle compairson:
(One or two sentence to compare the two Puzzles)
### Conclusion:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""
def parse_pairwise_tie_cot(s):
    s=s.lower()
    try:
        response=s.split("the decision")[1].split(".")[0]

        if "puzzle a" in response:
            return 0
        elif "puzzle b" in response:
            return 1
        elif "tie" in response:
            return 2
    except:
        pass
    try:
        if "conclusion" in s:
            response =s.split("conclusion")[1].split(".")[0]
        else:
            response=s.split("decision")[1].split(".")[0]
        if "puzzle a" in response:
            return 0
        elif "puzzle b" in response:
            return 1
        elif "tie" in response:
            return 2
        
    except:
        pass
    return 2

pairwise_high_educational_value="""You are an insightful and evaluative assistant tasked with assessing the educational value of pairs of Python Programming Puzzles. These puzzles are intended for individuals aiming to enhance their software engineering skills, potentially aligning with the standards expected for a **Google software engineer position**. Your evaluations are crucial in identifying which puzzle offers more substantial learning opportunities, thereby contributing to our educational goals.

For each pair of puzzles, you should determine which one provides a richer educational experience. This includes considering factors such as the variety of programming concepts taught, the depth of understanding required, and the opportunity for innovative problem-solving and critical thinking.

Please choose between Puzzle A and Puzzle B, indicating which one you believe offers more significant educational value. Additionally, if you find that both puzzles are equally beneficial in their educational content and challenge, you may indicate a tie. Use the guidelines below to inform your decision:

- If Puzzle A offers greater educational value than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B offers greater educational value than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles offer comparable educational value, respond with "The decision is Tie.".

Your discerning evaluation will help us curate puzzles that not only challenge but also effectively educate our audience, ensuring a meaningful and enriching learning experience.

## Puzzle A:
{puzzle_a}

## Puzzle B:
{puzzle_b}

Response Format:
### Decision:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""

pairwise_low_educational_value="""You are an insightful and evaluative assistant tasked with assessing the educational value of pairs of Python Programming Puzzles. These puzzles are intended for individuals aiming to start learning software engineering skills, potentially aligning with the standards expected for a **beginner in Python**. Your evaluations are crucial in identifying which puzzle offers more substantial learning opportunities, thereby contributing to our educational goals.

For each pair of puzzles, you should determine which one provides a richer educational experience. This includes considering factors such as the variety of programming concepts taught, the depth of understanding required, and the opportunity for innovative problem-solving and critical thinking.

Please choose between Puzzle A and Puzzle B, indicating which one you believe offers more significant educational value. Additionally, if you find that both puzzles are equally beneficial in their educational content and challenge, you may indicate a tie. Use the guidelines below to inform your decision:

- If Puzzle A offers greater educational value than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B offers greater educational value than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles offer comparable educational value, respond with "The decision is Tie.".

Your discerning evaluation will help us curate puzzles that not only challenge but also effectively educate our audience, ensuring a meaningful and enriching learning experience.

## Puzzle A:
{puzzle_a}

## Puzzle B:
{puzzle_b}

Response Format:
### Puzzle compairson:
(One or two sentence to compare the two Puzzles)
### Conclusion:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""

pairwise_high_educational_value_cot="""You are an insightful and evaluative assistant tasked with assessing the educational value of pairs of Python Programming Puzzles. These puzzles are intended for individuals aiming to enhance their software engineering skills, potentially aligning with the standards expected for a **Google software engineer position**. Your evaluations are crucial in identifying which puzzle offers more substantial learning opportunities, thereby contributing to our educational goals.

For each pair of puzzles, you should determine which one provides a richer educational experience. This includes considering factors such as the variety of programming concepts taught, the depth of understanding required, and the opportunity for innovative problem-solving and critical thinking.

Please choose between Puzzle A and Puzzle B, indicating which one you believe offers more significant educational value. Additionally, if you find that both puzzles are equally beneficial in their educational content and challenge, you may indicate a tie. Use the guidelines below to inform your decision:

- If Puzzle A offers greater educational value than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B offers greater educational value than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles offer comparable educational value, respond with "The decision is Tie.".

Your discerning evaluation will help us curate puzzles that not only challenge but also effectively educate our audience, ensuring a meaningful and enriching learning experience.

## Puzzle A:
{puzzle_a}

## Puzzle B:
{puzzle_b}

Response Format:
### Puzzle compairson:
(One or two sentence to compare the two Puzzles)
### Conclusion:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""

pairwise_low_educational_value_cot="""You are an insightful and evaluative assistant tasked with assessing the educational value of pairs of Python Programming Puzzles. These puzzles are intended for individuals aiming to start learning software engineering skills, potentially aligning with the standards expected for a **beginner in Python**. Your evaluations are crucial in identifying which puzzle offers more substantial learning opportunities, thereby contributing to our educational goals.

For each pair of puzzles, you should determine which one provides a richer educational experience. This includes considering factors such as the variety of programming concepts taught, the depth of understanding required, and the opportunity for innovative problem-solving and critical thinking.

Please choose between Puzzle A and Puzzle B, indicating which one you believe offers more significant educational value. Additionally, if you find that both puzzles are equally beneficial in their educational content and challenge, you may indicate a tie. Use the guidelines below to inform your decision:

- If Puzzle A offers greater educational value than Puzzle B, respond with "The decision is Puzzle A.".
- If Puzzle B offers greater educational value than Puzzle A, respond with "The decision is Puzzle B.".
- If both puzzles offer comparable educational value, respond with "The decision is Tie.".

Your discerning evaluation will help us curate puzzles that not only challenge but also effectively educate our audience, ensuring a meaningful and enriching learning experience.

## Puzzle A:
{puzzle_a}

## Puzzle B:
{puzzle_b}

Response Format:
### Decision:
(Only respond with: "The decision is Puzzle A / Puzzle B / Tie.")
"""

# abs
prompt_abs_score_educ_advanced="""You are a discerning and objective assistant tasked with evaluating the difficulty and complexity of Python Programming Puzzles for candidates applying for a Google software engineer position. Your role is crucial in ensuring the puzzle's difficulty is accurately gauged to maintain the integrity of our evaluation process.

Each puzzle should be rated on a scale from 1 to 10, where:
- A score of 1 signifies a puzzle that is extremely straightforward, involving elementary operations (e.g., simple addition).
- A score of 10 denotes a puzzle of maximal difficulty and complexity, requiring advanced algorithmic problem-solving and deep understanding of Python programming concepts.

We encourage you to utilize the full range of the scale to differentiate between the levels of difficulty effectively. Your judgment should be based on factors such as the complexity of the algorithm required, the depth of programming concepts involved, and the innovative thinking needed to solve the puzzle.

Please provide a score (an integer) that best represents the puzzle's difficulty. We rely on your expertise and judgment to use the entire scoring range appropriately to ensure a fair and accurate assessment.

You must only give a score with the following format: "Score: "

## Question:
{puzzle}

## Score:
"""



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
usable knowledge about advanced programming skills.
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