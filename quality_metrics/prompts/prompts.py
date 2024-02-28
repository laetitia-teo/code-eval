pp_prompt_user_1ex = """You will be given a programming puzzle, and your role is to solve it. The puzzle will be implemented as a function `f` and you should write a function `g` that satisfies `f(g()) is True`.
Here is an example first:
Puzzle:
```python
{puzzle}
```

Solution:
```python
{solution}
```

Now let's try to solve a new one.
Puzzle:
```python
{archive_puzzle}
```"""

pp_prompt_assistant = """Solution:
```python
{archive_solution}
```"""

# TODO prompts for in-context learning
