def return_prompt_format(model_id, text):
    """
    return the prompt with the correct format for the given model_id

    list model_compatible:['deepseek-coder', 'openchat']
    """

    if "deepseek-coder" in model_id:
        prompt_model="""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
### Instruction:
{instruction}
### Response:
"""
    if "openchat" in model_id:
        prompt_model = """GPT4 Correct User: {instruction}<|end_of_turn|>GPT4 Correct Assistant:"""
    
    else: 
        raise ValueError(f"Model {model_id} not supported")
    
    return prompt_model.format(instruction=text)