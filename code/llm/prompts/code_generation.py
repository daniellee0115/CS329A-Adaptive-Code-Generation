"""
Adapted from LiveCodeBench
"""

import json
from evaluation.code_generation import CodeGenerationProblem

class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."

    SYSTEM_MESSAGE_GEMINITHINK = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def get_generic_question_template(sample):

    question = None
    starter_code = None
    if isinstance(sample, CodeGenerationProblem):
        question = sample.question_content
        starter_code = sample.starter_code
    else:
        question = sample["question"] 
        starter_code = sample["starter_code"]
    
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (Be sure to wrap the generated code within ```python and ```)\n\n"
    return prompt
