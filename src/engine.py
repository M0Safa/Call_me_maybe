import re
import numpy as np
from typing import List
from src.parser import FunctionDefinition, TestPrompt
from llm_sdk import Small_LLM_Model

class JsonStateEngine:
    def __init__(self, prompt_obj: TestPrompt, available_functions: List[FunctionDefinition], vocab_map: dict, model: Small_LLM_Model):
        self.prompt = prompt_obj.prompt
        self.functions = available_functions
        self.prompt_prefix = f'{{"prompt":"{self.prompt}","name":"'
        self.choosed_fun = None
        self.model = model
        self.vocab = vocab_map

    def choose_function(self) -> str:
    
        digit_tokens = {}
        for score_val in range(10):  # Fix: 0 to 10 inclusive needs range(11)
            for tid, tstr in self.vocab.items():
                try:
                    if int(tstr) == score_val:
                        digit_tokens[score_val] = tid
                        break
                except ValueError:
                    continue

        best_score = -float('inf')
        chosen_function_name = self.functions[0].name

        # 2. Loop over every function definition and evaluate it
        for fn in self.functions:
            # Build the descriptive context prompt for this specific candidate
            evaluation_prompt = (
        "Task: Rate the utility of the following function for completing the user request.\n"
        "Grading Scale:\n"
        "- 9: Perfect match (The function specifically performs this exact task).\n"
        "- 4: Partial match (The function handles a component of the task but lacks specifics).\n"
        "- 0: Completely irrelevant (The function does something entirely unrelated).\n\n"
        f"User Request: \"{self.prompt}\"\n"
        f"Function Description: \"{fn.description}\"\n\n"
        "Rate utility with exactly one digit from 0 to 10.\n"
        "Utility Score: "
)

            # Encode and extract logits for the final token position
            input_ids = self.model.encode(evaluation_prompt).tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            logits = np.array(self.model.get_logits_from_input_ids(input_ids))

            # Maximize stability by grabbing the highest logit value among target variations
            target_logits = {}
            for digit_val, token_id in digit_tokens.items():
                target_logits[digit_val] = logits[token_id]

            # Calculate the relative confidence score
            score: float = 0
            for i in range(10):
                if i > 5:
                    score += i *target_logits[i]
                else:
                    score -= i *target_logits[i]

            print(f"  -> Candidate '{fn.name}' alignment confidence score: {score:.4f}")

            # Track the candidate with the highest probability
            if score > best_score:
                best_score = score
                self.choosed_fun = fn

        print(f"Winner Selected via Scoring: '{self.choosed_fun.name}'")
        return self.choosed_fun.name

    def choose_parameters (self, Gen_text: str):
        for par, type in self.choosed_fun.parameters.items()
        

        