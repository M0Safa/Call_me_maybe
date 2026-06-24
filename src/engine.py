import numpy as np
from typing import List, Any
from .parser import FunctionDefinition, TestPrompt
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonStateEngine:
    def __init__(self, prompt_obj: TestPrompt,
                 available_functions: List[FunctionDefinition],
                 vocab_map: dict, model: Small_LLM_Model):
        self.prompt = prompt_obj.prompt
        self.functions = available_functions
        self.choosed_fun: FunctionDefinition = self.functions[0]
        self.model = model
        self.vocab = vocab_map

    def choose_function(self) -> str:
        digits = "0123456789"
        digit_tokens = self.model.encode(digits).tolist()
        if isinstance(digit_tokens[0], list):
            digit_tokens = digit_tokens[0]

        best_score = -float('inf')

        for fn in self.functions:
            evaluation_prompt = (
                        "Task: Rate the utility of the following "
                        "function for completing the user request.\n"
                        "Grading Scale:\n"
                        "- 9: Perfect match (The function specifically"
                        "performs this exact task).\n"
                        "- 4: Partial match (The function handles a component"
                        "of the task but lacks specifics).\n"
                        "- 0: Completely irrelevant (The function does"
                        " something entirely unrelated).\n"
                        f"User Request: \"{self.prompt}\"\n"
                        f"Function Description: \"{fn.description}\"\n\n"
                        "Rate utility with exactly one digit from 0 to 10.\n"
                        "Utility Score: "
                        )

            input_ids = self.model.encode(evaluation_prompt).tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            logits = np.array(self.model.get_logits_from_input_ids(input_ids))

            target_logits = []
            for token_id in digit_tokens:
                target_logits.append(logits[token_id])

            score: float = 0
            for i in range(10):
                if i > 5:
                    score += i * target_logits[i]
                else:
                    score -= i * target_logits[i]

            if score > best_score:
                best_score = score
                self.choosed_fun = fn

        return f"{self.choosed_fun.name}"

    def extract_parameters(self) -> dict:
        extracted_params: dict[str, Any] = {}
        prompt_token_ids = self.model.encode(self.prompt).tolist()
        if isinstance(prompt_token_ids[0], list):
            prompt_token_ids = prompt_token_ids[0]

        prompt = ""
        prompt += "extract the parameter to the function from the"
        prompt += f" user request\nUser Request: {self.prompt}\n"
        prompt += f"Function {self.choosed_fun.name}:"
        prompt += f"{self.choosed_fun.description}\n"
        True_tokens = [tid for tid, tstr in self.vocab.items()
                       if tstr in ["True", "true", " True", " true"]]
        False_tokens = [tid for tid, tstr in self.vocab.items()
                        if tstr in ["False", "false", " False", " false"]]

        for param_name, param_info in self.choosed_fun.parameters.items():
            param_type = param_info.type

            prompt += f"Extract parameter '{param_name}' ({param_type}): "

            input_ids = self.model.encode(prompt).tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            logits = np.array(self.model.get_logits_from_input_ids(input_ids))
            if param_type == "boolean":
                max_true = max([logits[tid] for tid in True_tokens])
                max_false = max([logits[tid] for tid in False_tokens])
                if max_true > max_false:
                    extracted_params[param_name] = True
                else:
                    extracted_params[param_name] = False
                prompt += f"{extracted_params[param_name]}\n"
                continue

            mask = np.full_like(logits, -float('inf'))

            for tid in prompt_token_ids:
                tstr = self.vocab[tid]

                if param_type == "number":
                    if not all(c in "0123456789.-" for c in tstr.strip()):
                        continue

                mask[tid] = logits[tid]

            first_token_id = int(np.argmax(mask))

            if mask[first_token_id] == -float('inf'):
                if param_type == "number":
                    extracted_params[param_name] = 0
                else:
                    extracted_params[param_name] = ""
                continue

            start_index = prompt_token_ids.index(first_token_id)

            value_token_ids = []
            is_string: bool = False
            for i in range(start_index, len(prompt_token_ids)):
                current_tid = prompt_token_ids[i]
                current_str = self.vocab[current_tid]
                if "'" in current_str or '"' in current_str:
                    is_string = not is_string

                if i > start_index and (current_str.isspace() or
                                        current_str.startswith(" ")):
                    if not is_string:
                        break

                if param_type == "number":
                    if not all(c in "0123456789.-" for
                               c in current_str.strip()):
                        break

                value_token_ids.append(current_tid)

            extracted_value_str = "".join([self.vocab[tid] for
                                          tid in value_token_ids]).strip()

            if len(extracted_value_str) > 1:
                if (extracted_value_str.startswith("'") and
                    extracted_value_str.endswith("'")) or \
                    (extracted_value_str.startswith('"') and
                     extracted_value_str.endswith('"')):
                    extracted_value_str = extracted_value_str[1:-1]
            try:
                if param_type == "number":
                    extracted_params[param_name] = float(extracted_value_str)
                elif param_type == "int":
                    extracted_params[param_name] = int(extracted_value_str)
                else:
                    extracted_params[param_name] = extracted_value_str
            except ValueError:
                extracted_params[param_name] = 0 \
                    if param_type == "number" else ""
            prompt += f"{extracted_params[param_name]}\n"

        return extracted_params
