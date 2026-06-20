import re
from typing import List
from src.parser import FunctionDefinition, TestPrompt

class JsonStateEngine:
    def __init__(self, prompt_obj: TestPrompt, available_functions: List[FunctionDefinition]):
        self.prompt_text = prompt_obj.prompt
        self.functions = available_functions
        self.prompt_prefix = f'{{"prompt":"{self.prompt_text}","name":"'
        # Define the exact bridge string needed to transition between a function name and parameters
        self.param_bridge = '","parameters":{'

    def get_allowed_strings(self, current_generated_text: str) -> List[str]:
        """
        Analyzes what has been written so far, and returns a list of acceptable
        string fragments or characters that can be appended next.
        """
        # State 1: Enforce Root Object Opening
        if not current_generated_text:
            return ["{"]

        after_prefix = current_generated_text[len(self.prompt_prefix):]
        
        # Check if we are currently traversing across the bridge sequence for any matched function
        for fn in self.functions:
            # If we typed the function name and are now typing the bridge: '","parameters":{'
            if after_prefix.startswith(fn.name):
                bridge_progress = after_prefix[len(fn.name):]
                
                if bridge_progress == self.param_bridge:
                    # The bridge is 100% complete! Pass control to State 4 parameter value parsing
                    break
                    
                if self.param_bridge.startswith(bridge_progress):
                    # Still mid-bridge. Return the exact next character required to advance
                    next_bridge_char_idx = len(bridge_progress)
                    return [self.param_bridge[next_bridge_char_idx]]

        # If we are still typing the core function name itself
        matching_fns = [fn.name for fn in self.functions if fn.name.startswith(after_prefix)]
        if matching_fns:
            allowed_chars = []
            for fn_name in matching_fns:
                if after_prefix == fn_name:
                    # Exact function name complete! Start the bridge by allowing its first character
                    return [self.param_bridge[0]]
                else:
                    remainder = fn_name[len(after_prefix):]
                    allowed_chars.append(remainder[0])
            return list(set(allowed_chars))

        # State 4: Parsing Function-Specific Parameters
        for fn in self.functions:
            fn_marker = f'{self.prompt_prefix}{fn.name}{self.param_bridge}'
            if current_generated_text.startswith(fn_marker):
                param_text = current_generated_text[len(fn_marker):]
                return self._get_allowed_param_tokens(param_text, fn)

        return []

    def _get_allowed_param_tokens(self, param_text: str, fn: FunctionDefinition) -> List[str]:
        """
        Calculates character allowances inside the parameters object based on schema types.
        """
        if not fn.parameters:
            return ["}}"]

        completed_keys = re.findall(r'"([^"]+)":', param_text)
        remaining_keys = [k for k in fn.parameters.keys() if k not in completed_keys]

        # All parameters have been completely written out
        if not remaining_keys:
            if param_text.endswith(","):
                return []
            return ["}}"]

        current_target_key = remaining_keys[0]
        expected_type = fn.parameters[current_target_key].type

        # Build the exact string signature required for the active key
        expected_key_prefix = f'"{current_target_key}":'
        
        last_comma_idx = param_text.rfind(",")
        active_window = param_text[last_comma_idx + 1:] if last_comma_idx != -1 else param_text

        if not active_window:
            return ['"']
            
        if expected_key_prefix.startswith(active_window):
            return [expected_key_prefix[len(active_window)]]

        # Generating the value for the active parameter key
        if active_window.startswith(expected_key_prefix):
            value_part = active_window[len(expected_key_prefix):]
            
            has_more = len(remaining_keys) > 1
            separator = "," if has_more else ""

            # 1. Enforce String Constraints
            if expected_type == "string":
                if not value_part:
                    return ['"']
                if value_part.startswith('"') and not value_part.endswith('"') or value_part == '"':
                    return ['"', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '-', '_']
                if value_part.endswith('"') and value_part != '"':
                    return [separator] if separator else ["}"]

            # 2. Enforce Number Constraints
            elif expected_type == "number":
                if not value_part:
                    return ["-"] + [str(i) for i in range(10)]
                
                if re.match(r'^-?\d*\.?\d*$', value_part):
                    allowed = [str(i) for i in range(10)]
                    if "." not in value_part:
                        allowed.append(".")
                    allowed.append(separator if separator else "}")
                    return allowed

            # 3. Enforce Boolean Constraints
            elif expected_type == "boolean":
                if "true".startswith(value_part) or "false".startswith(value_part):
                    if value_part in ["true", "false"]:
                        return [separator] if separator else ["}"]
                    allowed = []
                    if "true".startswith(value_part):
                        allowed.append("true"[len(value_part)])
                    if "false".startswith(value_part):
                        allowed.append("false"[len(value_part)])
                    return allowed

        return []