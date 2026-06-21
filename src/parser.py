from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import json
import os
import sys


class ParameterSchema(BaseModel):
    type: Literal["string", "number", "boolean", "int"]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: Dict[str, str]


class TestPrompt(BaseModel):
    prompt: str


class ProjectConfig:
    def __init__(self, functions: List[FunctionDefinition], prompts: List[TestPrompt]):
        self.functions = functions
        self.prompts = prompts

    def get_function_by_name(self, name: str) -> Optional[FunctionDefinition]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None
    
def load_and_validate_inputs(functions_path: str, input_path: str) -> ProjectConfig:
    """
    Loads input JSON files and validates them via Pydantic.
    Handles missing files and invalid formatting gracefully.
    """
    # 1. Check if files exist explicitly
    if not os.path.exists(functions_path):
        print(f"Error: Functions definition file not found at '{functions_path}'", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(input_path):
        print(f"Error: Input test prompts file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Parse functions definition file
    try:
        with open(functions_path, "r", encoding="utf-8") as f:
            raw_functions = json.load(f)
            # Validate every item in the array with Pydantic
            functions = [FunctionDefinition(**fn) for fn in raw_functions]
    except json.JSONDecodeError as e:
        print(f"Error: '{functions_path}' is not a valid JSON file. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error validating function schemas: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Parse test prompts file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
            prompts = [TestPrompt(**p) for p in raw_prompts]
    except json.JSONDecodeError as e:
        print(f"Error: '{input_path}' is not a valid JSON file. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error validating prompt entries: {e}", file=sys.stderr)
        sys.exit(1)

    return ProjectConfig(functions=functions, prompts=prompts)