import argparse
import os
import sys
import json
import numpy as np
from src.parser import load_and_validate_inputs
from src.vocab import load_and_clean_vocab
from src.engine import JsonStateEngine

try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    print("Error: llm_sdk package not found in paths.", file=sys.stderr)
    sys.exit(1)

def generate_constrained_json(model: Small_LLM_Model, engine: JsonStateEngine, vocab_map: dict) -> str:
    system_context = (
        f"You are a precise function calling agent.\n"
        f"Based on this user request: '{engine.prompt_text}'\n"
        f"Select the correct function and parameters from the schema, outputting valid JSON matching the format exactly.\n"
        f"JSON Output:\n"
    )
    generated_text = engine.prompt_prefix
    full_context = system_context + generated_text
    max_tokens = 250
    
    for _ in range(max_tokens):
        # 1. Run inference iteration through current text strings
        encoded_tokens = model.encode(full_context)
    
        if hasattr(encoded_tokens, "tolist"):
            input_ids = encoded_tokens.tolist()
        else:
            input_ids = list(encoded_tokens)
            
        # Ensure it's flat if the SDK encoder returns a batched 2D tensor (e.g., [[id1, id2]])
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        logits = model.get_logits_from_input_ids(input_ids)
        logits_np = np.array(logits)

        # 2. Get active character rules from lookahead state machine
        allowed_strings = engine.get_allowed_strings(generated_text)
        if not allowed_strings:
            break

        # 3. Construct and apply the negative infinity constraint mask
        mask = np.full_like(logits_np, -float('inf'))
        
        for token_id, token_str in vocab_map.items():
            is_legal = False
            for allowed in allowed_strings:
                if allowed.startswith(token_str):
                    is_legal = True
                    break
            
            if is_legal:
                mask[token_id] = logits_np[token_id]

        # 4. Greedy Selection step
        selected_token_id = int(np.argmax(mask))
        
        if mask[selected_token_id] == -float('inf'):
            print("Warning: Logit masking trapped. Forcing loop break.", file=sys.stderr)
            break

        # Commit token text selection to the running string history context
        token_text = vocab_map[selected_token_id]
        generated_text += token_text
        full_context += token_text

        if generated_text.endswith("}}"):
            break

    return generated_text

def main() -> None:
    parser = argparse.ArgumentParser(description="42 Constrained Decoding Function Caller")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    config = load_and_validate_inputs(args.functions_definition, args.input)
    
    model = Small_LLM_Model()
    vocab_map = load_and_clean_vocab(model)
    
    output_results = []

    print("\n--- Running Constrained Token Pipeline Execution ---")
    for idx, prompt_obj in enumerate(config.prompts):
        print(f"Running Prompt [{idx + 1}/{len(config.prompts)}]: '{prompt_obj.prompt}'")
        
        engine = JsonStateEngine(prompt_obj, config.functions)
        raw_json_output = generate_constrained_json(model, engine, vocab_map)
        
        try:
            parsed_object = json.loads(raw_json_output)
            output_results.append(parsed_object)
        except json.JSONDecodeError:
            print(f"Error parsing compiled string: '{raw_json_output}'", file=sys.stderr)
            output_results.append({
                "prompt": prompt_obj.prompt,
                "name": "error_extraction_failed",
                "parameters": {}
            })

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_results, out_f, indent=2)
        
    print(f"\nCompleted! Execution outputs compiled safely into: '{output_path}'")

if __name__ == "__main__":
    main()