import argparse
import os
import sys
import json
from .parser import load_and_validate_inputs
from .vocab import load_and_clean_vocab
from .engine import JsonStateEngine

try:
    from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
except ImportError:
    print("Error: llm_sdk package not found in paths.", file=sys.stderr)
    sys.exit(1)


def generate_constrained_json(engine: JsonStateEngine) -> dict:
    function = engine.choose_function()
    parameters = engine.extract_parameters()
    prompt_entry = {
            "prompt": engine.prompt,
            "name": function,
            "parameters": parameters
        }
    return prompt_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    args = parser.parse_args()

    config = load_and_validate_inputs(args.functions_definition, args.input)

    model = Small_LLM_Model()
    vocab_map = load_and_clean_vocab(model)

    output_results = []

    print("\n--- Running Constrained Token Pipeline Execution ---")
    for i, p in enumerate(config.prompts):
        print(f"Running Prompt [{i + 1}/{len(config.prompts)}]: '{p.prompt}'")
        engine = JsonStateEngine(p, config.functions, vocab_map, model)
        json_output = generate_constrained_json(engine)
        output_results.append(json_output)

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_results, out_f, indent=4, ensure_ascii=False)
    print(f"\nCompleted!: '{output_path}'")


if __name__ == "__main__":
    main()
