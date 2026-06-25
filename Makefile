FUNCTIONS = --functions_definition data/input/functions_definition.json
PROMPTS = --input data/input/function_calling_tests.json
OUTPUT = --output data/output/function_calls.json
install:
	uv sync

run:
	uv run python -m src $(FUNCTIONS) $(PROMPTS) $(OUTPUT)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache
	rm -rf data/output/*
	
lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
