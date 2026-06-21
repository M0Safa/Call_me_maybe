install:
	uv pip install pydantic numpy flake8 mypy

run:
	uv run python -m src  --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json

debug:
	uv run python -m pdb -m src

lint:
	flake8 src/
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache
	rm -rf data/output/*