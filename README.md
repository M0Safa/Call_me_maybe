# Call_me_maybe

This project has been created as part of the 42 curriculum by [mosafa]https://profile-v3.intra.42.fr/users/mosafa


## Description

This project focuses on the implementation of Constrained Token Generation, an advanced Natural Language Processing (NLP) technique that forces large language models to 
output data matching precise structural schemas or structural layouts. Traditional generative models often hallucinate or fail to output valid formats like JSON because 
their vocabulary options are too broad. By intervening at the raw logit level, a constrained generation engine applies a real-time token mask to the model's vocabulary 
map on every token step.

## Instructions

to install all packages run:

```Bash
make
#or
make install
```

to run the script:

```Bash
make run
```
to remove all cashe and output:

```Bash
make clean
```
to debug:

```Bash
make debug
```
to check flake8 and mypy:

```Bash
make lint
```

the path of functions,prompt and output json is set by default you can changed by adding it to the make file with the right flag


## Resource

llm understanding: https://youtu.be/wjZofJX0v4M?si=4v6h2QSwZe6sfRGX


**I used AI to understand how to use the llm_sdk and numpy.**

## Algorithm Explanation

	My constrained decoding approach applies a Two-Step Real-Time Token Mask directly at the model's raw logit layer. Step 1 scores the entire vocabulary space to 
	cleanly classify and lock in the correct target function signature. Step 2 utilizes an Autoregressive Prefix-Matching Pointer to isolate the start index of each 
	argument, moving an active index search window forward to safely slice text parameters directly from the prompt array without generating text from scratch.

## Design Decisions

	To maximize speed, we swapped slow character-by-character generation for dynamic index-pointer tracking. We structured our type constraints using strict vocabulary 
	masks that instantly filter out illegal tokens based on the active argument type (e.g., blocking letters for numbers, or restricting selections to boolean variants).

## Performance Analysis
	The engine achieves 100% accuracy and deterministic reliability during Step 1 function selection, completely eliminating syntax hallucinations. However, during Step 2 parameter extraction, performance dropped to approximately 70% accuracy due to small-model context sensitivities.

## Challenges Faced

	Our primary hurdle was a token-reuse bug where multi-string functions continuously extracted the exact same initial keyword for every parameter. We resolved this by implementing a progressive search window that advances forward right after a segment is claimed.

## Testing Strategy

	We validated functional runtime correctness by executing the system against a batch suite of eleven specific test prompts provided in the project specification. These prompts covered diverse operational edge cases, including simple arithmetic addition, multi-string regex substitutions, and literal string modifications.

##  Example usage

	a set of functions and prompts json are given in the data directory to test the script.