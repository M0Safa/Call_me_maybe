from typing import Dict
import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def load_and_clean_vocab(model: Small_LLM_Model) -> Dict[int, str]:
    """
    clean the vocab and return them

    Args:
        Small_LLM_Model: the model object
    Returns:
        Dict[int, str]: the cleaned vocab
    """
    vocab_path = model.get_path_to_vocab_file()

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
    except Exception:
        raise RuntimeError(f"Failed to read vocabulary file at {vocab_path}")

    id_to_token: Dict[int, str] = {}

    for token_str, token_id in raw_vocab.items():

        cleaned_str = token_str.replace("Ġ", " ").replace("˙G", " ")

        id_to_token[int(token_id)] = cleaned_str

    return id_to_token
