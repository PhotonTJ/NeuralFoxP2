"""
Neural FOXP2 Data Module

Contains data loading and preprocessing utilities.
"""
from typing import List, Dict, Tuple, Set
from datasets import load_dataset

from config import config


def load_matched_prompts(
    n_prompts: int = None,
    dataset_name: str = None,
    split: str = None,
    seed: int = None
) -> List[Dict[str, str]]:
   
    n_prompts = n_prompts or config.n_prompts
    dataset_name = dataset_name or config.dataset_name
    split = split or config.dataset_split
    seed = seed or config.seed
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    matched_prompts = []
    for ex in dataset.shuffle(seed=seed).select(range(n_prompts * 3)):
        pair = clean_pair(ex)
        if pair:
            matched_prompts.append(pair)
        if len(matched_prompts) >= n_prompts:
            break
    
    print(f"Loaded {len(matched_prompts)} matched prompts")
    return matched_prompts


def clean_pair(ex: dict) -> Dict[str, str]:
    try:
        en = ex["translation"]["en"].strip()
        hi = ex["translation"]["hi"].strip()
        if en and hi:
            return {"en": en, "hi": hi}
    except (KeyError, AttributeError):
        pass
    return None


def build_language_token_sets(tokenizer) -> Tuple[List[int], List[int]]:
   
    V_hi: Set[int] = set()
    V_en: Set[int] = set()

    for tok_id in range(tokenizer.vocab_size):
        tok = tokenizer.convert_ids_to_tokens(tok_id)

        # Devanagari Unicode block
        if any("\u0900" <= ch <= "\u097F" for ch in tok):
            V_hi.add(tok_id)
        # Latin alphabet heuristic
        elif all(ch.isascii() for ch in tok) and any(ch.isalpha() for ch in tok):
            V_en.add(tok_id)

    print(f"Token sets: V_hi={len(V_hi)}, V_en={len(V_en)}")
    return list(V_hi), list(V_en)

