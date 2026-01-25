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
    """
    Load English-Hindi parallel prompts from the IITB dataset.
    
    Args:
        n_prompts: Number of prompts to load
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use
        seed: Random seed for shuffling
        
    Returns:
        List of dicts with 'en' and 'hi' keys
    """
    n_prompts = n_prompts or config.n_prompts
    dataset_name = dataset_name or config.dataset_name
    split = split or config.dataset_split
    seed = seed or config.seed
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    matched_prompts = []
    for ex in dataset.shuffle(seed=seed):
        pair = clean_pair(ex)
        if pair:
            matched_prompts.append(pair)
        if len(matched_prompts) >= n_prompts:
            break
    
    print(f"Loaded {len(matched_prompts)} matched prompts")
    return matched_prompts


def clean_pair(ex: dict) -> Dict[str, str]:
    """
    Clean a translation pair from the dataset.
    
    Args:
        ex: Raw example from dataset
        
    Returns:
        Dict with 'en' and 'hi' keys, or None if invalid
    """
    try:
        en = ex["translation"]["en"].strip()
        hi = ex["translation"]["hi"].strip()
        if en and hi:
            return {"en": en, "hi": hi}
    except (KeyError, AttributeError):
        pass
    return None


def build_language_token_sets(tokenizer) -> Tuple[List[int], List[int]]:
    """
    Build token sets for Hindi and English.
    
    Uses Unicode ranges to identify language-specific tokens:
    - Hindi: Devanagari block U+0900 to U+097F
    - English: Latin alphabet a-z, A-Z
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        (V_hi, V_en) tuple of token ID lists
    """
    V_hi: Set[int] = set()
    V_en: Set[int] = set()

    for tok_id in range(tokenizer.vocab_size):
        tok = tokenizer.decode([tok_id])

        # Devanagari Unicode block
        if any("\u0900" <= ch <= "\u097F" for ch in tok):
            V_hi.add(tok_id)
        # Latin alphabet heuristic
        elif any("a" <= ch.lower() <= "z" for ch in tok):
            V_en.add(tok_id)

    print(f"Token sets: V_hi={len(V_hi)}, V_en={len(V_en)}")
    return list(V_hi), list(V_en)
