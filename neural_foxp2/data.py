"""
Neural FOXP2 Data Module

Language-agnostic data loading and token set construction.
"""
from typing import List, Dict, Tuple, Set, Optional
from datasets import load_dataset
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from .languages import LanguageConfig


def _nested_get(d: dict, dotted_key: str):
    """Access nested dict value using dot notation: 'translation.en' -> d['translation']['en']"""
    keys = dotted_key.split('.')
    val = d
    for k in keys:
        val = val[k]
    return val


def load_matched_prompts(
    language: LanguageConfig,
    n_prompts: int = None,
    dataset_name: str = None,
    split: str = None,
    seed: int = None
) -> List[Dict[str, str]]:
    """
    Load matched English-TargetLanguage prompt pairs.
    
    Args:
        language: Target language configuration
        n_prompts: Number of pairs to load
        dataset_name: Override dataset name
        split: Dataset split
        seed: Random seed
        
    Returns:
        List of dicts with keys 'en' and 'tgt'
    """
    n_prompts = n_prompts or config.n_prompts
    dataset_name = dataset_name or language.default_dataset
    split = split or language.dataset_split
    seed = seed or config.seed
    
    print(f"Loading dataset: {dataset_name} for {language.name}")
    
    load_kwargs = {"split": split}
    if language.dataset_config:
        load_kwargs["name"] = language.dataset_config
    
    dataset = load_dataset(dataset_name, **load_kwargs)
    
    matched_prompts = []
    select_count = min(n_prompts * 3, len(dataset))
    for ex in dataset.shuffle(seed=seed).select(range(select_count)):
        pair = _clean_pair(ex, language)
        if pair:
            matched_prompts.append(pair)
        if len(matched_prompts) >= n_prompts:
            break
    
    print(f"Loaded {len(matched_prompts)} matched prompts for {language.name}")
    return matched_prompts


def _clean_pair(ex: dict, language: LanguageConfig) -> Optional[Dict[str, str]]:
    """Extract and validate an (English, target) pair from a dataset example."""
    try:
        en = _nested_get(ex, language.dataset_src_field).strip()
        tgt = _nested_get(ex, language.dataset_tgt_field).strip()
        if en and tgt:
            return {"en": en, "tgt": tgt}
    except (KeyError, AttributeError, TypeError):
        pass
    return None


def build_language_token_sets(
    tokenizer,
    language: LanguageConfig
) -> Tuple[List[int], List[int]]:
    """
    Partition the tokenizer vocabulary into target-language and English token sets.
    
    A token is classified as target-language if it contains any character
    within the language's Unicode ranges. A token is classified as English
    if all characters are ASCII and at least one is alphabetic.
    
    Args:
        tokenizer: HuggingFace tokenizer
        language: Target language configuration
        
    Returns:
        (V_tgt, V_en) — lists of token IDs
    """
    V_tgt: Set[int] = set()
    V_en: Set[int] = set()

    for tok_id in range(tokenizer.vocab_size):
        tok = tokenizer.convert_ids_to_tokens(tok_id)
        if tok is None:
            continue

        # Check if token contains target-language script characters
        if language.is_target_token(tok):
            V_tgt.add(tok_id)
        # Latin/ASCII alphabet heuristic for English
        elif all(ch.isascii() for ch in tok) and any(ch.isalpha() for ch in tok):
            V_en.add(tok_id)

    print(f"Token sets for {language.name}: V_tgt={len(V_tgt)}, V_en={len(V_en)}")
    return list(V_tgt), list(V_en)
