"""
Neural FOXP2 Languages Module

Language configuration registry with Unicode script detection,
dataset mappings, and token classification helpers.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable


@dataclass
class LanguageConfig:
    """Configuration for a target language."""
    code: str                  # ISO 639-1 code: hi, es, zh, bn, te
    name: str                  # Display name
    unicode_ranges: List[Tuple[int, int]]   # List of (start_codepoint, end_codepoint) for script detection
    default_dataset: str       # HuggingFace dataset name
    dataset_src_field: str     # Source (English) field path in dataset
    dataset_tgt_field: str     # Target language field path in dataset
    dataset_split: str = "test"
    dataset_config: Optional[str] = None  # subset/config name if needed

    def is_target_char(self, ch: str) -> bool:
        """Check if a character belongs to this language's script."""
        cp = ord(ch)
        return any(start <= cp <= end for start, end in self.unicode_ranges)

    def is_target_token(self, token: str) -> bool:
        """Check if a token contains characters from this language's script."""
        return any(self.is_target_char(ch) for ch in token)


# Language Registry
LANGUAGE_REGISTRY: Dict[str, LanguageConfig] = {
    "hindi": LanguageConfig(
        code="hi",
        name="Hindi",
        unicode_ranges=[(0x0900, 0x097F)],  # Devanagari
        default_dataset="cfilt/iitb-english-hindi",
        dataset_src_field="translation.en",
        dataset_tgt_field="translation.hi",
    ),
    "spanish": LanguageConfig(
        code="es",
        name="Spanish",
        unicode_ranges=[(0x00C0, 0x00FF), (0x0100, 0x024F)],  # Latin Extended A+B with diacritics
        default_dataset="Helsinki-NLP/opus-100",
        dataset_src_field="translation.en",
        dataset_tgt_field="translation.es",
        dataset_config="en-es",
    ),
    "chinese": LanguageConfig(
        code="zh",
        name="Chinese",
        unicode_ranges=[(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F)],  # CJK Unified + Ext A + Punctuation
        default_dataset="Helsinki-NLP/opus-100",
        dataset_src_field="translation.en",
        dataset_tgt_field="translation.zh",
        dataset_config="en-zh",
    ),
    "bengali": LanguageConfig(
        code="bn",
        name="Bengali",
        unicode_ranges=[(0x0980, 0x09FF)],  # Bengali script
        default_dataset="Helsinki-NLP/opus-100",
        dataset_src_field="translation.en",
        dataset_tgt_field="translation.bn",
        dataset_config="en-bn",
    ),
    "telugu": LanguageConfig(
        code="te",
        name="Telugu",
        unicode_ranges=[(0x0C00, 0x0C7F)],  # Telugu script
        default_dataset="Helsinki-NLP/opus-100",
        dataset_src_field="translation.en",
        dataset_tgt_field="translation.te",
        dataset_config="en-te",
    ),
}


def get_language(name: str) -> LanguageConfig:
    """Look up a language by name. Raises KeyError if not found."""
    name_lower = name.lower()
    if name_lower not in LANGUAGE_REGISTRY:
        available = ", ".join(sorted(LANGUAGE_REGISTRY.keys()))
        raise KeyError(f"Unknown language '{name}'. Available: {available}")
    return LANGUAGE_REGISTRY[name_lower]
