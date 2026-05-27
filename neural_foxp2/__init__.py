"""
Neural FOXP2 — Language-Specific Steering for Targeted Language Improvement in LLMs

A unified pipeline for identifying and steering language-selective features
in transformer models using Sparse Autoencoders.
"""
__version__ = "0.2.0"

from .languages import LANGUAGE_REGISTRY, LanguageConfig, get_language
from .models import SparseAutoencoder, load_model_and_tokenizer, get_model_layers
from .data import load_matched_prompts, build_language_token_sets
from .stage1 import run_stage1
from .stage2 import run_stage2
from .stage3 import run_stage3
