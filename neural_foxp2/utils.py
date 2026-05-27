"""
Neural FOXP2 Utilities Module

Contains helper functions used across all pipeline stages.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List


@torch.no_grad()
def get_residuals(model, tokenizer, prompt: str, layer_idx: int) -> np.ndarray:
    """
    Extract residual stream activations at a specific layer for the last token.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        layer_idx: Layer index to extract from
        
    Returns:
        Numpy array of shape [d_model] — last token activations
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Extract last token hidden state at specified layer
    h = outputs.hidden_states[layer_idx][0, -1]
    return h.float().cpu().numpy()


@torch.no_grad()
def early_language_mass(
    logits: torch.Tensor,
    V_tgt: List[int],
    V_en: List[int]
) -> torch.Tensor:
    """
    Compute language mass differential from logits.
    
    ΔM = Σ p(tgt_tokens) − Σ p(en_tokens)
    
    Handles both single-sample and batched inputs:
      - logits shape [V] or [1, V] → returns scalar tensor
      - logits shape [B, V] → returns tensor of shape [B]
    
    Args:
        logits: Model logits tensor, shape [..., vocab_size]
        V_tgt: Target-language token IDs
        V_en: English token IDs
        
    Returns:
        Mass differential tensor (positive = target-language bias)
    """
    probs = F.softmax(logits, dim=-1)
    m_tgt = probs[..., V_tgt].sum(dim=-1)
    m_en = probs[..., V_en].sum(dim=-1)
    return m_tgt - m_en


def compute_selectivity(
    z_en: np.ndarray,
    z_tgt: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute standardized selectivity scores for SAE features.
    
    Sel_j = (mean_tgt - mean_en) / (std_tgt + std_en + eps)
    
    Args:
        z_en: English activations [N, n_features]
        z_tgt: Target-language activations [N, n_features]
        eps: Small constant for numerical stability
        
    Returns:
        Selectivity scores [n_features]
    """
    mean_diff = z_tgt.mean(axis=0) - z_en.mean(axis=0)
    std_sum = z_tgt.std(axis=0) + z_en.std(axis=0) + eps
    return mean_diff / std_sum


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector."""
    return v / (np.linalg.norm(v) + eps)


def normalize_activations(acts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize activation vectors along last dimension."""
    return acts / (acts.norm(dim=-1, keepdim=True) + eps)
