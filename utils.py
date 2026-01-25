"""
Neural FOXP2 Utilities Module

Contains helper functions used across stages.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List


@torch.no_grad()
def get_residuals(model, tokenizer, prompt: str, layer_idx: int) -> np.ndarray:
    """
    Extract residual stream activations at a specific layer.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        layer_idx: Layer index to extract from
        
    Returns:
        Numpy array of shape [d_model] - last token activations
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
def early_language_mass(logits: torch.Tensor, V_hi: List[int], V_en: List[int]) -> float:
    """
    Compute language mass differential from logits.
    
    M = sum(p_hi) - sum(p_en)
    
    Args:
        logits: Model logits tensor
        V_hi: Hindi token IDs
        V_en: English token IDs
        
    Returns:
        Mass differential (positive = Hindi bias)
    """
    probs = F.softmax(logits, dim=-1)
    m_hi = probs[..., V_hi].sum().item()
    m_en = probs[..., V_en].sum().item()
    return m_hi - m_en


def compute_selectivity(z_en: np.ndarray, z_hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute standardized selectivity scores for SAE features.
    
    Sel_j = (mean_hi - mean_en) / (std_hi + std_en + eps)
    
    Args:
        z_en: English activations [N, n_features]
        z_hi: Hindi activations [N, n_features]
        eps: Small constant for numerical stability
        
    Returns:
        Selectivity scores [n_features]
    """
    mean_diff = z_hi.mean(axis=0) - z_en.mean(axis=0)
    std_sum = z_hi.std(axis=0) + z_en.std(axis=0) + eps
    return mean_diff / std_sum


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector."""
    return v / (np.linalg.norm(v) + eps)


def normalize_activations(acts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize activation vectors along last dimension."""
    return acts / (acts.norm(dim=-1, keepdim=True) + eps)
