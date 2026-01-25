"""
Neural FOXP2 Stage II Module

Stage II: Low-Rank Steering Directions and Window Selection
- II-1: Language-shift matrix computation
- II-2: Per-layer SVD
- II-3: Rank selection (effective rank + eigengap)
- II-4: Spectral mass and bootstrap stability
- II-5: Contiguous window selection
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from config import config
from models import SparseAutoencoder
from utils import get_residuals


def compute_layer_activations(
    model,
    tokenizer,
    prompts: List[Dict],
    layer_idx: int,
    sae: SparseAutoencoder
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get SAE feature activations for English/Hindi prompt pairs."""
    activations = []
    for pair in prompts:
        h_en = torch.tensor(get_residuals(model, tokenizer, pair["en"], layer_idx), dtype=torch.float32)
        h_en = h_en / (h_en.norm() + 1e-6)
        z_en = sae.encode(h_en).detach().numpy()
        
        h_hi = torch.tensor(get_residuals(model, tokenizer, pair["hi"], layer_idx), dtype=torch.float32)
        h_hi = h_hi / (h_hi.norm() + 1e-6)
        z_hi = sae.encode(h_hi).detach().numpy()
        
        activations.append((z_en, z_hi))
    return activations


def compute_language_shift_matrix(
    model,
    tokenizer,
    prompts: List[Dict],
    layer_idx: int,
    sae: SparseAutoencoder,
    language_neurons: List[int]
) -> np.ndarray:
    """
    Compute Δz = z(x_hi) - z(x_en), restricted to language neuron coordinates.
    
    Returns:
        ΔZ matrix of shape [N_prompts, n_features]
    """
    activations = compute_layer_activations(model, tokenizer, prompts, layer_idx, sae)
    
    delta_z_list = []
    for z_en, z_hi in activations:
        delta_z = z_hi - z_en
        delta_z_restricted = np.zeros_like(delta_z)
        for j in language_neurons:
            if j < len(delta_z):
                delta_z_restricted[j] = delta_z[j]
        delta_z_list.append(delta_z_restricted)
    
    return np.stack(delta_z_list)


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank using entropy formula.
    
    r_eff = exp(-Σ p_i log p_i) where p_i = σ_i² / Σ σ_j²
    """
    s2 = singular_values ** 2
    total = s2.sum()
    if total < 1e-10:
        return 1
    p = s2 / total
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def find_eigengap(singular_values: np.ndarray, r_max: int) -> int:
    """Find the index with largest gap ratio σ_i / σ_{i+1}."""
    if len(singular_values) < 2:
        return 1
    gaps = singular_values[:-1] / (singular_values[1:] + 1e-10)
    gaps = gaps[:r_max]
    return int(np.argmax(gaps)) + 1


def compute_steering_directions(
    delta_Z: np.ndarray,
    r_max: int = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform SVD and select steering subspace.
    
    Returns:
        (singular_values, V_matrix, chosen_rank)
    """
    r_max = r_max or config.r_max
    
    U, S, Vt = np.linalg.svd(delta_Z, full_matrices=False)
    
    r_eff = int(np.ceil(compute_effective_rank(S)))
    r_eff = min(r_eff, r_max)
    
    r_gap = find_eigengap(S, r_max)
    
    r = min(r_eff, r_gap)
    r = max(r, 1)
    
    return S, Vt.T, r


def compute_spectral_mass(singular_values: np.ndarray, r: int) -> float:
    """Mass of top-r singular values as fraction of total."""
    s2 = singular_values ** 2
    return s2[:r].sum() / (s2.sum() + 1e-10)


def compute_stability_bootstrap(
    model,
    tokenizer,
    prompts: List[Dict],
    layer_idx: int,
    sae: SparseAutoencoder,
    language_neurons: List[int],
    n_bootstrap: int = None
) -> float:
    """
    Bootstrap stability using principal angle consistency.
    """
    n_bootstrap = n_bootstrap or config.n_bootstrap
    n = len(prompts)
    projectors = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        resampled = [prompts[i] for i in indices]
        
        delta_Z = compute_language_shift_matrix(model, tokenizer, resampled, layer_idx, sae, language_neurons)
        S, V, r = compute_steering_directions(delta_Z)
        
        V_r = V[:, :r]
        P = V_r @ V_r.T
        projectors.append(P)
    
    consistencies = []
    for i in range(len(projectors)):
        for j in range(i+1, len(projectors)):
            trace = np.trace(projectors[i] @ projectors[j])
            r = min(projectors[i].shape[0], 10)
            consistencies.append(trace / r)
    
    return np.median(consistencies) if consistencies else 1.0


def find_best_contiguous_window(
    layer_results: Dict[int, Dict],
    min_window: int = None,
    max_window: int = None
) -> Tuple[List[int], float]:
    """
    Find contiguous layer window maximizing sum of scores.
    
    Returns:
        (best_window, best_score)
    """
    min_window = min_window or config.min_window
    max_window = max_window or config.max_window
    
    layers = sorted(layer_results.keys())
    best_window = None
    best_score = -1
    
    for start_idx in range(len(layers)):
        for end_idx in range(start_idx + min_window - 1, min(start_idx + max_window, len(layers))):
            window = layers[start_idx:end_idx+1]
            score = sum(layer_results[l]["score"] for l in window)
            
            if score > best_score:
                best_score = score
                best_window = window
    
    return best_window, best_score


def run_stage2(
    model,
    tokenizer,
    matched_prompts: List[Dict],
    sae_per_layer: Dict[int, SparseAutoencoder],
    language_neurons_per_layer: Dict[int, List[int]],
    layer_range: List[int] = None
) -> Dict[str, Any]:
    """
    Run complete Stage II pipeline.
    
    Returns:
        Dictionary with:
        - layer_results: per-layer SVD results
        - intervention_window: selected layer window
        - window_score: combined window score
    """
    layer_range = layer_range or config.layer_range
    
    print("\n" + "="*70)
    print("STAGE II: Identify Low-Rank Steering Directions")
    print("="*70)
    
    layer_results = {}
    
    for layer_idx in tqdm(layer_range, desc="Stage II: Layer Analysis"):
        try:
            sae = sae_per_layer[layer_idx]
            language_neurons = language_neurons_per_layer[layer_idx]
            
            # Compute language shift matrix
            delta_Z = compute_language_shift_matrix(
                model, tokenizer,
                matched_prompts[:config.n_prompts_svd],
                layer_idx, sae, language_neurons
            )

            # SVD + rank selection
            S, V, r = compute_steering_directions(delta_Z)

            # Spectral mass
            mass = compute_spectral_mass(S, r)

            # Bootstrap stability
            stability = compute_stability_bootstrap(
                model, tokenizer,
                matched_prompts[:config.n_prompts_stability],
                layer_idx, sae, language_neurons
            )

            layer_results[layer_idx] = {
                "singular_values": S,
                "steering_vectors": V[:, :r],
                "rank": r,
                "mass": mass,
                "stability": stability,
                "score": mass * stability
            }

            print(f"  Layer {layer_idx}: rank={r}, mass={mass:.3f}, "
                  f"stability={stability:.3f}, score={mass*stability:.3f}")

        except Exception as e:
            print(f"  Layer {layer_idx}: Error - {e}")
    
    # Select best contiguous window
    intervention_window, window_score = find_best_contiguous_window(layer_results)
    
    print(f"\nIntervention Window: {intervention_window}")
    print(f"Window Score: {window_score:.4f}")
    
    return {
        "layer_results": layer_results,
        "intervention_window": intervention_window,
        "window_score": window_score
    }
