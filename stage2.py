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

    device = next(model.parameters()).device
    activations = []
    for pair in prompts:
        h_en = torch.tensor(
            get_residuals(model, tokenizer, pair["en"], layer_idx),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        
        h_en = h_en / (h_en.norm() + 1e-6)
        z_en = sae.encode(h_en)[0].detach().cpu().numpy()
        
        h_hi = torch.tensor(
            get_residuals(model, tokenizer, pair["hi"], layer_idx),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        
        h_hi = h_hi / (h_hi.norm() + 1e-6)
        z_hi = sae.encode(h_hi)[0].detach().cpu().numpy()
        
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
   
    activations = compute_layer_activations(model, tokenizer, prompts, layer_idx, sae)

    idx = np.asarray(language_neurons, dtype=int)
    delta_z_list = []
    for z_en, z_hi in activations:
        delta_z = z_hi - z_en
        restricted = np.zeros_like(delta_z)
        restricted[idx]= delta_z[idx]
        delta_z_list.append(restricted)
    
    return np.stack(delta_z_list)


def compute_effective_rank(singular_values: np.ndarray) -> float:
    
    s2 = singular_values ** 2
    total = s2.sum()
    if total < 1e-10:
        return 1
    p = s2 / total
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def find_eigengap(singular_values: np.ndarray, r_max: int) -> int:
    
    if len(singular_values) < 2:
        return 1
    gaps = singular_values[:-1] / (singular_values[1:] + 1e-10)
    gaps = gaps[:r_max]
    return int(np.argmax(gaps)) + 1


def compute_steering_directions(
    delta_Z: np.ndarray,
    r_max: int = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    
    r_max = r_max or config.r_max
    
    U, S, Vt = np.linalg.svd(delta_Z, full_matrices=False)
    
    r_eff = int(np.ceil(compute_effective_rank(S)))
    r_eff = min(r_eff, r_max)
    
    r_gap = find_eigengap(S, r_max)
    
    r = min(r_eff, r_gap)
    r = max(r, 1)
    
    return S, Vt.T, r


def compute_spectral_mass(singular_values: np.ndarray, r: int) -> float:
    
    s2 = singular_values ** 2
    return s2[:r].sum() / (s2.sum() + 1e-10)


def compute_stability_bootstrap(
    delta_Z: np.ndarray,
    n_bootstrap: int = None
) -> float:
    
    n_bootstrap = n_bootstrap or config.n_bootstrap
    n = delta_Z.shape[0]
    projectors = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        sample = delta_Z[idx]
        
        S, V, r = compute_steering_directions(sample)
        
        V_r = V[:, :r]
        P = V_r @ V_r.T
        projectors.append(P)
    
    consistencies = []
    for i in range(len(projectors)):
        for j in range(i+1, len(projectors)):
            trace = np.trace(projectors[i] @ projectors[j])
            consistencies.append(trace / np.trace(projectors[i]))
    
    return np.median(consistencies) if consistencies else 1.0


def find_best_contiguous_window(
    layer_results: Dict[int, Dict],
    min_window: int = None,
    max_window: int = None
) -> Tuple[List[int], float]:
    
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
            delta_Z_stability = compute_language_shift_matrix(
                model, tokenizer,
                matched_prompts[:config.n_prompts_stability],
                layer_idx, sae, language_neurons
            )
            # Bootstrap stability
            stability = compute_stability_bootstrap(
                delta_Z_stability
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

