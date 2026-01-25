"""
Neural FOXP2 Stage III Module

Stage III: Signed Sparse Edit and Intervention
- III-1: Signed edit components (δz)
- III-2: Apply sparse code update
- III-3: Defaultness gain measurement
"""
import torch
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from config import config
from models import SparseAutoencoder
from utils import get_residuals, early_language_mass, normalize_vector


def compute_target_prototype(
    model,
    tokenizer,
    prompts: List[Dict],
    layer_idx: int,
    sae: SparseAutoencoder,
    language_neurons: List[int]
) -> np.ndarray:
    """
    Compute μ_hi = E[ Π_N ( z_hi − z_en ) ]
    Mean English→Hindi shift, restricted to language neurons.
    """
    shifts = []
    idx = np.array(language_neurons, dtype=int)

    for pair in prompts:
        h_en = torch.tensor(get_residuals(model, tokenizer, pair["en"], layer_idx), dtype=torch.float32)
        h_hi = torch.tensor(get_residuals(model, tokenizer, pair["hi"], layer_idx), dtype=torch.float32)

        h_en = h_en / (h_en.norm() + 1e-6)
        h_hi = h_hi / (h_hi.norm() + 1e-6)

        z_en = sae.encode(h_en.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        z_hi = sae.encode(h_hi.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        shift = np.zeros_like(z_en)
        shift[idx] = z_hi[idx] - z_en[idx]
        shifts.append(shift)

    mu_hi = np.mean(shifts, axis=0)
    return normalize_vector(mu_hi)


def compute_english_attractor(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    sae: SparseAutoencoder,
    language_neurons: List[int]
) -> np.ndarray:
    """
    Compute μ_en = E[ Π_N ( z(x) ) ]
    English attractor direction, restricted to language neurons.
    """
    activations = []
    idx = np.array(language_neurons, dtype=int)

    for p in prompts:
        h = torch.tensor(get_residuals(model, tokenizer, p, layer_idx), dtype=torch.float32)
        h = h / (h.norm() + 1e-6)
        z = sae.encode(h.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        z_restricted = np.zeros_like(z)
        z_restricted[idx] = z[idx]
        activations.append(z_restricted)

    mu_en = np.mean(activations, axis=0)
    return normalize_vector(mu_en)


class Stage3InterventionHook:
    """
    FOXP2 Stage III signed sparse intervention.
    Uses per-layer SAEs and language neurons.
    """

    def __init__(
        self,
        sae_per_layer: Dict[int, SparseAutoencoder],
        layer_prototypes: Dict[int, Dict],
        language_neurons_per_layer: Dict[int, List[int]],
        lambda_val: float,
        beta_val: float,
        debug: bool = False
    ):
        self.sae_per_layer = sae_per_layer
        self.prototypes = layer_prototypes
        self.language_neurons_per_layer = language_neurons_per_layer
        self.lambda_val = lambda_val
        self.beta_val = beta_val
        self.debug = debug
        self.call_count = 0

    def make_hook(self, layer_idx: int):
        proto = self.prototypes[layer_idx]
        sae = self.sae_per_layer[layer_idx]
        language_neurons = np.array(sorted(self.language_neurons_per_layer[layer_idx]), dtype=int)

        mu_hi = proto["mu_hi"]
        mu_en = proto["mu_en"]
        P = proto["projector"]
        pos_dir = mu_hi

        def hook(module, args, kwargs, output):
            self.call_count += 1
            is_tuple = isinstance(output, tuple)
            h_orig = output[0] if is_tuple else output
            h = h_orig.clone()

            if h.dim() == 3:
                h_last = h[:, -1, :]
            else:
                h_last = h[-1, :].unsqueeze(0)

            h_cpu = h_last.detach().cpu().float()
            z = sae.encode(h_cpu).detach().cpu().numpy()[0]

            # Negative: English suppression inside steering subspace
            proj_coef = np.dot(z, mu_en)
            neg_dir = P @ (proj_coef * mu_en)

            # Total feature delta
            delta_z = self.lambda_val * pos_dir - self.beta_val * neg_dir

            z_delta = np.zeros_like(z)
            z_delta[language_neurons] = delta_z[language_neurons]

            delta_h = sae.decode(
                torch.tensor(z_delta, dtype=torch.float32)
            ).to(h.device, dtype=h.dtype)

            h_new = h_last + delta_h

            if h.dim() == 3:
                h[:, -1, :] = h_new
            else:
                h[-1, :] = h_new.squeeze(0)

            if self.debug:
                print(f"[STAGE III] |Δz|max={np.max(np.abs(z_delta)):.4f}, "
                      f"|Δh|={delta_h.norm().item():.4f}")

            return (h,) + output[1:] if is_tuple else h

        return hook


def compute_baseline_defaultness(
    model,
    tokenizer,
    prompts: List[str],
    V_hi: List[int],
    V_en: List[int]
) -> float:
    """Compute baseline language mass without intervention."""
    masses = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
        mass = early_language_mass(outputs.logits[:, -1, :], V_hi, V_en)
        masses.append(mass)
    return np.mean(masses)


def evaluate_edit_strength(
    model,
    tokenizer,
    sae_per_layer: Dict[int, SparseAutoencoder],
    layer_prototypes: Dict[int, Dict],
    language_neurons_per_layer: Dict[int, List[int]],
    intervention_window: List[int],
    lambda_val: float,
    beta_val: float,
    prompts: List[str],
    V_hi: List[int],
    V_en: List[int]
) -> float:
    """Evaluate edit strength with given lambda/beta parameters."""
    intervention = Stage3InterventionHook(
        sae_per_layer, layer_prototypes, language_neurons_per_layer,
        lambda_val, beta_val
    )

    handles = []
    for layer_idx in intervention_window:
        hook = intervention.make_hook(layer_idx)
        handle = model.model.layers[layer_idx].register_forward_hook(hook, with_kwargs=True)
        handles.append(handle)

    edited_masses = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
        mass = early_language_mass(outputs.logits[:, -1, :], V_hi, V_en)
        edited_masses.append(mass)

    for h in handles:
        h.remove()

    return np.mean(edited_masses)


def run_stage3(
    model,
    tokenizer,
    matched_prompts: List[Dict],
    sae_per_layer: Dict[int, SparseAutoencoder],
    language_neurons_per_layer: Dict[int, List[int]],
    layer_results: Dict[int, Dict],
    intervention_window: List[int],
    V_hi: List[int],
    V_en: List[int],
    weak_prompts: List[str] = None
) -> Dict[str, Any]:
    """
    Run complete Stage III pipeline with grid search optimization.
    
    Returns:
        Dictionary with:
        - layer_prototypes: computed prototypes per layer
        - best_params: optimal (lambda, beta)
        - best_gain: defaultness gain
        - grid_results: full grid search results
    """
    weak_prompts = weak_prompts or config.weak_prompts
    
    print("\n" + "="*70)
    print("STAGE III: Intervention Edit Rule")
    print("="*70)
    
    # Compute prototypes for each layer in intervention window
    print("\nComputing prototype directions...")
    layer_prototypes = {}
    
    for layer_idx in tqdm(intervention_window, desc="Computing prototypes"):
        sae = sae_per_layer[layer_idx]
        language_neurons = language_neurons_per_layer[layer_idx]
        
        mu_hi = compute_target_prototype(
            model, tokenizer,
            matched_prompts[:config.n_prototype_prompts],
            layer_idx, sae, language_neurons
        )
        
        mu_en = compute_english_attractor(
            model, tokenizer,
            weak_prompts,
            layer_idx, sae, language_neurons
        )
        
        # Project into steering subspace
        r = layer_results[layer_idx]["rank"]
        V = layer_results[layer_idx]["steering_vectors"][:, :r]
        P = V @ V.T
        
        mu_hi_proj = normalize_vector(P @ mu_hi)
        mu_en_proj = normalize_vector(P @ mu_en)
        
        layer_prototypes[layer_idx] = {
            "mu_hi": mu_hi_proj,
            "mu_en": mu_en_proj,
            "projector": P,
            "steering_vectors": V,
            "rank": r
        }
        
        print(f"  Layer {layer_idx}: ||μ_hi||={np.linalg.norm(mu_hi_proj):.4f}, "
              f"||μ_en||={np.linalg.norm(mu_en_proj):.4f}")
    
    # Grid search for optimal lambda, beta
    print("\nGrid search for optimal (λ, β)...")
    
    eval_prompts = weak_prompts[:config.n_eval_prompts]
    baseline_mean = compute_baseline_defaultness(model, tokenizer, eval_prompts, V_hi, V_en)
    
    best_params = None
    best_gain = -float("inf")
    grid_results = []
    
    for lambda_val in config.lambda_grid:
        for beta_ratio in config.beta_ratios:
            beta_val = beta_ratio * lambda_val
            
            edited_mean = evaluate_edit_strength(
                model, tokenizer, sae_per_layer, layer_prototypes, language_neurons_per_layer,
                intervention_window, lambda_val, beta_val,
                eval_prompts, V_hi, V_en
            )
            
            gain = edited_mean - baseline_mean
            
            grid_results.append({
                "lambda": lambda_val,
                "beta": beta_val,
                "gain": gain,
                "baseline": baseline_mean,
                "edited": edited_mean
            })
            
            print(f"  λ={lambda_val:.1f}, β={beta_val:.1f}: gain={gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_params = (lambda_val, beta_val)
    
    print(f"\nBest parameters: λ={best_params[0]:.1f}, β={best_params[1]:.1f}, gain={best_gain:.4f}")
    
    return {
        "layer_prototypes": layer_prototypes,
        "best_params": best_params,
        "best_gain": best_gain,
        "grid_results": grid_results,
        "baseline_mean": baseline_mean
    }
