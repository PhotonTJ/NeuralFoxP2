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

    device = next(model.parameters()).device
    shifts = []
    idx = torch.tensor(language_neurons, dtype=torch.long, device=device)

    for pair in prompts:
        h_en = torch.tensor(
            get_residuals(model, tokenizer, pair["en"], layer_idx),
            dtype=torch.float32,
            device=device
        )
        h_hi = torch.tensor(
            get_residuals(model, tokenizer, pair["hi"], layer_idx),
            dtype=torch.float32,
            device=device
        )

        
        h_en = h_en / (h_en.norm() + 1e-6)
        h_hi = h_hi / (h_hi.norm() + 1e-6)

        with torch.no_grad():
            z_en = sae.encode(h_en.unsqueeze(0))[0]
            z_hi = sae.encode(h_hi.unsqueeze(0))[0]

        shift = torch.zeros_like(z_en)
        shift[idx] = z_hi[idx] - z_en[idx]
        
        shifts.append(shift)

    mu_hi = torch.stack(shifts).mean(dim=0)
    
    return normalize_vector(mu_hi.cpu().numpy())


def compute_english_attractor(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    sae: SparseAutoencoder,
    language_neurons: List[int]
) -> np.ndarray:

    device = next(model.parameters()).device
    activations = []
    idx = torch.tensor(language_neurons, dtype=torch.long, device=device)

    for p in prompts:
        h = torch.tensor(
            get_residuals(model, tokenizer, p, layer_idx),
            dtype=torch.float32,
            device=device
        )
        h = h / (h.norm() + 1e-6)
        with torch.no_grad():
            z = sae.encode(h.unsqueeze(0))[0]

        z_restricted = torch.zeros_like(z)
        z_restricted[idx] = z[idx]
        
        activations.append(z_restricted)

    mu_en = torch.stack(activations).mean(dim=0)
    return normalize_vector(mu_en.cpu().numpy())


class Stage3InterventionHook:
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

        device = next(sae.parameters()).device
        
        mu_hi = torch.tensor(proto["mu_hi"], device=device)
        mu_en = torch.tensor(proto["mu_en"], device=device)
        P = torch.tensor(proto["projector"], device=device, dtype=torch.float32)
        
        language_neurons = torch.tensor(
            sorted(self.language_neurons_per_layer[layer_idx]),
            device=device,
            dtype=torch.long
        )
        
        def hook(module, args, kwargs, output):
            self.call_count += 1
            
            is_tuple = isinstance(output, tuple)
            h_orig = output[0] if is_tuple else output
            h = h_orig.clone()

            if h.dim() == 3:
                h_last = h[:, -1, :]
            else:
                h_last = h[-1, :].unsqueeze(0)

            z = sae.encode(h_last)[0]

            # Negative: English suppression inside steering subspace
            proj_coef = torch.dot(z, mu_en)
            neg_dir = P @ (proj_coef * mu_en)

            # Total feature delta
            delta_z = self.lambda_val * mu_hi - self.beta_val * neg_dir

            z_delta = torch.zeros_like(z)
            z_delta[language_neurons] = delta_z[language_neurons]

            delta_h = sae.decode(z_delta.unsqueeze(0).to(device))[0]

            h_new = h_last + delta_h

            if h.dim() == 3:
                h[:, -1, :] = h_new
            else:
                h[-1, :] = h_new.squeeze(0)

            if self.debug:
                print(
                    f"[STAGE III] |Δz|max={torch.max(torch.abs(z_delta)).item():.4f}, "
                    f"|Δh|={delta_h.norm().item():.4f}"
                )

            return (h,) + output[1:] if is_tuple else h

        return hook


def compute_baseline_defaultness(
    model,
    tokenizer,
    prompts: List[str],
    V_hi: List[int],
    V_en: List[int],
    batch_size=32
) -> float:

    device = next(model.parameters()).device
    masses = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        logits = outputs.logits[:, -1, :]
        mass = early_language_mass(logits, V_hi, V_en)
        
        masses.extend(mass.tolist())
        
    return float(np.mean(masses))


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
    V_en: List[int],
    batch_size=32
) -> float:
    
    intervention = Stage3InterventionHook(
        sae_per_layer,
        layer_prototypes,
        language_neurons_per_layer,
        lambda_val,
        beta_val
    )

    handles = []
    
    try:
        for layer_idx in intervention_window:
            hook = intervention.make_hook(layer_idx)
            handle = model.model.layers[layer_idx].register_forward_hook(
                hook, 
                with_kwargs=True
            )
            handles.append(handle)
        
        device = next(model.parameters()).device
        edited_masses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
    
            logits = outputs.logits[:, -1, :]
            mass = early_language_mass(logits, V_hi, V_en)
            
            edited_masses.extend(mass.tolist())
        
    finally:
        for h in handles:
            h.remove()

    return float(np.mean(edited_masses))


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
   
    weak_prompts = weak_prompts or config.weak_prompts
    
    
    print("STAGE III: Intervention Edit Rule")
    
    
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

