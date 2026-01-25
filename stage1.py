"""
Neural FOXP2 Stage I Module

Stage I: Localize Language Neurons
- I-1: Dictionary training (SAE)
- I-2: Selectivity computation
- I-3/I-4: Causal lift measurement
- I-5: Language neuron set identification
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from config import config
from models import SparseAutoencoder
from utils import get_residuals, early_language_mass, compute_selectivity, normalize_activations


class SAEFeatureIntervention:
    """Hook for single-feature SAE intervention."""
    
    def __init__(self, sae: SparseAutoencoder, feature_idx: int, alpha: float):
        self.sae = sae
        self.j = feature_idx
        self.alpha = alpha

    def __call__(self, module, args, kwargs, output):
        is_tuple = isinstance(output, tuple)
        h_orig = output[0] if is_tuple else output
        h = h_orig.clone()

        # Extract last token
        if h.dim() == 3:
            h_last = h[:, -1, :].clone()
        else:
            h_last = h[-1, :].unsqueeze(0).clone()

        # Pure single-feature intervention in SAE space
        z_delta = torch.zeros(1, self.sae.n_features)
        z_delta[0, self.j] = self.alpha

        # Decode delta and move to original device/dtype
        delta_h = self.sae.decode(z_delta).to(device=h.device, dtype=h.dtype)
        h_new = h_last + delta_h

        # Write back
        if h.dim() == 3:
            h[:, -1, :] = h_new
        else:
            h[-1, :] = h_new.squeeze(0)

        return (h,) + output[1:] if is_tuple else h


def train_sae(
    model,
    tokenizer,
    matched_prompts: List[Dict],
    layer: int,
    n_features: int = None,
    epochs: int = None,
    lr: float = None,
    lambda_sparse: float = None
) -> SparseAutoencoder:
    """
    Train a Sparse Autoencoder for a specific layer.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        matched_prompts: List of en/hi prompt pairs
        layer: Layer index
        n_features: SAE feature count
        epochs: Training epochs
        lr: Learning rate
        lambda_sparse: Sparsity penalty weight
        
    Returns:
        Trained SparseAutoencoder
    """
    n_features = n_features or config.n_features
    epochs = epochs or config.epochs
    lr = lr or config.lr
    lambda_sparse = lambda_sparse or config.lambda_sparse
    
    # Collect activations
    acts = []
    for pair in matched_prompts:
        acts.append(get_residuals(model, tokenizer, pair["en"], layer))
        acts.append(get_residuals(model, tokenizer, pair["hi"], layer))

    acts = torch.tensor(np.stack(acts), dtype=torch.float32)
    acts = normalize_activations(acts)

    # Initialize SAE
    sae = SparseAutoencoder(d_model=acts.shape[1], n_features=n_features)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(epochs), desc=f"Training SAE Layer {layer}"):
        perm = torch.randperm(acts.size(0))
        acts_shuffled = acts[perm]

        z = sae.encode(acts_shuffled)
        recon = sae.decode(z)

        recon_loss = ((acts_shuffled - recon) ** 2).mean()
        sparse_loss = z.abs().mean()
        loss = recon_loss + lambda_sparse * sparse_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize columns
        with torch.no_grad():
            sae.W[:] = sae.W / (sae.W.norm(dim=0, keepdim=True) + 1e-6)

    return sae


def train_all_saes(
    model,
    tokenizer,
    matched_prompts: List[Dict],
    layer_range: List[int] = None
) -> Dict[int, SparseAutoencoder]:
    """
    Train SAEs for all layers in range.
    
    Returns:
        Dictionary mapping layer index to trained SAE
    """
    layer_range = layer_range or config.layer_range
    sae_per_layer = {}
    
    for layer in layer_range:
        print(f"\n{'='*60}")
        print(f"Training SAE for Layer {layer}")
        print(f"{'='*60}")
        
        sae = train_sae(model, tokenizer, matched_prompts, layer)
        sae_per_layer[layer] = sae
        
    return sae_per_layer


@torch.no_grad()
def compute_language_activations(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    matched_prompts: List[Dict],
    layer: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SAE activations for English and Hindi prompts.
    
    Returns:
        (z_en, z_hi) arrays of shape [N, n_features]
    """
    z_en, z_hi = [], []

    for pair in matched_prompts:
        h_en = torch.tensor(get_residuals(model, tokenizer, pair["en"], layer), dtype=torch.float32)
        h_hi = torch.tensor(get_residuals(model, tokenizer, pair["hi"], layer), dtype=torch.float32)

        h_en = h_en / (h_en.norm() + 1e-6)
        h_hi = h_hi / (h_hi.norm() + 1e-6)

        z_en.append(sae.encode(h_en).cpu().numpy())
        z_hi.append(sae.encode(h_hi).cpu().numpy())

    return np.stack(z_en), np.stack(z_hi)


def compute_feature_lift(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    layer: int,
    feature_idx: int,
    prompts: List[str],
    V_hi: List[int],
    V_en: List[int],
    alphas: Tuple[float, ...] = None
) -> float:
    """
    Compute causal lift slope for a single SAE feature.
    
    LiftSlope = median_α(ΔM_α / α)
    
    Returns:
        Lift slope value
    """
    alphas = alphas or config.lift_alphas
    
    # Baseline (no intervention)
    baseline_masses = {}
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        mass = early_language_mass(outputs.logits[:, -1, :], V_hi, V_en)
        baseline_masses[p] = mass
    
    slopes = []

    for alpha in alphas:
        for sign in [+1, -1]:
            hook = SAEFeatureIntervention(sae, feature_idx, sign * alpha)
            handle = model.model.layers[layer].register_forward_hook(hook, with_kwargs=True)

            deltas = []
            for p in prompts:
                inputs = tokenizer(p, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                mass = early_language_mass(outputs.logits[:, -1, :], V_hi, V_en)
                deltas.append(mass - baseline_masses[p])

            handle.remove()
            if deltas:
                slopes.append(np.mean(deltas) / (sign * alpha))

    return float(np.median(slopes)) if slopes else 0.0


def run_stage1(
    model,
    tokenizer,
    matched_prompts: List[Dict],
    V_hi: List[int],
    V_en: List[int],
    layer_range: List[int] = None,
    weak_prompts: List[str] = None
) -> Dict[str, Any]:
    """
    Run complete Stage I pipeline.
    
    Returns:
        Dictionary with per-layer results:
        - sae_per_layer: trained SAEs
        - selectivity_per_layer: selectivity scores
        - language_neurons_per_layer: identified language neurons
        - stage1_output_per_layer: full output per layer
    """
    layer_range = layer_range or config.layer_range
    weak_prompts = weak_prompts or config.weak_prompts
    
    # Train SAEs
    print("\n" + "="*70)
    print("STAGE I-A: Training SAEs")
    print("="*70)
    sae_per_layer = train_all_saes(model, tokenizer, matched_prompts, layer_range)
    
    # Compute selectivity and lifts
    selectivity_per_layer = {}
    top_features_per_layer = {}
    feature_lifts_per_layer = {}
    language_neurons_per_layer = {}
    stage1_output_per_layer = {}
    
    for layer in layer_range:
        print(f"\n{'='*60}")
        print(f"Stage I-B/C: Processing Layer {layer}")
        print(f"{'='*60}")
        
        sae = sae_per_layer[layer]
        
        # I-2: Selectivity
        z_en, z_hi = compute_language_activations(model, tokenizer, sae, matched_prompts, layer)
        selectivity = compute_selectivity(z_en, z_hi)
        selectivity_per_layer[layer] = selectivity
        
        # Top-K features
        top_idx = np.argsort(selectivity)[-config.top_k_features:]
        top_features_per_layer[layer] = top_idx
        
        # I-3/I-4: Causal lift
        feature_lifts = {}
        for j in top_idx:
            lift = compute_feature_lift(
                model, tokenizer, sae, layer, int(j),
                weak_prompts, V_hi, V_en
            )
            feature_lifts[j] = lift
        feature_lifts_per_layer[layer] = feature_lifts
        
        # I-5: Joint scoring
        S = np.maximum(selectivity[top_idx], 0)
        C = np.maximum(np.array([feature_lifts[j] for j in top_idx]), 0)
        joint_scores = S * C
        
        ranked_indices = np.argsort(joint_scores)[::-1]
        ranked_features = top_idx[ranked_indices]
        ranked_scores = joint_scores[ranked_indices]
        
        language_neurons = [int(j) for j, score in zip(ranked_features, ranked_scores) if score > 0]
        language_neurons_per_layer[layer] = language_neurons
        
        print(f"Layer {layer}: {len(language_neurons)} language neurons identified")
        
        stage1_output_per_layer[layer] = {
            "layer": layer,
            "selectivity": selectivity,
            "top_features": top_idx,
            "feature_lifts": feature_lifts,
            "language_neurons": language_neurons,
            "joint_scores": dict(zip(ranked_features.tolist(), ranked_scores.tolist()))
        }
    
    return {
        "sae_per_layer": sae_per_layer,
        "selectivity_per_layer": selectivity_per_layer,
        "language_neurons_per_layer": language_neurons_per_layer,
        "stage1_output_per_layer": stage1_output_per_layer
    }
