"""
Neural FOXP2 Models Module

Contains the Sparse Autoencoder class, model loading utilities,
and architecture-aware layer access helpers.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import sys
import os

# Add parent directory to path so config.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting interpretable features from LLM activations.
    
    Implements tied weights: decode uses W.T
    """
    
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_features))
        self.d_model = d_model
        self.n_features = n_features

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to sparse feature activations."""
        return torch.relu(h @ self.W + self.b)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to hidden state space."""
        return z @ self.W.T
    
    def forward(self, h: torch.Tensor) -> tuple:
        """Full forward pass: encode then decode."""
        z = self.encode(h)
        recon = self.decode(z)
        return recon, z


def get_model_layers(model) -> nn.ModuleList:
    """
    Return the list of transformer layer modules, regardless of architecture.
    
    Supports:
      - LLaMA / Mistral / Qwen: model.model.layers
      - Falcon: model.transformer.h
      - GPT-NeoX: model.gpt_neox.layers
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # LLaMA, Mistral, Qwen
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # Falcon, GPT-2 family
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # GPT-NeoX
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model).__name__}. "
            f"Cannot locate transformer layers."
        )


def load_model_and_tokenizer(
    model_name: str = None,
    hf_token: str = None,
    device_map: str = "auto"
):
    """
    Load a causal language model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        hf_token: HuggingFace API token
        device_map: Device placement strategy
        
    Returns:
        (model, tokenizer) tuple
    """
    model_name = model_name or config.model_name
    hf_token = hf_token or config.hf_token
    
    if hf_token:
        login(token=hf_token)
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Verify layer access works
    layers = get_model_layers(model)
    print(f"Model loaded successfully. Architecture: {type(model).__name__}, "
          f"{len(layers)} layers, device={next(model.parameters()).device}")
    
    return model, tokenizer
