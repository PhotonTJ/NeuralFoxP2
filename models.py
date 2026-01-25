"""
Neural FOXP2 Models Module

Contains the Sparse Autoencoder class and model loading utilities.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

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


def load_model_and_tokenizer(
    model_name: str = None,
    hf_token: str = None,
    device_map: str = "auto"
):
    """
    Load the LLaMA model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        hf_token: HuggingFace API token
        device_map: Device placement strategy
        
    Returns:
        model, tokenizer tuple
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
    
    print(f"Model loaded successfully on {next(model.parameters()).device}")
    
    return model, tokenizer
