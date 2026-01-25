"""
Neural FOXP2 Configuration Module

Contains all hyperparameters and configuration settings for the pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """Configuration for Neural FOXP2 pipeline."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token: str = ""  # Set your HuggingFace token here
    
    # Layer range for SAE training
    layer_range: List[int] = field(default_factory=lambda: list(range(8, 24)))
    
    # SAE hyperparameters
    n_features: int = 512
    epochs: int = 150
    lr: float = 5e-4
    lambda_sparse: float = 5e-3
    
    # Data settings
    n_prompts: int = 2500
    dataset_name: str = "cfilt/iitb-english-hindi"
    dataset_split: str = "test"
    seed: int = 42
    
    # Stage I settings
    top_k_features: int = 20
    lift_alphas: Tuple[float, ...] = (5.0, 10.0, 15.0)
    
    # Stage II settings
    r_max: int = 10
    min_window: int = 3
    max_window: int = 6
    n_bootstrap: int = 8
    n_prompts_svd: int = 500
    n_prompts_stability: int = 600
    
    # Stage III settings
    lambda_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    beta_ratios: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    n_eval_prompts: int = 15
    n_prototype_prompts: int = 200
    
    # Weak prompts for testing
    weak_prompts: List[str] = field(default_factory=lambda: [
        "Answer briefly.",
        "Give a short explanation.",
        "What is the correct answer?",
        "Explain in one sentence.",
        "Learning never stops when curiosity leads forward.",
        "Small daily efforts build surprisingly strong futures.",
        "Discipline beats motivation on most ordinary days.",
        "Clear goals turn confusion into steady progress.",
        "Consistent practice sharpens skills faster than talent.",
        "Good questions unlock deeper understanding over time.",
        "Focus today creates opportunities you meet tomorrow.",
        "Patience helps hard problems slowly reveal solutions.",
        "Curiosity drives research beyond comfortable assumptions daily.",
        "Write clearly to think better and decide.",
        "Learning compounds quietly when habits remain consistent.",
        "Failure teaches lessons success often hides completely.",
        "Strong foundations support ambitious ideas under pressure.",
        "Progress favors those who start before ready.",
        "Feedback accelerates improvement when received with humility.",
        "Attention is limited so spend it intentionally.",
        "Complex problems simplify after defining constraints carefully.",
        "Practice transforms theory into reliable real performance.",
        "Consistency today protects momentum during difficult weeks.",
        "Calm thinking improves decisions under unexpected pressure."
    ])


# Global config instance
config = Config()
