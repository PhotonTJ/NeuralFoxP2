"""
Neural FOXP2 Configuration Module

Contains all hyperparameters and configuration settings for the pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token: str = ""  # Set your HuggingFace token here
    
    # Layer range for SAE training
    layer_range: List[int] = field(default_factory=lambda: list(range(8, 24)))
    
    # SAE hyperparameters
    n_features: int = 1024
    epochs: int = 150
    lr: float = 5e-4
    lambda_sparse: float = 5e-3
    
    # Data settings
    n_prompts: int = 2500
    dataset_name: str = "cfilt/iitb-english-hindi"
    dataset_split: str = "test"
    seed: int = 42
    
    # Stage I settings
    top_k_features: int = 50
    lift_alphas: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0)
    
    # Stage II settings
    r_max: int = 12
    min_window: int = 3
    max_window: int = 6
    n_bootstrap: int = 20
    n_prompts_svd: int = 1000
    n_prompts_stability: int = 1200
    
    # Stage III setting
    lambda_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    beta_ratios: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    n_eval_prompts: int = 400
    n_prototype_prompts: int = 200
    
    # Weak prompts for testing
    weak_prompts: List[str] = field(default_factory=lambda: [ "Answer briefly.",
    "Give a short response.",
    "Explain in one sentence.",
    "Provide a concise answer.",
    "Summarize clearly.",
    "Respond simply.",
    "State the result.",
    "Give the conclusion.",
    "Describe briefly.",
    "Answer directly.",
    "What is the outcome?",
    "What should be done?",
    "What is the reason?",
    "What is the explanation?",
    "What is the solution?",
    "What happens next?",
    "What is the main idea?",
    "What is the key point?",
    "What is the correct answer?",
    "What is the summary?",
    "Write a short note.",
    "Write one sentence.",
    "Provide a brief description.",
    "Give a quick explanation.",
    "Respond in a few words.",
    "State it clearly.",
    "Explain simply.",
    "Answer in short form.",
    "Describe the idea.",
    "Provide the answer only.",
    "Learning improves with practice.",
    "Small steps lead to progress.",
    "Consistency builds strong habits.",
    "Clear thinking supports good decisions.",
    "Patience helps solve problems.",
    "Effort produces results over time.",
    "Focus increases performance.",
    "Simple plans work best.",
    "Careful work avoids mistakes.",
    "Curiosity encourages discovery.",
    "Progress happens gradually each day.",
    "Understanding grows with reflection.",
    "Good habits create stability.",
    "Preparation reduces uncertainty.",
    "Attention improves accuracy.",
    "Calm thinking prevents errors.",
    "Practice strengthens skills.",
    "Balance leads to better outcomes.",
    "Planning saves time later.",
    "Clear goals guide action.",
    "Provide a short explanation of this.",
    "Give a simple answer to this.",
    "Explain the idea briefly.",
    "Respond with clarity.",
    "State the answer plainly.",
    "Describe the concept shortly.",
    "Offer a brief response.",
    "Summarize the point.",
    "Explain without details.",
    "Answer concisely.",
    "Knowledge grows through effort.",
    "Careful steps avoid confusion.",
    "Thinking deeply improves results.",
    "Steady work brings improvement.",
    "Simple answers are often best.",
    "Clarity supports understanding.",
    "Good structure helps communication.",
    "Small changes create impact.",
    "Learning never fully stops.",
    "Practice leads to mastery."
    ])


# Global config instance
config = Config()

