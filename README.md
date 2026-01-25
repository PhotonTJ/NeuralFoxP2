# Neural FOXP2

**Language Steering via Sparse Autoencoder Interventions**

Implementation of the Neural FOXP2 methodology for identifying and intervening on language neurons in Large Language Models.

## Overview

This project implements a three-stage pipeline for language control in LLMs:

1. **Stage I: Localize Language Neurons**
   - Train Sparse Autoencoders (SAEs) on model activations
   - Compute selectivity scores for English vs Hindi
   - Measure causal lift via interventions
   - Identify language-specific neural features

2. **Stage II: Low-Rank Steering Directions**
   - Compute language shift matrices (English → Hindi)
   - Perform SVD to find steering subspace
   - Select optimal contiguous layer window

3. **Stage III: Intervention Edit Rule**
   - Compute prototype directions (μ_hi, μ_en)
   - Grid search for optimal intervention strength (λ, β)
   - Apply signed sparse edits for language steering

## Installation

```bash
# Clone the repository
cd D:\NeuralFoxP2

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Full Pipeline

```bash
python main.py --hf-token YOUR_TOKEN --layers 8-23 --n-prompts 2500
```

### Single Layer (Quick Test)

```bash
python main.py --hf-token YOUR_TOKEN --layers 18 --epochs 50
```

### Run Specific Stage

```bash
# Run only Stage I
python main.py --stage 1 --hf-token YOUR_TOKEN

# Resume from Stage I checkpoint
python main.py --stage 2 --resume-from ./outputs/stage1_checkpoint.pkl
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | "" | HuggingFace API token |
| `--layers` | "8-23" | Layer range (e.g., "8-23" or "18") |
| `--n-prompts` | 2500 | Number of parallel prompts |
| `--epochs` | 150 | SAE training epochs |
| `--output-dir` | "./outputs" | Results directory |
| `--resume-from` | None | Checkpoint path to resume from |
| `--stage` | "all" | Stage to run: "all", "1", "2", or "3" |

## Project Structure

```
NeuralFoxP2/
├── config.py       # Configuration and hyperparameters
├── models.py       # SAE class and model loading
├── data.py         # Data loading and preprocessing
├── utils.py        # Helper functions
├── stage1.py       # Stage I: Language neuron localization
├── stage2.py       # Stage II: Steering direction identification
├── stage3.py       # Stage III: Intervention edit rule
├── main.py         # Main pipeline runner
├── README.md       # This file
└── requirements.txt
```

## Configuration

Edit `config.py` to modify hyperparameters:

```python
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    layer_range: List[int] = field(default_factory=lambda: list(range(8, 24)))
    n_features: int = 512        # SAE feature dimensions
    epochs: int = 150            # SAE training epochs
    lr: float = 5e-4             # Learning rate
    lambda_sparse: float = 5e-3  # Sparsity penalty
    # ... more options
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with 24GB+ VRAM (for Llama-3.1-8B)
- HuggingFace account with Llama access

## Output

The pipeline saves checkpoints after each stage:
- `outputs/stage1_checkpoint.pkl` - SAEs and language neurons
- `outputs/stage2_checkpoint.pkl` - Steering directions
- `outputs/final_checkpoint.pkl` - Complete results

## License

MIT License
