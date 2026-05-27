# Neural FOXP2

**Language Steering via Sparse Autoencoder Interventions — Multi-Language Pipeline**

Neural FOXP2 identifies and manipulates language-specific neurons inside Large Language Models using Sparse Autoencoders. This repository supports **Hindi** and **Spanish** pipelines, selectable at runtime.
(To keep the presentation concise, this repository includes analysis for only two languages. The approach, however, is readily extensible to additional languages, as noted in the paper.)
---

## Project Structure

```
NeuralFoxP2/
├── main.py              # Root launcher (language selector)
├── config.py            # Shared configuration & hyperparameters
├── README.md            # This file
│
├── hindi/               # Hindi ↔ English pipeline
│   ├── main.py          # Pipeline entry point
│   ├── config.py        # (imports shared config)
│   ├── data.py          # Data loading & preprocessing
│   ├── models.py        # SAE class & model loading
│   ├── utils.py         # Helper functions
│   ├── stage1.py        # Stage I  — Language neuron localization
│   ├── stage2.py        # Stage II — Steering direction identification
│   ├── stage3.py        # Stage III — Intervention edit rule
│   ├── requirements.txt
│   └── README.md
│
└── spanish/             # Spanish ↔ English pipeline
    ├── main.py
    ├── config.py
    ├── data.py
    ├── models.py
    ├── utils.py
    ├── stage1.py
    ├── stage2.py
    ├── stage3.py
    ├── requirements.txt
    └── README.md
```

---

## Pipeline Overview

Both language pipelines follow the same three-stage process:

| Stage | Name | What it does |
|-------|------|-------------|
| **I** | Localize Language Neurons | Train SAEs on model activations, compute selectivity & causal lift, identify language-specific features |
| **II** | Low-Rank Steering Directions | Compute language shift matrices, SVD for steering subspace, select optimal layer window |
| **III** | Intervention Edit Rule | Compute prototype directions, grid-search for optimal λ/β, apply signed sparse edits |

---

## Prerequisites

- **Python** 3.8+
- **PyTorch** 2.0+
- **NVIDIA GPU** with ≥ 24 GB VRAM (for Llama-3.1-8B)
- **HuggingFace** account with Llama model access and an API token

---

## Installation

```bash
cd D:\NeuralFoxP2

# Install dependencies (same for both languages)
pip install -r hindi/requirements.txt
```

---

## Usage

### Recommended — via root launcher

The root `main.py` lets you pick the language and forwards all arguments to the selected pipeline.

```bash
# Full Hindi pipeline
python main.py --language hindi --hf-token YOUR_TOKEN

# Full Spanish pipeline
python main.py --language spanish --hf-token YOUR_TOKEN

# Single-layer quick test (Hindi)
python main.py --language hindi --hf-token YOUR_TOKEN --layers 18 --epochs 50

# Run only Stage 1
python main.py --language hindi --stage 1 --hf-token YOUR_TOKEN

# Resume from a checkpoint
python main.py --language spanish --stage 2 --resume-from ./outputs/stage1_checkpoint.pkl
```

### Alternative — run directly from a language folder

```bash
cd hindi
python main.py --hf-token YOUR_TOKEN --layers 8-23 --n-prompts 2500

cd ../spanish
python main.py --hf-token YOUR_TOKEN --layers 8-23 --n-prompts 2500
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--language` | *(required)* | Language pipeline: `hindi` or `spanish` *(root launcher only)* |
| `--hf-token` | `""` | HuggingFace API token |
| `--layers` | `8-23` | Layer range (e.g. `8-23` or `18`) |
| `--n-prompts` | `2500` | Number of parallel prompts |
| `--epochs` | `150` | SAE training epochs |
| `--output-dir` | `./outputs` | Directory for results & checkpoints |
| `--resume-from` | `None` | Path to a checkpoint `.pkl` file |
| `--stage` | `all` | Stage to run: `all`, `1`, `2`, or `3` |

---

## Configuration

Edit `config.py` (root-level) to modify shared hyperparameters:

```python
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    layer_range: List[int] = field(default_factory=lambda: list(range(8, 24)))
    n_features: int = 1024
    epochs: int = 150
    lr: float = 5e-4
    lambda_sparse: float = 5e-3
    # ... see config.py for full list
```

---

## Output

Each pipeline saves checkpoints after every stage:

| File | Contents |
|------|----------|
| `outputs/stage1_checkpoint.pkl` | SAEs and language neurons |
| `outputs/stage2_checkpoint.pkl` | Steering directions & layer window |
| `outputs/final_checkpoint.pkl` | Complete results (all stages) |

---

## License

MIT License
