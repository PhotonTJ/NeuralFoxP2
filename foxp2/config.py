"""Model registry, language registry and the frozen FOXP2 operating protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


# --------------------------------------------------------------------------------------
# Models and their public SAE suites (all residual-stream, every layer).
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    key: str
    hf_id: str
    arch: str                     # "llama" | "gemma2" | "qwen3"
    n_layers: int
    d_model: int
    sae_family: str               # "gemma_scope" | "llama_scope" | "qwen_scope"
    sae_repo: str
    sae_width: Optional[str] = None   # Gemma Scope width folder, e.g. "width_16k"
    sae_topk: Optional[int] = None    # Qwen-Scope TopK
    chat_kwargs: dict = field(default_factory=dict)
    attn_implementation: Optional[str] = None


MODELS: dict[str, ModelSpec] = {
    "llama31-8b": ModelSpec(
        key="llama31-8b", hf_id="meta-llama/Llama-3.1-8B-Instruct", arch="llama",
        n_layers=32, d_model=4096, sae_family="llama_scope",
        sae_repo="OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x"),
    "gemma2-9b": ModelSpec(
        key="gemma2-9b", hf_id="google/gemma-2-9b-it", arch="gemma2",
        n_layers=42, d_model=3584, sae_family="gemma_scope",
        sae_repo="google/gemma-scope-9b-pt-res", sae_width="width_16k",
        attn_implementation="eager"),
    "gemma2-2b": ModelSpec(
        key="gemma2-2b", hf_id="google/gemma-2-2b-it", arch="gemma2",
        n_layers=26, d_model=2304, sae_family="gemma_scope",
        sae_repo="google/gemma-scope-2b-pt-res", sae_width="width_16k",
        attn_implementation="eager"),
    "qwen3-8b": ModelSpec(
        key="qwen3-8b", hf_id="Qwen/Qwen3-8B", arch="qwen3",
        n_layers=36, d_model=4096, sae_family="qwen_scope",
        sae_repo="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", sae_topk=50,
        # Non-thinking mode, otherwise the first decoded tokens are "<think>" scaffolding.
        chat_kwargs={"enable_thinking": False}),
}

# --------------------------------------------------------------------------------------
# Languages (FLORES+ config names).  Chinese is "cmn_Hans" in FLORES+, not "zho_Hans".
# --------------------------------------------------------------------------------------
FLORES_CODE = {
    "en": "eng_Latn", "hi": "hin_Deva", "zh": "cmn_Hans", "bn": "ben_Beng",
    "te": "tel_Telu", "es": "spa_Latn",
    # extension targets: low-resource Latin, low-resource own script, high-resource Arabic
    # script, and a Devanagari language that shares Hindi's script
    "sw": "swh_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "mr": "mar_Deva",
    # high-resource European Latin-script languages: close to English in script and vocabulary
    "de": "deu_Latn", "fr": "fra_Latn",
    # related / confusable languages used only for the leakage matrix
    "ur": "urd_Arab", "ne": "npi_Deva",
}
LANG_NAME = {
    "en": "English", "hi": "Hindi", "zh": "Chinese", "bn": "Bengali", "te": "Telugu",
    "es": "Spanish", "sw": "Swahili", "am": "Amharic", "ar": "Arabic", "mr": "Marathi",
    "de": "German", "fr": "French",
    "ur": "Urdu", "ne": "Nepali",
}
SCRIPT = {
    "en": "Latn", "hi": "Deva", "zh": "Hani", "bn": "Beng", "te": "Telu", "es": "Latn",
    "sw": "Latn", "am": "Ethi", "ar": "Arab", "mr": "Deva", "ur": "Arab", "ne": "Deva",
    "de": "Latn", "fr": "Latn",
}
TARGETS_CORE = ["hi", "zh", "bn", "te", "es"]          # the submitted paper's five
TARGETS_EXT = ["sw", "am", "ar", "mr", "de", "fr"]     # added for the revision
TARGETS = TARGETS_CORE + TARGETS_EXT
LEAK_EXTRA = ["ur", "ne"]


def leak_langs(target: str) -> list[str]:
    """Every non-target, non-English language whose mass we watch under a target edit."""
    return [l for l in TARGETS + LEAK_EXTRA if l != target]


def resolve_langs(arg: str) -> list[str]:
    """'all' | 'core' | 'ext' | comma list."""
    return {"all": TARGETS, "core": TARGETS_CORE, "ext": TARGETS_EXT}.get(arg) or arg.split(",")


# --------------------------------------------------------------------------------------
# Frozen protocol.  Every value is set once here (or on D_dev by the code) and logged.
# --------------------------------------------------------------------------------------
@dataclass
class FOXP2Config:
    # ---- data ----
    contrast: str = "response"      # "response" (recommended) | "input" (original paper)
    k_resp: int = 3                 # response positions per pair used for the contrast
    n_disc: int = 600               # FLORES+ dev sentences used for discovery
    n_dev: int = 397                # remaining FLORES+ dev sentences used for tuning
    n_eval: int = 300               # FLORES+ devtest sentences used for evaluation
    n_lift_prompts: int = 96        # weak discovery prompts for the causal-lift screen
    n_dev_prompts: int = 160        # weak dev prompts for window / strength tuning
    n_heldout_prompts: int = 300    # FLORES+ devtest items x eval templates for the held-out table
    token_ratio: float = 4.0        # one-vs-rest frequency ratio for diagnostic tokens
    token_min_count: int = 3

    # ---- Stage I ----
    layers: Optional[list[int]] = None   # None = every layer
    n_screen: int = 256             # selectivity candidates kept per layer and side
    lift_mode: str = "verify"       # "attribution" | "verify" | "intervention"
    n_verify: int = 48              # candidates per layer/side re-checked by real intervention
    n_verify_prompts: int = 32
    lift_alphas: tuple = (0.5, 1.0)  # in units of the feature's natural activation
    K_tgt: int = 64                 # target-promoting features per layer
    K_en: int = 32                  # English-promoting features per layer (suppression)
    T: int = 3                      # defaultness horizon (decoding steps)

    # ---- Stage II ----
    subspace: str = "uncentered"    # "uncentered" (paper) | "mean" (no SVD)
    rank_candidates: tuple = (1, 2, 4, 8)
    max_rank: int = 16
    n_boot: int = 200
    stab_min: float = 0.6
    min_width: int = 3
    max_width: int = 12
    n_window_candidates: int = 8
    alpha_leak: float = 2.0
    alpha_kl: float = 1.0
    alpha_util: float = 2.0
    alpha_stab: float = 0.05
    window_lam_grid: tuple = (2.0, 4.0, 8.0, 16.0)   # each candidate window is scored at its best lam
    window_beta: float = 0.5

    # ---- Stage III ----
    lam_grid: tuple = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)   # total strength; per layer = lam/|W|
    caa_lam_grid: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)  # CAA is ~4x stronger per unit
    beta_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    eps_leak: float = 0.08          # max mass gain of any non-target language
    eps_kl: float = 0.08            # max off-decision KL (already-committed contexts)
    eps_util: float = 0.03          # max in-language NLL increase (nats/token) at deployed positions
    util_skip: int = 3              # response tokens treated as the language commitment (not scored)
    # "prompt": a guardrail is max(fixed eps, cost of simply instructing "Answer in <L>.") on D_dev,
    #           i.e. steering may not cost more than prompting does.  "fixed": the eps values above.
    guardrail: str = "prompt"
    mode: str = "decode_window"     # "decode_window" | "all"
    k_decode: int = 8               # decode steps edited after the prompt-final position
    use_gate: bool = True           # explicit-instruction gate
    gate_min_acc: float = 0.80

    # ---- misc ----
    batch_size: int = 16
    seed: int = 17

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
