# Neural FOXP2: revised pipeline

Inference-time default-language steering with frozen public SAEs, revised according to the
ARR review.  It covers four instruction-tuned checkpoints and five target languages, and
exports one steered Hugging Face checkpoint per (model, language).  Four models × eleven
languages = 44 checkpoints.

| Key | Checkpoint | SAE suite (residual stream, every layer) |
|---|---|---|
| `llama31-8b` | meta-llama/Llama-3.1-8B-Instruct | Llama Scope `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` (JumpReLU, 32K) |
| `gemma2-9b` | google/gemma-2-9b-it | Gemma Scope `google/gemma-scope-9b-pt-res`, 16K, canonical L0≈100 |
| `gemma2-2b` | google/gemma-2-2b-it | Gemma Scope `google/gemma-scope-2b-pt-res`, 16K, canonical L0≈100 |
| `qwen3-8b` | Qwen/Qwen3-8B (non-thinking mode) | Qwen-Scope `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (TopK-50, 64K) |

Targets (`--langs all | core | ext | hi,sw,...`):

| Set | Languages | Why |
|---|---|---|
| `core` (the submitted paper's five) | `hi` hin_Deva, `zh` cmn_Hans, `bn` ben_Beng, `te` tel_Telu, `es` spa_Latn | |
| `ext` (added in the revision) | `sw` swh_Latn (Swahili) | Low-resource, Latin script: no script cue to lean on |
| | `am` amh_Ethi (Amharic) | Low-resource, its own script, weakly supported by all four models |
| | `ar` arb_Arab (Arabic) | High-resource, non-Latin, right-to-left; shares its script with Urdu |
| | `mr` mar_Deva (Marathi) | Shares Devanagari with Hindi and Nepali: the hardest leakage case |
| | `de` deu_Latn (German) | High-resource, Latin script, many cognates and shared subwords with English |
| | `fr` fra_Latn (French) | High-resource, Latin script; heavy vocabulary overlap with English and Spanish |

Leakage is tracked for every other target plus Urdu and Nepali. Marathi and Arabic need
GlotLID (`pip install fasttext-wheel`): a script heuristic cannot separate Marathi from
Hindi or Arabic from Urdu.  Benchmark coverage differs by language: XNLI covers `sw`, `ar`,
`de` and `fr`; XQuAD covers `de` and `ar` but
not `sw` or `fr`; Amharic and Swahili are in IrokoBench (AfriXNLI,
AfriMMLU); Marathi is in IndicXNLI.

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login          # then accept the FLORES+, Llama-3.1 and Gemma-2 terms on HF
python tests/smoke_test.py     # CPU, ~1 min, no downloads: checks the whole plumbing

# one pair first (smaller grids, fewer bootstrap draws)
python scripts/run_pipeline.py --models llama31-8b --langs hi --fast

# the full grid (4 models x 11 languages) + evaluation of every checkpoint
bash scripts/run_all.sh
```

Outputs:
- `runs/<model>/<lang>/` holds `stage1.pt` and `stage1_report.json` (FVU, token affinities), plus
  `stage2.json` (spectra, nulls, causal curve, windows).
- It also holds `stage3_grid.csv` (every λ/β point, every ablation) and `artifact.pt` / `artifact.json`.
- `checkpoints/<model>-foxp2-<lang>/` is the steered checkpoint.

## What changed relative to the submitted method

| Review issue | Change in code |
|---|---|
| Input language confounded with output language | **Response-language contrast**: same weak prompt, English vs target response, activations at the first `k_resp=3` response positions (`stage1.collect_contrast`). The original input contrast remains as `--contrast input` for the ablation. |
| Lift tested one feature at a time, no causal check | Attribution screen over every candidate (one backward pass per batch), then real additive interventions on the top candidates (`--lift_mode verify`). Lift is per unit of the feature's natural activation. `--lift_mode intervention` is fully gradient-free. |
| Base-SAE/IT mismatch "cannot alter the forward pass" | The FVU of each SAE on the IT model's own activations is measured and reported per layer. The Llama Scope threshold convention is chosen by FVU. |
| "Sparse + low-rank" may be just the mean | Uncentred **and** centred spectra, mean-shift share, ‖Pμ‖/‖μ‖. Three nulls go through the same pipeline: sign-flip rows, a random support, and an English-only wikinews-vs-wikivoyage contrast (`stage2.layer_analysis`). |
| Window chosen without a causal curve | A per-layer causal curve (single-layer edit gain on dev), restricted to bootstrap-stable layers. Top windows are then scored with the full J(W), with all weights logged. |
| Rank read off the spectrum only | Rank chosen by dev efficacy over {1,2,4,8, spectral, mean}, with the spectral choice reported alongside. |
| μ_en term is not English-specific | Fractional removal of **English-promoting features N_en** (Sel<0, lift<0). β=1 removes them fully and can never push them below zero. |
| Edit everywhere / overrides explicit requests | The edit covers the **prompt-final position + first `k_decode` steps** only. An **explicit-instruction gate** (linear probe, trained on dev templates, tested on held-out templates) turns steering off when the user names a language. |
| Per-layer vs global strengths; β frontier claim | One global (λ, β) with per-layer λ/\|W\|, at the max-gain *feasible* point under frozen guardrails. The full grid is saved. |
| KL / utility penalise the intended change | KL is measured on **already-committed** target contexts; utility is in-language NLL drift ("all"-position stress test). Neither penalises switching language. |
| Ablations at the full method's strength | Positive-only, negative-only, sparse-only, random support, out-of-window and **residual diff-in-means (CAA)** are each tuned separately under the same guardrails. |
| Early-token metric only; strawman prompt baseline | `evaluate.py` reports **response-level** language rate, switch-away rate and the response-language leakage matrix. It also reports prompt baselines at response level (including the instruction in the target language) and the conflict test. |
| FLORES discovery/eval overlap | Discovery and tuning use FLORES+ `dev` only; evaluation and translation use `devtest`. Prompt templates are disjoint across splits. |

## Steered checkpoints

Base weights are **symlinked** from the HF cache (a few MB per language). Use
`--copy_weights` to make self-contained copies you can move between machines.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "checkpoints/llama31-8b-foxp2-hi"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                             dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(path)
model.foxp2_steer.gamma = 0.0   # unedited model (bit-identical); 1.0 = tuned operating point
```

Runtime controls are available as `from_pretrained` kwargs or as environment variables (the env vars win):

| Setting | kwarg | env var |
|---|---|---|
| Strength γ | `foxp2_gamma` | `FOXP2_GAMMA` |
| Positions edited | `foxp2_mode="decode_window"` (generation) / `"all"` (log-likelihood tasks) | `FOXP2_MODE` |
| Decode steps edited | `foxp2_k_decode` | `FOXP2_KDECODE` |
| Instruction gate | `foxp2_use_gate` | `FOXP2_GATE=0/1` |

### Benchmarks with lm-evaluation-harness

```bash
CK=checkpoints/llama31-8b-foxp2-hi
# multiple-choice / log-likelihood tasks: edit every position
FOXP2_MODE=all lm_eval --model hf \
  --model_args pretrained=$CK,trust_remote_code=True,dtype=bfloat16 \
  --tasks xnli_hi --batch_size 8
# generative tasks: default decode_window mode
lm_eval --model hf --model_args pretrained=$CK,trust_remote_code=True,dtype=bfloat16 \
  --tasks <generative task> --batch_size 8
# baseline on the same weights
FOXP2_GAMMA=0 lm_eval ...
```

Points to note when evaluating:
- **XNLI and XQuAD have no Bengali or Telugu.** Use an Indic benchmark (e.g. IndicXNLI, IndicQA) for those languages. Check task names with `lm_eval --tasks list`.
- **The gate fires on prompts that name a language.** "Translate into Hindi: …" is one example, so translation scores are unchanged by design. Set `FOXP2_GATE=0` to stress-test the raw edit.
- **Supported harnesses.** The edit is a forward hook, so use HF-backend harnesses. vLLM/TGI will silently run the unedited model.

## Notes and limits
- **Compute.** The dominant costs are activation collection and interventional verification in Stage I, and the λ×β×ablation grid in Stage III. Start with `--fast` and `--skip_ablations`, then run the full protocol for the paper numbers.
- **Approximation for Qwen-Scope.** These SAEs are TopK; the exported steerer approximates TopK on the selected features with a calibrated threshold. Its agreement with exact TopK is logged in `stage1.pt` (`topk_approx_corr`).
- **Language ID.** This uses GlotLID if `fasttext` is installed, otherwise a script heuristic that cannot separate Hindi, Marathi and Nepali. Install GlotLID for the leakage matrix.
- **Llama Scope encoder convention.** The Llama Scope tensor names are matched by substring and the threshold convention is chosen by FVU. Check `stage1_report.json` for an FVU well below 0.5 before trusting a layer.

## Everything for the paper in one command

```bash
bash scripts/run_everything.sh                    # pipeline -> evaluate -> frontier -> lm-eval -> figures
FAST=1 LMLIMIT=200 bash scripts/run_everything.sh # quick first pass
MODELS=llama31-8b LANGS=hi,bn STEPS="evaluate figures" bash scripts/run_everything.sh
```

Figures (`scripts/make_figures.py`; each also writes a CSV with the exact numbers):

| Figure | Reviewer question it answers |
|---|---|
| fig01 frontier (per model, all languages) | Does FOXP2 beat CAA and prompting at equal utility cost? |
| fig02 causal curve heatmap + window boxes | Is the decision localized, and where? |
| fig03 low-rank evidence vs three nulls | Is "low-rank" more than a mean shift / more than any contrast? |
| fig04 SAE FVU by depth | Does the base-model dictionary transfer to the IT checkpoints? |
| fig05 support coverage | How much of the residual language shift do the SAE features carry? |
| fig06 held-out heatmap (methods × languages) | Every ablation, every language, on data never used for selection |
| fig07 held-out macro with 95% CIs | Are the differences between methods significant across languages? |
| fig08 response level | Do whole responses stay in the target language? Mid-response switching? |
| fig09 leakage matrix | Where do non-target responses go (English, related languages)? |
| fig10 conflict test + gate accuracy | Does "answer in English" still win? Does the gate generalize? |
| fig11 dose-response in λ | Is the effect controllable and monotone? |
| fig12 β interaction | Does English suppression do anything on its own? |
| fig13 benchmark deltas (target + English) | What does steering cost on real tasks? |
| fig14 selected windows | Are windows consistent across languages and models? |
| fig15 feature script share | Are the features script detectors or language-level? |
| table_main_heldout.tex | Main table: held-out DEFAULT gain [95% CI] / utility cost |

`python tests/fake_results.py /tmp/fake` writes SYNTHETIC outputs for testing the plotting
code only; never put those figures in a paper.
