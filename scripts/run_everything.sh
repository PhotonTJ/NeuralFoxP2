#!/usr/bin/env bash
# =====================================================================================
# Neural FOXP2: every run needed for the paper, all models x all 11 languages.
#
#   bash scripts/run_everything.sh                 # full protocol, everything
#   FAST=1 bash scripts/run_everything.sh          # quick settings (for a first pass)
#   MODELS=llama31-8b LANGS=hi,bn bash scripts/run_everything.sh
#   STEPS="figures" bash scripts/run_everything.sh # only redo some steps
#
# Steps (in order): pipeline evaluate frontier lmeval figures
# Prereqs: huggingface-cli login; accept the FLORES+, Llama-3.1 and Gemma-2 terms on HF;
#          pip install -r requirements.txt  (fasttext-wheel is needed for Marathi/Arabic LID)
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-llama31-8b,gemma2-9b,gemma2-2b,qwen3-8b}
LANGS=${LANGS:-hi,zh,bn,te,es,sw,am,ar,mr,de,fr}
STEPS=${STEPS:-pipeline evaluate frontier lmeval figures}
OUT=${OUT:-runs}; CK=${CK:-checkpoints}; RES=${RES:-results}; FIG=${FIG:-figures}
LMBS=${LMBS:-8}                       # lm-eval batch size
LMLIMIT=${LMLIMIT:-}                  # e.g. LMLIMIT=200 for a quick lm-eval pass
FASTFLAG=""; [[ "${FAST:-0}" == "1" ]] && FASTFLAG="--fast"
LIMITFLAG=""; [[ -n "$LMLIMIT" ]] && LIMITFLAG="--limit $LMLIMIT"
has() { [[ " $STEPS " == *" $1 "* ]]; }
IFS=',' read -ra MS <<< "$MODELS"; IFS=',' read -ra LS <<< "$LANGS"
log() { echo -e "\n==== $* ====  $(date '+%F %T')"; }

# ---- 1. Stages I-III + held-out ablation table + steered checkpoints ------------------
if has pipeline; then
  log "pipeline: models=$MODELS langs=$LANGS $FASTFLAG"
  python scripts/run_pipeline.py --models "$MODELS" --langs "$LANGS" --out "$OUT" \
      --export_dir "$CK" $FASTFLAG 2>&1 | tee -a pipeline.log
fi

# ---- 2. Response-level evaluation of every checkpoint (devtest, unseen templates) -----
if has evaluate; then
  for m in "${MS[@]}"; do for l in "${LS[@]}"; do
    ck="$CK/$m-foxp2-$l"; [[ -d "$ck" ]] || { echo "[skip] $ck missing"; continue; }
    if [[ -f "$ck/eval_$l.json" && "${FORCE_EVAL:-0}" != "1" ]]; then
      echo "[resume] $ck/eval_$l.json exists, skipping (FORCE_EVAL=1 to redo)"; continue
    fi
    log "evaluate $m $l"
    python scripts/evaluate.py --ckpt "$ck" --model "$m" --lang "$l" --translation 2>&1 | tee -a evaluate.log
  done; done
fi

# ---- 3. Per-pair frontier plots (gain vs utility and KL) -----------------------------
if has frontier; then
  for m in "${MS[@]}"; do for l in "${LS[@]}"; do
    [[ -f "$OUT/$m/$l/stage3_grid.csv" ]] && python scripts/plot_frontier.py "$OUT/$m/$l"
  done; done
fi

# ---- 4. Benchmarks with lm-evaluation-harness (unedited = FOXP2_GAMMA=0, same weights) -
#      target-language tasks + English tasks (collateral damage on English)
if has lmeval; then
  for m in "${MS[@]}"; do for l in "${LS[@]}"; do
    ck="$CK/$m-foxp2-$l"; [[ -d "$ck" ]] || continue
    for kind in target english; do
      for g in 0 1; do
        dest="$RES/$m/$l/$kind/gamma$g"
        if compgen -G "$dest/**/results_*.json" >/dev/null 2>&1 || \
           [[ -n "$(find -L "$dest" -name 'results_*.json' 2>/dev/null | head -1)" ]]; then
          echo "[resume] $dest done, skipping"; continue
        fi
        # English, unedited: identical for every language of a model -> run once, link the rest
        shared="$RES/$m/_english_gamma0"
        if [[ "$kind" == english && "$g" == 0 && -d "$shared" ]]; then
          mkdir -p "$(dirname "$dest")"; ln -sfn "$(realpath "$shared")" "$dest"; continue
        fi
        [[ "$kind" == english && "$g" == 0 ]] && dest="$shared"
        ll=$(python scripts/lmeval_tasks.py "$l" "$kind" ll)
        gen=$(python scripts/lmeval_tasks.py "$l" "$kind" gen)
        if [[ -n "$ll" ]]; then
          log "lm-eval $m $l $kind gamma=$g (log-likelihood: $ll)"
          FOXP2_GAMMA=$g FOXP2_MODE=all lm_eval --model hf \
            --model_args "pretrained=$ck,trust_remote_code=True,dtype=bfloat16" \
            --tasks "$ll" --batch_size "$LMBS" $LIMITFLAG --output_path "$dest/ll"
        fi
        if [[ -n "$gen" ]]; then
          log "lm-eval $m $l $kind gamma=$g (generative: $gen)"
          FOXP2_GAMMA=$g lm_eval --model hf \
            --model_args "pretrained=$ck,trust_remote_code=True,dtype=bfloat16" \
            --tasks "$gen" --batch_size "$LMBS" $LIMITFLAG --output_path "$dest/gen"
        fi
        if [[ "$dest" == "$shared" ]]; then
          mkdir -p "$RES/$m/$l/$kind"; ln -sfn "$(realpath "$shared")" "$RES/$m/$l/$kind/gamma0"
        fi
      done
    done
  done; done
fi

# ---- 5. All paper figures + main table ------------------------------------------------
if has figures; then
  log "figures"
  python scripts/make_figures.py --runs "$OUT" --ckpts "$CK" --lmeval "$RES" --out "$FIG"
fi
log "done"
