#!/usr/bin/env bash
# Full grid: 4 models x 11 target languages, then evaluate every exported checkpoint.
# Prereqs: huggingface-cli login; accept the FLORES+, Llama-3.1 and Gemma-2 licences on HF.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs}; CK=${CK:-checkpoints}

python scripts/run_pipeline.py --models all --langs all --out "$OUT" --export_dir "$CK" "$@"

for m in llama31-8b gemma2-9b gemma2-2b qwen3-8b; do
  for l in hi zh bn te es sw am ar mr de fr; do
    python scripts/evaluate.py --ckpt "$CK/$m-foxp2-$l" --model "$m" --lang "$l" --translation
  done
done
