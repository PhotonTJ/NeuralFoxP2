"""Export a FOXP2 artifact as a steered Hugging Face checkpoint.

Layout of an exported checkpoint:
    config.json                    base config + auto_map + `foxp2` block
    model-*.safetensors            symlinks to the base model's shards (or copies with --copy)
    foxp2.safetensors              the steerer buffers (a few MB)
    model.safetensors.index.json   base weight map + foxp2 tensors
    modeling_foxp2.py, foxp2_steer.py
    tokenizer files, generation_config.json, foxp2_manifest.json, README.md

It loads with stock `transformers` (trust_remote_code=True), so lm-evaluation-harness and any
HF-based harness can evaluate it directly.  vLLM / TGI do not run arbitrary forward hooks and
are not supported.
"""
from __future__ import annotations

import json
import os
import shutil

import torch

ARCH_CLASS = {"llama": "FOXP2LlamaForCausalLM", "gemma2": "FOXP2Gemma2ForCausalLM",
              "qwen3": "FOXP2Qwen3ForCausalLM"}
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")


def base_snapshot(base: str) -> str:
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    return snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                   "*.jinja", "tokenizer*"])


def export_checkpoint(artifact: dict, base: str, out_dir: str, arch: str, copy: bool = False,
                      extra_manifest: dict | None = None, token_sets: dict | None = None) -> str:
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = base_snapshot(base)
    os.makedirs(out_dir, exist_ok=True)
    cls = ARCH_CLASS[arch]

    # ---- weights ----
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
        weight_map = dict(index["weight_map"])
        rename = {f: f for f in set(weight_map.values())}
    else:
        single = os.path.join(src, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"no safetensors weights in {src}")
        with safe_open(single, "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
        # must not be called model.safetensors, otherwise the index (and foxp2 tensors) is ignored
        rename = {"model.safetensors": "model-base.safetensors"}
        weight_map = {k: rename[v] for k, v in weight_map.items()}
        index = {"metadata": {}}
    for f_src, f_dst in rename.items():
        a, b = os.path.realpath(os.path.join(src, f_src)), os.path.join(out_dir, f_dst)
        if os.path.lexists(b):
            os.remove(b)
        shutil.copy2(a, b) if copy else os.symlink(a, b)

    sd = {f"foxp2_steer.{k}": v.detach().float().contiguous()
          for k, v in artifact["state_dict"].items()}
    save_file(sd, os.path.join(out_dir, "foxp2.safetensors"), metadata={"format": "pt"})
    weight_map.update({k: "foxp2.safetensors" for k in sd})
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {**index.get("metadata", {}), "total_size": total},
               "weight_map": weight_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)

    # ---- config ----
    cfg = json.load(open(os.path.join(src, "config.json")))
    fc = dict(artifact["config"])
    cfg.update({"architectures": [cls],
                "auto_map": {"AutoModelForCausalLM": f"modeling_foxp2.{cls}"},
                "foxp2": fc, "foxp2_gamma": fc.get("gamma", 1.0), "foxp2_mode": fc["mode"],
                "foxp2_k_decode": fc["k_decode"], "foxp2_use_gate": fc["use_gate"]})
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    # ---- tokenizer and other small files ----
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if (os.path.isfile(p) and not f.endswith(WEIGHT_EXT) and f not in
                ("config.json", "model.safetensors.index.json")):
            shutil.copy2(os.path.realpath(p), os.path.join(out_dir, f))
    shutil.copy2(os.path.join(HERE, "hf_remote", "modeling_foxp2.py"), out_dir)
    shutil.copy2(os.path.join(HERE, "foxp2_steer.py"), out_dir)
    if token_sets is not None:
        json.dump(token_sets, open(os.path.join(out_dir, "foxp2_token_sets.json"), "w"))

    manifest = {"base": base, "base_snapshot": src, "class": cls, "foxp2": fc,
                "variants": {k: {kk: vv for kk, vv in v.items() if kk != "dev"}
                             for k, v in artifact.get("variants", {}).items()},
                "gate": artifact.get("gate", {}), "protocol": artifact.get("protocol", {}),
                "weights": "copied" if copy else "symlinked to base snapshot",
                **(extra_manifest or {})}
    json.dump(manifest, open(os.path.join(out_dir, "foxp2_manifest.json"), "w"), indent=1,
              default=str)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(f"""# FOXP2-steered {base} (target: {fc['target']})

Base weights are unchanged; a forward hook adds the FOXP2 edit at layers {fc['layers'][0]}-{fc['layers'][-1]}.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(PATH)
```

Overrides: `foxp2_gamma`, `foxp2_mode` ("decode_window" for generation, "all" for log-likelihood
tasks), `foxp2_k_decode`, `foxp2_use_gate` as `from_pretrained` kwargs, or env vars
FOXP2_GAMMA / FOXP2_MODE / FOXP2_KDECODE / FOXP2_GATE.  `FOXP2_GAMMA=0` is the unedited model.

Operating point: lambda={fc['lam']}, beta={fc['beta']}, rank={fc['rank']}, gate={fc['use_gate']}.
Inside guardrails {fc.get('guardrail_eps')}: **{fc.get('feasible', 'unknown')}**.
""")
    return out_dir


def export_from_artifact_file(artifact_path, base, out_dir, arch, copy=False):
    return export_checkpoint(torch.load(artifact_path, weights_only=False), base, out_dir, arch, copy)
