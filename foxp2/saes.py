"""Loaders for the three frozen public SAE suites, with a reconstruction self-check.

Each loader returns an `SAE` with the convention

    pre(x)  = (x * in_scale - in_bias) @ W_enc + b_enc          W_enc: [d, m]
    z(x)    = JumpReLU_thr(pre)   or   TopK(pre)
    x_hat   = (z @ W_dec + b_dec) * out_scale                     W_dec: [m, d]

and the *linear* decoder used for every edit is  D_lin(dz) = (dz @ W_dec) * out_scale
(no b_dec: the edit is a difference of decodes and is added on top of h, never substituted).

Llama Scope stores a dataset-wise activation norm and a scalar JumpReLU threshold; whether
the threshold acts on the raw pre-activation or on pre * ||W_dec_j|| is not documented, so
`self_check` tries both conventions on real activations and keeps the lower-FVU one.  The
measured FVU is written to the manifest (review item: report the base-SAE / IT-model
mismatch cost instead of asserting it is harmless).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SAE:
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    act: str                          # "jumprelu" | "topk"
    threshold: torch.Tensor | None = None
    topk: int | None = None
    in_scale: float = 1.0
    out_scale: float = 1.0
    in_bias: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def d(self):
        return self.W_enc.shape[0]

    @property
    def m(self):
        return self.W_enc.shape[1]

    def to(self, device, dtype=torch.float32):
        for k in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold", "in_bias"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device=device, dtype=dtype))
        return self

    def sae_in(self, x):
        x = x.to(self.W_enc.dtype) * self.in_scale
        return x - self.in_bias if self.in_bias is not None else x

    def pre(self, x):
        return self.sae_in(x) @ self.W_enc + self.b_enc

    def activate(self, pre):
        if self.act == "jumprelu":
            return pre * (pre > self.threshold)
        vals, idx = pre.topk(self.topk, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def encode(self, x):
        return self.activate(self.pre(x))

    def decode(self, z):
        return (z @ self.W_dec + self.b_dec) * self.out_scale

    def dec_rows(self, idx):
        """Linear decoder rows (already in residual units) for features `idx`: [k, d]."""
        return self.W_dec[idx] * self.out_scale

    def fvu(self, x):
        x = x.float().to(self.W_enc.device)
        r = self.decode(self.encode(x))
        return float(((x - r) ** 2).sum() / ((x - x.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-8))

    def topk_threshold(self, x):
        """Median k-th largest pre-activation; used to approximate TopK on a feature subset."""
        kth = self.pre(x).topk(self.topk, dim=-1).values[..., -1]
        return float(kth.median())


# --------------------------------------------------------------------------------------
def _dl(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename)


def load_gemma_scope(repo: str, layer: int, width: str = "width_16k") -> SAE:
    """Canonical = the average_l0 closest to 100 (SAELens' canonical rule)."""
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo) if f.startswith(f"layer_{layer}/{width}/")
             and f.endswith("params.npz")]
    if not files:
        raise FileNotFoundError(f"no Gemma Scope params for layer {layer} {width} in {repo}")
    l0 = lambda f: int(re.search(r"average_l0_(\d+)", f).group(1))
    f = min(files, key=lambda f: abs(l0(f) - 100))
    p = np.load(_dl(repo, f))
    t = lambda k: torch.from_numpy(np.asarray(p[k])).float()
    return SAE(W_enc=t("W_enc"), b_enc=t("b_enc"), W_dec=t("W_dec"), b_dec=t("b_dec"),
               act="jumprelu", threshold=t("threshold"),
               meta={"family": "gemma_scope", "file": f, "average_l0": l0(f)})


def load_llama_scope(repo: str, layer: int, expansion: str = "8x") -> SAE:
    from safetensors.torch import load_file
    folder = f"Llama3_1-8B-Base-L{layer}R-{expansion}"
    hp = json.load(open(_dl(repo, f"{folder}/hyperparams.json")))
    sd = load_file(_dl(repo, f"{folder}/checkpoints/final.safetensors"))
    d, m = hp["d_model"], hp["d_sae"]

    def find(*needles):
        for k, v in sd.items():
            if all(n in k for n in needles):
                return v.float()
        raise KeyError(f"{needles} not in {list(sd)}")

    W_enc, W_dec = find("encoder", "weight"), find("decoder", "weight")
    W_enc = W_enc if W_enc.shape == (d, m) else W_enc.T
    W_dec = W_dec if W_dec.shape == (m, d) else W_dec.T
    b_enc, b_dec = find("encoder", "bias"), find("decoder", "bias")
    norm = hp.get("dataset_average_activation_norm") or {"in": math.sqrt(d), "out": math.sqrt(d)}
    in_scale, out_scale = math.sqrt(d) / norm["in"], norm["out"] / math.sqrt(d)
    thr = float(hp.get("jump_relu_threshold", 0.0))
    in_bias = b_dec.clone() if hp.get("apply_decoder_bias_to_pre_encoder", False) else None
    assert hp["hook_point_in"] == f"blocks.{layer}.hook_resid_post", hp["hook_point_in"]
    sae = SAE(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, act="jumprelu",
              threshold=torch.full((m,), thr), in_scale=in_scale, out_scale=out_scale,
              in_bias=in_bias, meta={"family": "llama_scope", "folder": folder,
                                      "threshold": thr, "norm": norm})
    sae.meta["dec_norm"] = W_dec.norm(dim=1)
    return sae


def load_qwen_scope(repo: str, layer: int, topk: int = 50) -> SAE:
    sd = torch.load(_dl(repo, f"layer{layer}.sae.pt"), map_location="cpu")
    W_enc = sd["W_enc"].float()       # (m, d)
    W_dec = sd["W_dec"].float()       # (d, m)
    return SAE(W_enc=W_enc.T.contiguous(), b_enc=sd["b_enc"].float(),
               W_dec=W_dec.T.contiguous(), b_dec=sd["b_dec"].float(), act="topk", topk=topk,
               meta={"family": "qwen_scope", "file": f"layer{layer}.sae.pt"})


def load_sae(spec, layer: int) -> SAE:
    if spec.sae_family == "gemma_scope":
        return load_gemma_scope(spec.sae_repo, layer, spec.sae_width or "width_16k")
    if spec.sae_family == "llama_scope":
        return load_llama_scope(spec.sae_repo, layer)
    if spec.sae_family == "qwen_scope":
        return load_qwen_scope(spec.sae_repo, layer, spec.sae_topk or 50)
    raise ValueError(spec.sae_family)


def self_check(sae: SAE, x: torch.Tensor) -> dict:
    """Pick the encoding convention with the lowest FVU on real activations; report FVU."""
    report = {}
    if sae.meta.get("family") == "llama_scope" and "dec_norm" in sae.meta:
        base_thr = sae.threshold.clone()
        dn = sae.meta["dec_norm"].to(base_thr.device)
        report["fvu_thr_on_pre"] = sae.fvu(x)
        sae.threshold = base_thr / dn.clamp_min(1e-6)      # pre * ||d_j|| > thr
        report["fvu_thr_on_pre_x_decnorm"] = sae.fvu(x)
        if report["fvu_thr_on_pre"] <= report["fvu_thr_on_pre_x_decnorm"]:
            sae.threshold = base_thr
            report["convention"] = "thr_on_pre"
        else:
            report["convention"] = "thr_on_pre_x_decnorm"
        sae.meta.pop("dec_norm", None)
    report["fvu"] = sae.fvu(x)
    if report["fvu"] > 0.5:
        print(f"[warn] SAE FVU {report['fvu']:.3f} on this checkpoint's activations: the base "
              f"dictionary transfers poorly here; interpret features at this layer with care.")
    return report
