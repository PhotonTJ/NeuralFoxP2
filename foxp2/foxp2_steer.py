"""FOXP2 runtime steerer.  Torch-only, self-contained: this file is copied verbatim into every
exported checkpoint and loaded by `modeling_foxp2.py` through `trust_remote_code`.

Edit applied to the residual stream h at the output of every layer l in the window W:

    h <- h + gamma * g(x) * ( v_pos[l]  -  beta * sum_{j in N_en(l)} z_j(h) * d_j )

  v_pos[l]  = D_lin( lambda_l * P_l mu_l )      target promotion (Stage II subspace, fixed)
  z_j(h)    = JumpReLU(pre_j(h))                activation of English-promoting feature j
  d_j       = linear decoder row of feature j   (so beta=1 removes it completely)
  g(x)      = 0 if the instruction gate fires on the prompt (explicit language request), else 1

Positions edited ("decode_window" mode): the prompt-final position (which produces the first
response token) and the next `k_decode` decoding steps.  "all" mode edits every position and
is the right mode for log-likelihood (multiple-choice) benchmarks.

Runtime overrides (env vars win over config): FOXP2_GAMMA, FOXP2_MODE, FOXP2_KDECODE,
FOXP2_GATE (0/1).  gamma = 0 reproduces the unedited model bit for bit.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def _seq_info(args, kwargs):
    x = kwargs.get("input_ids")
    if x is None and args:
        x = args[0]
    if x is None:
        x = kwargs.get("inputs_embeds")
    seq_len = int(x.shape[1]) if x is not None and x.dim() >= 2 else 1
    pkv = kwargs.get("past_key_values")
    past = 0
    if pkv is not None:
        try:
            past = int(pkv.get_seq_length())
        except Exception:
            try:
                past = int(pkv[0][0].shape[-2])
            except Exception:
                past = 0
    return seq_len, past


class FOXP2Steerer(nn.Module):
    def __init__(self, layers, d_model: int, k_en: int, mode: str = "decode_window",
                 k_decode: int = 8, beta: float = 1.0, gamma: float = 1.0, use_gate: bool = True):
        super().__init__()
        self.layers = [int(l) for l in layers]
        n, k_en = len(self.layers), max(int(k_en), 1)
        self.register_buffer("v_pos", torch.zeros(n, d_model))
        self.register_buffer("enc_W", torch.zeros(n, d_model, k_en))
        self.register_buffer("enc_b", torch.zeros(n, k_en))
        self.register_buffer("enc_thr", torch.zeros(n, k_en))
        self.register_buffer("enc_in_scale", torch.ones(n))
        self.register_buffer("enc_in_bias", torch.zeros(n, d_model))
        self.register_buffer("dec_W", torch.zeros(n, k_en, d_model))
        self.register_buffer("en_mask", torch.zeros(n, k_en))
        self.register_buffer("gate_w", torch.zeros(d_model))
        self.register_buffer("gate_b", torch.zeros(1))
        self.mode, self.k_decode = mode, int(k_decode)
        self.beta, self.gamma, self.use_gate = float(beta), float(gamma), bool(use_gate)
        self._handles = []
        self._prefill, self._step, self._gate = True, 0, None
        self._cache = {}
        self.force_last = None   # analysis only: edit the last n positions of a prefill

    # ---------------------------------------------------------------- state management
    def invalidate(self):
        self._cache = {}

    def _params(self, device):
        key = str(device)
        if key not in self._cache:
            names = ["v_pos", "enc_W", "enc_b", "enc_thr", "enc_in_scale", "enc_in_bias",
                     "dec_W", "en_mask", "gate_w", "gate_b"]
            self._cache[key] = {k: getattr(self, k).detach().to(device=device, dtype=torch.float32)
                                for k in names}
        return self._cache[key]

    def _settings(self):
        e = os.environ
        gamma = float(e.get("FOXP2_GAMMA", self.gamma))
        mode = e.get("FOXP2_MODE", self.mode)
        k_dec = int(e.get("FOXP2_KDECODE", self.k_decode))
        gate = e.get("FOXP2_GATE")
        use_gate = self.use_gate if gate is None else gate not in ("0", "false", "False")
        return gamma, mode, k_dec, use_gate

    def begin_forward(self, seq_len: int, past: int):
        if past == 0 or seq_len > 1:
            self._prefill, self._step, self._gate = True, 0, None
        else:
            self._prefill = False
            self._step += 1

    # ---------------------------------------------------------------- the edit
    def edit(self, i: int, h: torch.Tensor) -> torch.Tensor:
        gamma, mode, k_dec, use_gate = self._settings()
        if gamma == 0.0:
            return h
        B, T, _ = h.shape
        if mode == "all":
            sl = slice(0, T)
        elif self._prefill:
            n = self.force_last or 1
            sl = slice(max(T - n, 0), T)
        elif self._step <= k_dec:
            sl = slice(0, T)
        else:
            return h
        P = self._params(h.device)
        if i == 0 and (self._prefill or self._gate is None):
            if use_gate and bool(P["gate_w"].abs().sum() > 0):
                logit = h[:, -1].float() @ P["gate_w"] + P["gate_b"]
                self._gate = (logit < 0).float()          # 1 = weak prompt -> steer
            else:
                self._gate = torch.ones(B, device=h.device)
        g = self._gate if (self._gate is not None and self._gate.shape[0] == B) \
            else torch.ones(B, device=h.device)
        x = h[:, sl].float()
        pre = (x * P["enc_in_scale"][i] - P["enc_in_bias"][i]) @ P["enc_W"][i] + P["enc_b"][i]
        z = pre * (pre > P["enc_thr"][i]).float() * P["en_mask"][i]
        delta = P["v_pos"][i] - self.beta * (z @ P["dec_W"][i])
        delta = gamma * g[:, None, None] * delta
        out = h.clone()
        out[:, sl] = (x + delta).to(h.dtype)
        return out

    # ---------------------------------------------------------------- hooks
    def attach(self, model: nn.Module, decoder_layers) -> "FOXP2Steerer":
        self.detach()

        def pre_hook(module, args, kwargs):
            self.begin_forward(*_seq_info(args, kwargs))
            return None

        self._handles.append(model.register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, l in enumerate(self.layers):
            def hook(module, args, output, i=i):
                h = output[0] if isinstance(output, tuple) else output
                h2 = self.edit(i, h)
                if h2 is h:
                    return None
                return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
            self._handles.append(decoder_layers[l].register_forward_hook(hook))
        return self

    def detach(self):
        for hd in self._handles:
            hd.remove()
        self._handles = []

    def summary(self) -> dict:
        return {"layers": self.layers, "mode": self.mode, "k_decode": self.k_decode,
                "beta": self.beta, "gamma": self.gamma, "use_gate": self.use_gate,
                "v_pos_norm": [round(float(v), 4) for v in self.v_pos.float().norm(dim=-1)],
                "n_en_features": [int(m) for m in self.en_mask.sum(-1)],
                "gate_active": bool(self.gate_w.abs().sum() > 0)}
