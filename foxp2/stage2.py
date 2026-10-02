"""Stage II: low-rank steering directions, diagnostics with nulls, and the intervention window.

Changes vs. the submitted paper (review fixes):
  * The language-shift matrix is analysed both UNCENTERED (paper) and CENTERED.  We report
    the mean-shift share of ||dZ||_F^2, ||P mu|| / ||mu||, the centred spectrum and its
    effective rank, so a reader can see whether "low rank" is more than the mean direction.
  * Three nulls run through the identical pipeline: (i) independent column shuffle of the
    centred matrix (keeps every feature's marginal, destroys cross-feature correlation;
    sign-flipping rows would be an orthogonal map and leave the spectrum unchanged),
    (ii) a random support of equal size, (iii) a content-attribute contrast inside English
    (wikinews vs wikivoyage).  Language should look lower-rank than all three.
  * The window is chosen from a per-layer CAUSAL curve (single-layer edit, teacher-forced
    dev gain), restricted to bootstrap-stable layers, and the best candidate windows are then
    scored with the full objective J(W) (gain, leakage, off-decision KL, in-language NLL,
    instability), with every alpha weight and width bound logged.
  * The rank is selected by dev steering efficacy among {1,2,4,8, spectral, mean}; the
    spectral choice is reported next to it so agreement can be checked.
  * Per-layer strength is lambda / |W| (an edit applied at |W| layers compounds through the
    residual stream; dividing keeps lambda comparable across window widths).
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch

from .config import LANG_NAME
from .data import weak_prompts
from .metrics import (committed_contexts, compare_early, early_defaultness, kl, last_logprobs,
                      response_nll, tf_masses, tf_sequences)
from .model_io import add_vectors, chat_ids


# --------------------------------------------------------------------------------------
def spectrum_stats(dZ: torch.Tensor, max_rank: int = 16) -> dict:
    s = torch.linalg.svdvals(dZ.double())
    s = s[s > 1e-12]
    if len(s) == 0:
        return {"r_eff": 0.0, "eigengap_idx": 0, "eigengap": 0.0, "top1_var": 0.0, "spectrum": []}
    p = s / s.sum()
    r_eff = float(torch.exp(-(p * p.log()).sum()))
    m = min(max_rank, len(s) - 1)
    gaps = (s[:m] / (s[1:m + 1] + 1e-12)) if m > 0 else torch.tensor([1.0])
    gi = int(gaps.argmax()) + 1
    var = (s ** 2) / (s ** 2).sum()
    return {"r_eff": r_eff, "eigengap_idx": gi, "eigengap": float(gaps.max()),
            "top1_var": float(var[0]), "top4_var": float(var[:4].sum()),
            "spectrum": [float(x) for x in s[:max_rank]]}


def subspace(dZ: torch.Tensor, r: int, centered: bool = False) -> torch.Tensor:
    X = dZ - dZ.mean(0, keepdim=True) if centered else dZ
    _, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return Vh[:r].T.float()        # [K, r]


def bootstrap_stability(dZ: torch.Tensor, r: int, B: int, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    V0 = subspace(dZ, r).double()
    sins = []
    for _ in range(B):
        ix = torch.randint(0, dZ.shape[0], (dZ.shape[0],), generator=g)
        Vb = subspace(dZ[ix], r).double()
        cos = torch.linalg.svdvals(V0.T @ Vb).clamp(0, 1)
        sins.append(float(torch.sqrt(1 - cos ** 2).mean()))
    return 1.0 - float(np.mean(sins))


def layer_analysis(L: dict, domains: list[str], cfg) -> dict:
    nt = L["n_tgt"]
    dZ = (L["Zt"] - L["Ze"])[:, :nt].float()
    mu = dZ.mean(0)
    st = spectrum_stats(dZ, cfg.max_rank)
    r_spec = max(1, min(math.ceil(st["r_eff"]), st["eigengap_idx"], nt))
    V = subspace(dZ, r_spec)
    Pmu = V @ (V.T @ mu)
    centred = spectrum_stats(dZ - mu, cfg.max_rank)
    g = torch.Generator().manual_seed(cfg.seed)
    C = dZ - mu
    shuf = torch.stack([C[torch.randperm(C.shape[0], generator=g), j] for j in range(C.shape[1])], 1)
    null_shuf = spectrum_stats(shuf, cfg.max_rank)
    R = L["rand"]
    null_rand = spectrum_stats((R["Zt"] - R["Ze"]).float(), cfg.max_rank)
    news = [i for i, d in enumerate(domains) if d == "wikinews"]
    voy = [i for i, d in enumerate(domains) if d == "wikivoyage"]
    n = min(len(news), len(voy))
    null_dom = (spectrum_stats((L["Ze"][news[:n]] - L["Ze"][voy[:n]])[:, :nt].float(), cfg.max_rank)
                if n >= 8 else None)
    stab = bootstrap_stability(dZ, r_spec, cfg.n_boot, cfg.seed)
    return {
        "r_spec": r_spec, "spectral": st, "centred": centred,
        "mean_share": float(dZ.shape[0] * (mu ** 2).sum() / (dZ ** 2).sum().clamp_min(1e-12)),
        "Pmu_over_mu": float(Pmu.norm() / mu.norm().clamp_min(1e-12)),
        "null_colshuffle": null_shuf, "null_random_support": null_rand, "null_domain": null_dom,
        "stability": stab, "mass": st["top1_var"],
    }


def positive_direction(L: dict, r, subspace_mode: str) -> torch.Tensor:
    """Feature-space target-promotion vector (over N_tgt) -> residual vector [d]."""
    nt = L["n_tgt"]
    dZ = (L["Zt"] - L["Ze"])[:, :nt].float()
    mu = dZ.mean(0)
    if subspace_mode == "mean" or r == "mean":
        dz = mu
    else:
        V = subspace(dZ, max(1, min(int(r), nt)))
        dz = V @ (V.T @ mu)
    return dz @ L["dec"][:nt].float()


# --------------------------------------------------------------------------------------
class DevBench:
    """Everything needed to score an edit on D_dev (prompts built once, baselines cached)."""

    def __init__(self, model, tok, spec, ldata, ts, cfg, others, split: str = "dev",
                 n: int | None = None):
        self.model, self.tok, self.ts, self.cfg, self.others = model, tok, ts, cfg, others
        self.target, self.split = ldata.target, split
        pool = ldata.dev if split == "dev" else ldata.eval
        units = pool[: (n or cfg.n_dev_prompts)]
        self.spec = spec
        self.prompt_texts = weak_prompts(units, split)
        self.prompts = [chat_ids(tok, p, spec.chat_kwargs) for p in self.prompt_texts]
        self.tf_seqs = tf_sequences(model, tok, self.prompts, cfg.T, cfg.batch_size)
        self.committed = committed_contexts(tok, self.prompts, [u.tgt for u in units])
        self.ref_tgt = [u.tgt for u in units]
        self.base_tf, _ = tf_masses(model, tok, self.tf_seqs, ts, self.target, cfg.T, cfg.batch_size)
        self.base_early = early_defaultness(model, tok, self.prompts, ts, self.target, cfg.T,
                                           batch_size=cfg.batch_size)
        self.base_lp = last_logprobs(model, tok, self.committed, cfg.batch_size)
        self.base_nll = response_nll(model, tok, self.prompts, self.ref_tgt, batch_size=cfg.batch_size)
        self.k_util = max(cfg.k_decode, cfg.util_skip + 2)
        self.base_nll_dep = response_nll(model, tok, self.prompts, self.ref_tgt, skip=cfg.util_skip,
                                         max_resp=self.k_util, batch_size=cfg.batch_size)
        self.prompt_ref = self.prompt_reference()
        fixed = {"leak": cfg.eps_leak, "kl": cfg.eps_kl, "util": cfg.eps_util}
        if cfg.guardrail == "prompt":
            pr = self.prompt_ref
            self.eps = {"leak": max(fixed["leak"], pr["leak_max"]),
                        "kl": max(fixed["kl"], pr["kl_offdecision"]),
                        "util": max(fixed["util"], pr["util_nll"])}
        else:
            self.eps = fixed
        if split != "dev":
            return
        print(f"[guardrail] prompt-only reference ('Answer in {LANG_NAME[self.target]}.'): "
              f"gain={pr_fmt(self.prompt_ref, 'gain_dM')} DEFAULT+={pr_fmt(self.prompt_ref, 'gain_DEFAULT')} "
              f"leak={pr_fmt(self.prompt_ref, 'leak_max')} kl={pr_fmt(self.prompt_ref, 'kl_offdecision')} "
              f"util={pr_fmt(self.prompt_ref, 'util_nll')}")
        print(f"[guardrail] mode={cfg.guardrail} eps={ {k: round(v, 3) for k, v in self.eps.items()} }")

    def prompt_reference(self) -> dict:
        """What the explicit instruction itself costs, measured exactly like an edit:
        gain/leak on the same dev prompts, KL and in-language NLL on the same committed
        target-language contexts, with the instruction added to the prompt."""
        L = LANG_NAME[self.target]
        instr = [chat_ids(self.tok, f"{p} Answer in {L}.", self.spec.chat_kwargs)
                 for p in self.prompt_texts]
        e = early_defaultness(self.model, self.tok, instr, self.ts, self.target, self.cfg.T,
                              batch_size=self.cfg.batch_size)
        r = compare_early(e, self.base_early, self.target, self.others)
        r["kl_offdecision"] = kl(self.base_lp, last_logprobs(
            self.model, self.tok, committed_contexts(self.tok, instr, self.ref_tgt), self.cfg.batch_size))
        r["util_nll"] = response_nll(self.model, self.tok, instr, self.ref_tgt, skip=self.cfg.util_skip,
                                     max_resp=self.k_util, batch_size=self.cfg.batch_size) - self.base_nll_dep
        return r

    def tf_gain(self) -> float:
        dM, _ = tf_masses(self.model, self.tok, self.tf_seqs, self.ts, self.target, self.cfg.T,
                          self.cfg.batch_size)
        return float(dM.mean() - self.base_tf.mean())

    def full(self, steer=None) -> dict:
        """Closed-loop gain/leak (current hooks) + off-decision KL + in-language NLL drift.

        util_nll (guardrail): NLL of reference target-language tokens util_skip..k_decode with
        the edit on exactly the positions it touches when deployed (prompt-final + first
        k_decode steps).  util_nll_all (reported only): the same responses with every position
        edited, a stress test that over-states the cost of decode_window steering."""
        e = early_defaultness(self.model, self.tok, self.prompts, self.ts, self.target, self.cfg.T,
                              batch_size=self.cfg.batch_size)
        r = compare_early(e, self.base_early, self.target, self.others)
        r["kl_offdecision"] = kl(self.base_lp, last_logprobs(self.model, self.tok, self.committed,
                                                             self.cfg.batch_size))
        if steer is not None and steer.mode != "all":
            steer.force_last = self.k_util + 1
        r["util_nll"] = response_nll(self.model, self.tok, self.prompts, self.ref_tgt,
                                     skip=self.cfg.util_skip, max_resp=self.k_util,
                                     batch_size=self.cfg.batch_size) - self.base_nll_dep
        mode = None
        if steer is not None:
            steer.force_last = None
            mode, steer.mode = steer.mode, "all"
        r["util_nll_all"] = response_nll(self.model, self.tok, self.prompts, self.ref_tgt,
                                         batch_size=self.cfg.batch_size) - self.base_nll
        if steer is not None:
            steer.mode = mode
        r["feasible"] = (r["leak_max"] <= self.eps["leak"] and r["kl_offdecision"] <= self.eps["kl"]
                         and r["util_nll"] <= self.eps["util"])
        r["excess"] = (max(0.0, r["leak_max"] - self.eps["leak"]) + max(0.0, r["kl_offdecision"] - self.eps["kl"])
                       + max(0.0, r["util_nll"] - self.eps["util"]))
        return r


def pr_fmt(r, k):
    return f"{r[k]:.3f}"


def steer_score(r: dict) -> float:
    """dM alone can be gamed (over-steering emits target-script garbage); DEFAULT needs the
    prefix LID to agree, so we average the two."""
    return 0.5 * (r["gain_dM"] + r["gain_DEFAULT"])


def objective(r: dict, stab_pen: float, cfg, eps: dict | None = None) -> float:
    """Gain minus penalties.  With `eps`, only the part of each cost above its guardrail is
    penalised (so windows/points are compared on what they cost beyond what is allowed)."""
    eps = eps or {"leak": 0.0, "kl": 0.0, "util": 0.0}
    return (steer_score(r) - cfg.alpha_leak * max(0.0, r["leak_max"] - eps["leak"])
            - cfg.alpha_kl * max(0.0, r["kl_offdecision"] - eps["kl"])
            - cfg.alpha_util * max(0.0, r["util_nll"] - eps["util"]) - cfg.alpha_stab * stab_pen)


# --------------------------------------------------------------------------------------
def run_stage2(model, tok, spec, ldata, ts, cfg, s1, out_dir, bench: DevBench, make_steerer):
    """`make_steerer(window, r, lam, beta, diag)` builds + attaches a FOXP2Steerer (Stage III code)."""
    layers = sorted(s1["layers"])
    domains = s1["meta"]["domains"]
    diag = {l: layer_analysis(s1["layers"][l], domains, cfg) for l in layers}
    for l in layers:
        d = diag[l]
        print(f"[stage2] layer {l}: r_spec={d['r_spec']} r_eff={d['spectral']['r_eff']:.2f} "
              f"centred_r_eff={d['centred']['r_eff']:.2f} mean_share={d['mean_share']:.2f} "
              f"Pmu/mu={d['Pmu_over_mu']:.3f} stab={d['stability']:.2f}")
    # How much of the full residual language shift (diff-in-means, CAA) the SAE support carries.
    for l in layers:
        L = s1["layers"][l]
        v, c = positive_direction(L, "mean", "mean").float(), L["caa"].float()
        diag[l]["support_vs_caa_norm"] = float(v.norm() / c.norm().clamp_min(1e-8))
        diag[l]["support_vs_caa_cos"] = float(torch.nn.functional.cosine_similarity(v, c, dim=0))
    print("[stage2] |support shift| / |residual shift|:",
          {l: round(diag[l]["support_vs_caa_norm"], 3) for l in layers})
    print("[stage2] cos(support shift, residual shift):",
          {l: round(diag[l]["support_vs_caa_cos"], 3) for l in layers})
    med = lambda f: round(float(np.median([f(diag[l]) for l in layers if f(diag[l]) is not None])), 2)
    print("[stage2] median r_eff  language(uncentred)={} language(centred)={} null:colshuffle(centred)={} "
          "null:random_support={} null:domain={}".format(
              med(lambda d: d["spectral"]["r_eff"]), med(lambda d: d["centred"]["r_eff"]),
              med(lambda d: d["null_colshuffle"]["r_eff"]), med(lambda d: d["null_random_support"]["r_eff"]),
              med(lambda d: d["null_domain"]["r_eff"] if d["null_domain"] else None)))
    print("[stage2] median top-1 variance share  language={} language(centred)={} null:colshuffle={} "
          "null:domain={}".format(
        med(lambda d: d["spectral"]["top1_var"]), med(lambda d: d["centred"]["top1_var"]),
        med(lambda d: d["null_colshuffle"]["top1_var"]),
        med(lambda d: d["null_domain"]["top1_var"] if d["null_domain"] else None)))

    # ---- per-layer causal curve: single-layer positive edit, teacher-forced dev gain ----
    curve = {}
    for l in layers:
        v = positive_direction(s1["layers"][l], diag[l]["r_spec"], cfg.subspace)
        with add_vectors(model, {l: v}, last_k=cfg.T):
            curve[l] = bench.tf_gain()
    print("[stage2] causal curve:", {l: round(c, 3) for l, c in curve.items()})

    # ---- candidate windows: contiguous, stable, ranked by summed curve ----
    eligible = [l for l in layers if diag[l]["stability"] >= cfg.stab_min]
    cands = []
    for a in eligible:
        for w in range(cfg.min_width, cfg.max_width + 1):
            W = list(range(a, a + w))
            if all(l in eligible for l in W):
                cands.append((sum(curve[l] for l in W) / math.sqrt(w), W))
    # depth-diverse candidates: the single-layer curve favours the last layers (they act on the
    # logits directly), so take the best windows starting in each third of the network.
    cands.sort(key=lambda x: -x[0])
    n_layers = spec.n_layers
    per_third = max(1, math.ceil(cfg.n_window_candidates / 3))
    seen, top = set(), []
    for third in range(3):
        lo, hi = third * n_layers / 3, (third + 1) * n_layers / 3
        k = 0
        for _, W in cands:
            if lo <= W[0] < hi and tuple(W) not in seen and k < per_third:
                seen.add(tuple(W))
                top.append(W)
                k += 1
    if not top:
        raise RuntimeError("no bootstrap-stable contiguous window; lower stab_min or min_width")

    # each window is scored at ITS best strength: early windows need more lam than late ones,
    # so a single shared lam systematically favours late windows.
    scored = []
    for W in top:
        pen = sum(1 - diag[l]["stability"] for l in W)
        best_w = None
        for lam in cfg.window_lam_grid:
            st = make_steerer(W, "spectral", lam, cfg.window_beta, diag)
            r = bench.full(st)
            st.detach()
            J = objective(r, pen, cfg, bench.eps)
            print(f"[stage2] W={W[0]}-{W[-1]} lam={lam} J={J:.3f} gain={r['gain_dM']:.3f} "
                  f"DEFAULT+={r['gain_DEFAULT']:.3f} leak={r['leak_max']:.3f} "
                  f"kl={r['kl_offdecision']:.3f} util={r['util_nll']:.3f} feas={r['feasible']}")
            if best_w is None or J > best_w["J"]:
                best_w = {"window": W, "lam": lam, "J": J, **r}
        scored.append(best_w)
    best = max(scored, key=lambda x: x["J"])
    W = best["window"]
    print(f"[stage2] selected window {W[0]}-{W[-1]} (best lam {best['lam']}, J={best['J']:.3f})")

    # ---- rank by efficacy (reported next to the spectral choice) ----
    rank_scores = {}
    for r in list(cfg.rank_candidates) + ["spectral", "mean"]:
        st = make_steerer(W, r, best["lam"], 0.0, diag)
        rank_scores[str(r)] = bench.tf_gain()
        st.detach()
    best_rank = max(rank_scores, key=rank_scores.get)
    print(f"[stage2] window {W}, rank efficacy {rank_scores} -> {best_rank}")

    out = {"window": W, "rank": best_rank, "rank_scores": rank_scores, "curve": curve,
           "window_candidates": scored, "diagnostics": diag,
           "objective_weights": {k: getattr(cfg, k) for k in
                                 ("alpha_leak", "alpha_kl", "alpha_util", "alpha_stab",
                                  "min_width", "max_width", "stab_min", "window_lam_grid",
                                  "window_beta", "guardrail")},
           "guardrail_eps": bench.eps, "prompt_reference": bench.prompt_ref}
    json.dump(out, open(os.path.join(out_dir, "stage2.json"), "w"), indent=1, default=str)
    return out
