"""Stage III: signed sparse steering, instruction gate, guardrail-constrained tuning, ablations.

Changes vs. the submitted paper (review fixes):
  * English suppression is English-SPECIFIC: it fractionally removes the current activation of
    the English-promoting features N_en found in Stage I (beta=1 removes them entirely and
    can never push a feature below zero).  The paper's mu_en term subtracted along the mean
    of *all* weak-prompt activations, which is not an English direction.
  * The edit is position-limited to where commitment happens (prompt-final position + the
    first k_decode decoding steps), per the commitment curve; "all" mode is kept for
    log-likelihood benchmarks and as an ablation.
  * An explicit-instruction gate (linear probe on the first window layer, trained on D_dev,
    tested on held-out templates) switches steering off when the user names a language, so
    "answer in English" is respected.
  * One global (lambda, beta) pair, per-layer lambda/|W| (consistent with Table 24 instead of
    the contradictory per-layer grid), chosen as the max-gain FEASIBLE point under the frozen
    guardrails; the full grid is saved so the frontier is visible.
  * Every ablation (positive-only, negative-only, sparse-only, random support, out-of-window,
    residual diff-in-means/CAA) is tuned separately under the same guardrails, so none of the
    gaps is a strength mismatch.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from .data import explicit_prompts, weak_prompts
from .foxp2_steer import FOXP2Steerer
from .model_io import batches, capture, chat_ids, decoder_layers, greedy, input_device, left_pad
from .stage2 import DevBench, objective, positive_direction, steer_score


# --------------------------------------------------------------------------------------
def build_steerer(model, s1, diag, window, rank, lam, beta, cfg, d_model, variant="full",
                  gate=None, attach=True) -> FOXP2Steerer:
    layers = [l for l in window if l in s1["layers"]]
    if not layers:
        raise ValueError(f"window {window} has no Stage I support")
    lam_eff = lam / len(layers)
    if variant == "pos_only":
        beta = 0.0
    if variant == "neg_only":
        lam_eff = 0.0
    k_en = 1
    for l in layers:
        L = s1["layers"][l]
        k_en = max(k_en, len(L["rand"]["idx"]) if variant == "random" else L["n_en"])
    if gate is not None and gate.get("layer") != layers[0]:
        # the probe reads the residual of the layer it was trained on; elsewhere it is invalid
        print(f"[steer] gate trained at layer {gate.get('layer')} != window start {layers[0]}; "
              f"gate disabled for this configuration")
        gate = None
    st = FOXP2Steerer(layers, d_model, k_en, mode=cfg.mode, k_decode=cfg.k_decode, beta=beta,
                      gamma=1.0, use_gate=cfg.use_gate and gate is not None,
                      sink_factor=cfg.sink_factor)
    with torch.no_grad():
        for i, l in enumerate(layers):
            L = s1["layers"][l]
            st.enc_in_scale[i] = float(L["in_scale"])
            st.enc_in_bias[i] = L["in_bias"].float()
            if variant == "caa":
                st.v_pos[i] = lam_eff * L["caa"].float()
                continue
            if variant == "random":
                R = L["rand"]
                mu = (R["Zt"] - R["Ze"]).float().mean(0)
                st.v_pos[i] = lam_eff * (mu @ R["dec"].float())
                en = torch.nonzero(mu < 0).flatten()
                src, sl = R, en
            else:
                r = diag[l]["r_spec"] if rank == "spectral" else rank
                mode = "mean" if variant == "sparse_only" else cfg.subspace
                st.v_pos[i] = lam_eff * positive_direction(L, r, mode)
                nt, ne = L["n_tgt"], L["n_en"]
                src, sl = L, torch.arange(nt, nt + ne)
            n = len(sl)
            if n:
                st.enc_W[i, :, :n] = src["enc_W"][:, sl].float()
                st.enc_b[i, :n] = src["enc_b"][sl].float()
                st.enc_thr[i, :n] = src["enc_thr"][sl].float()
                st.dec_W[i, :n] = src["dec"][sl].float()
                st.en_mask[i, :n] = 1.0
        if gate is not None:
            st.gate_w.copy_(gate["w"])
            st.gate_b.copy_(gate["b"])
    dev = input_device(model)
    st.to(dev)
    st.invalidate()
    if attach:
        st.attach(model, decoder_layers(model))
    return st


# --------------------------------------------------------------------------------------
@torch.no_grad()
def _last_resid(model, tok, spec, prompts, layer, batch_size):
    dev, out = input_device(model), []
    ids_all = [chat_ids(tok, p, spec.chat_kwargs) for p in prompts]
    for chunk in batches(ids_all, batch_size):
        ids, att = left_pad(chunk, tok.pad_token_id, dev)
        with capture(model, [layer], last_k=1) as st:
            model(input_ids=ids, attention_mask=att)
        out.append(st[layer][:, -1].float().cpu())
    return torch.cat(out)


def train_gate(model, tok, spec, ldata, cfg, layer: int, keep_rate: float = 0.95
               ) -> tuple[dict | None, dict]:
    """Probe at the window's first layer: explicit language instruction (1) vs weak prompt (0).

    Trained on every disc+dev weak template and every dev explicit template.  The decision
    threshold is set by leave-one-template-out cross-validation so that >= `keep_rate` of weak
    prompts written with an UNSEEN template are still steered (a probe that only memorises
    templates would otherwise switch steering off for new phrasings).  Reported accuracy is on
    the eval templates, which are used nowhere in training or calibration."""
    import random as _r
    from sklearn.linear_model import LogisticRegression
    from .data import EXPLICIT_LANGS, EXPLICIT_TEMPLATES, WEAK_TEMPLATES

    rng = _r.Random(cfg.seed)
    units = ldata.dev[: min(len(ldata.dev), 300)]
    wt = WEAK_TEMPLATES["disc"] + WEAK_TEMPLATES["dev"]
    et = EXPLICIT_TEMPLATES["dev"]
    fill = lambda t, u: t.format(kw=u.kw, topic=u.topic or "news")
    weak = [(fill(wt[i % len(wt)], u), i % len(wt)) for i, u in enumerate(units)]
    expl = [(et[i % len(et)].format(p=fill(wt[(i * 7) % len(wt)], u), L=rng.choice(EXPLICIT_LANGS)),
             i % len(et)) for i, u in enumerate(units)]
    X = torch.cat([_last_resid(model, tok, spec, [p for p, _ in weak], layer, cfg.batch_size),
                   _last_resid(model, tok, spec, [p for p, _ in expl], layer, cfg.batch_size)]).numpy()
    y = np.r_[np.zeros(len(weak)), np.ones(len(expl))]
    gw = np.array([g for _, g in weak])
    ge = np.array([g for _, g in expl])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd

    def fit(mask):
        return LogisticRegression(C=0.5, max_iter=3000).fit(Z[mask], y[mask])

    # leave-one-template-out: fold f holds out weak template f and explicit template f % |et|
    oot = np.full(len(y), np.nan)
    for f in range(len(wt)):
        held = np.r_[gw == f, ge == (f % len(et))]
        if held.all() or not held.any():
            continue
        oot[held] = fit(~held).decision_function(Z[held])
    weak_oot, expl_oot = oot[: len(weak)], oot[len(weak):]
    tau = float(np.nanquantile(weak_oot, keep_rate))
    clf = fit(np.ones(len(y), dtype=bool))
    w = clf.coef_[0] / sd
    b = clf.intercept_[0] - float((clf.coef_[0] * mu / sd).sum()) - tau

    weak_ev = weak_prompts(ldata.eval[: len(units)], "eval")
    exp_ev = [p for p, _ in explicit_prompts(ldata.eval[: len(units)], "eval", seed=cfg.seed + 1)]
    Xe = torch.cat([_last_resid(model, tok, spec, weak_ev, layer, cfg.batch_size),
                    _last_resid(model, tok, spec, exp_ev, layer, cfg.batch_size)]).numpy()
    ye = np.r_[np.zeros(len(weak_ev)), np.ones(len(exp_ev))]
    pred = (Xe @ w + b) > 0
    rep = {"layer": layer, "tau": tau, "heldout_acc": float((pred == ye).mean()),
           "weak_kept_rate": float((~pred[: len(weak_ev)]).mean()),
           "explicit_caught_rate": float(pred[len(weak_ev):].mean()),
           "cv_weak_kept_rate": float(np.nanmean(weak_oot <= tau)),
           "cv_explicit_caught_rate": float(np.nanmean(expl_oot > tau))}
    print(f"[gate] {rep}")
    if rep["heldout_acc"] < cfg.gate_min_acc or rep["weak_kept_rate"] < 0.9:
        print("[gate] held-out accuracy or weak-prompt retention too low; gate disabled")
        return None, rep
    return {"w": torch.tensor(w, dtype=torch.float32), "b": torch.tensor([b], dtype=torch.float32),
            "layer": layer}, rep


# --------------------------------------------------------------------------------------
def tune(model, s1, diag, window, rank, cfg, d_model, bench, variant, gate, lam_grid, beta_grid,
         label=None):
    label = label or variant
    grid = []
    for lam in lam_grid:
        for beta in beta_grid:
            st = build_steerer(model, s1, diag, window, rank, lam, beta, cfg, d_model, variant, gate)
            r = bench.full(st)
            st.detach()
            grid.append({"variant": variant, "lam": lam, "beta": beta, **{k: v for k, v in r.items()
                                                                           if k != "leak"},
                         **{f"leak_{k}": v for k, v in r["leak"].items()}})
            print(f"[tune:{label}] lam={lam} beta={beta} gain={r['gain_dM']:.3f} "
                  f"DEFAULT+={r['gain_DEFAULT']:.3f} leak={r['leak_max']:.3f} "
                  f"kl={r['kl_offdecision']:.3f} util={r['util_nll']:.3f} "
                  f"util_all={r['util_nll_all']:.3f} feas={r['feasible']}")
    feas = [g for g in grid if g["feasible"]]
    if feas:
        best = max(feas, key=lambda g: (steer_score(g), -g["leak_max"]))
    else:
        # no point inside the guardrails: take the best trade-off (gain minus the excess over
        # each guardrail), never simply the weakest edit, and flag the artifact as infeasible
        best = max(grid, key=lambda g: objective(g, 0.0, cfg, bench.eps))
        print(f"[tune:{label}] WARNING: no point satisfies the guardrails {bench.eps}; best "
              f"trade-off is lam={best['lam']} beta={best['beta']} (excess {best['excess']:.3f}). "
              f"Artifact is flagged infeasible.")
    return best, grid


def outside_window(window, layers_with_support, n_layers):
    w = len(window)
    cands = []
    for a in range(0, n_layers - w + 1):
        W = list(range(a, a + w))
        if set(W) & set(window) or not all(l in layers_with_support for l in W):
            continue
        cands.append(W)
    if not cands:
        return None
    return max(cands, key=lambda W: abs(W[0] - window[0]))


# --------------------------------------------------------------------------------------
def run_stage3(model, tok, spec, ldata, cfg, s1, s2, out_dir, bench, d_model, run_ablations=True):
    diag = s2["diagnostics"]
    W, rank = s2["window"], s2["rank"]
    rank = rank if rank in ("spectral", "mean") else int(rank)
    gate, gate_rep = (train_gate(model, tok, spec, ldata, cfg, W[0]) if cfg.use_gate else (None, {}))

    best, grid = tune(model, s1, diag, W, rank, cfg, d_model, bench, "full", gate,
                      cfg.lam_grid, cfg.beta_grid)
    rows = list(grid)
    variants = {"full": {"window": W, "lam": best["lam"], "beta": best["beta"], "dev": best}}
    if run_ablations:
        abl = {"pos_only": (W, cfg.lam_grid, (0.0,)),
               "neg_only": (W, (1.0,), cfg.beta_grid),
               "sparse_only": (W, cfg.lam_grid, (best["beta"],)),
               "random": (W, cfg.lam_grid, (best["beta"],)),
               "caa": (W, cfg.caa_lam_grid, (0.0,))}
        Wout = outside_window(W, set(s1["layers"]), spec.n_layers)
        if Wout:
            abl["outside_window"] = (Wout, cfg.lam_grid, (best["beta"],))
        for name, (Wv, lg, bg) in abl.items():
            v = "full" if name == "outside_window" else name
            b, g = tune(model, s1, diag, Wv, rank, cfg, d_model, bench, v, gate, lg, bg, label=name)
            for row in g:
                row["variant"] = name
            rows += g
            variants[name] = {"window": Wv, "lam": b["lam"], "beta": b["beta"], "dev": b,
                              "builder_variant": v}

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "stage3_grid.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- held-out comparison of every method at its dev-selected point (devtest sentences,
    #      eval-only templates; nothing here was used for any selection) ----
    ev = DevBench(model, tok, spec, ldata, bench.ts, cfg, bench.others, split="eval",
                  n=cfg.n_heldout_prompts)
    held = {"prompt": {**ev.prompt_ref, "lam": None, "beta": None}}
    for name, v in variants.items():
        st = build_steerer(model, s1, diag, v["window"], rank, v["lam"], v["beta"], cfg, d_model,
                           v.get("builder_variant", "full"), gate)
        held[name] = {**ev.full(st), "lam": v["lam"], "beta": v["beta"], "window": v["window"]}
        st.detach()
        h = held[name]
        print(f"[heldout:{name}] DEFAULT+={h['gain_DEFAULT']:.3f} {h['gain_DEFAULT_ci']} "
              f"gain={h['gain_dM']:.3f} leak={h['leak_max']:.3f} kl={h['kl_offdecision']:.3f} "
              f"util={h['util_nll']:.3f}")
    h = held["prompt"]
    print(f"[heldout:prompt] DEFAULT+={h['gain_DEFAULT']:.3f} {h['gain_DEFAULT_ci']} "
          f"gain={h['gain_dM']:.3f} leak={h['leak_max']:.3f} kl={h['kl_offdecision']:.3f} "
          f"util={h['util_nll']:.3f}")
    json.dump(held, open(os.path.join(out_dir, "stage3_heldout.json"), "w"), indent=1, default=str)

    final = build_steerer(model, s1, diag, W, rank, best["lam"], best["beta"], cfg, d_model,
                          "full", gate, attach=False)
    artifact = {
        "state_dict": {k: v.detach().cpu() for k, v in final.state_dict().items()},
        "config": {"layers": final.layers, "k_en": int(final.en_mask.shape[1]), "mode": cfg.mode,
                   "k_decode": cfg.k_decode, "beta": best["beta"], "lam": best["lam"],
                   "sink_factor": cfg.sink_factor,
                   "gamma": 1.0, "use_gate": gate is not None, "rank": str(s2["rank"]),
                   "window": W, "model": spec.key, "base_model": spec.hf_id,
                   "target": ldata.target, "sae_repo": spec.sae_repo,
                   "feasible": bool(best["feasible"]), "guardrail_eps": bench.eps},
        "variants": variants, "gate": gate_rep, "protocol": cfg.to_dict(),
    }
    torch.save(artifact, os.path.join(out_dir, "artifact.pt"))
    json.dump({k: v for k, v in artifact.items() if k != "state_dict"},
              open(os.path.join(out_dir, "artifact.json"), "w"), indent=1, default=str)
    print(f"[stage3] operating point lam={best['lam']} beta={best['beta']} window={W} "
          f"dev gain={best['gain_dM']:.3f} DEFAULT+={best['gain_DEFAULT']:.3f} "
          f"feasible={best['feasible']}")
    return artifact, gate
