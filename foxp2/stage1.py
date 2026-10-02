"""Stage I: localize language features in a frozen SAE basis.

Changes vs. the submitted paper (review fixes):
  * Selectivity is computed on a RESPONSE-language contrast with the prompt held fixed
    (activations at the first k response positions), so features reflect output
    realization, not input script.  `cfg.contrast="input"` reproduces the original.
  * Causal lift is screened for every candidate with one backward pass per batch
    (attribution: d dM / d alpha = sum_pos grad_h . d_j) and the top candidates are then
    re-checked with real additive interventions (`lift_mode="verify"`).  `"intervention"`
    re-checks every candidate (slow, gradient-free); `"attribution"` skips re-checking.
  * Lift is measured per unit of the feature's natural activation, so slopes are comparable
    across features with very different scales.
  * Two supports are selected: target-promoting features N_tgt (Sel>0, lift>0) and
    English-promoting features N_en (Sel<0, lift<0).  N_en makes the suppression term
    English-specific instead of "whatever is active".
  * Every layer reports the SAE's FVU on this checkpoint's activations, a random support of
    equal size (for the random-support ablation and Stage II nulls), and a token-affinity
    readout per selected feature (logit lens of its decoder row) with the fraction of
    top tokens in the target script (script-detector vs language-level features).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

from .config import SCRIPT
from .data import contrast_pairs, token_scripts, weak_prompts
from .metrics import delta_m_from_logits, tf_logits, tf_sequences
from .model_io import (add_vectors, batches, capture, chat_ids, decoder_layers, input_device,
                       left_pad, text_ids)
from .saes import self_check


# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_contrast(model, tok, spec, pairs, layers, k, contrast, batch_size):
    """Residuals at the contrast positions.  Returns ({l: H_tgt [N,k,d]}, {l: H_en}, keep_idx)."""
    keep, seq_t, seq_e = [], [], []
    for i, p in enumerate(pairs):
        if contrast == "response":
            base = chat_ids(tok, p.prompt_en, spec.chat_kwargs)
            rt, re_ = text_ids(tok, p.resp_tgt)[:k], text_ids(tok, p.resp_en)[:k]
            if len(rt) < k or len(re_) < k:
                continue
            seq_t.append(base + rt)
            seq_e.append(base + re_)
        else:
            seq_t.append(chat_ids(tok, p.prompt_tgt, spec.chat_kwargs))
            seq_e.append(chat_ids(tok, p.prompt_en, spec.chat_kwargs))
        keep.append(i)
    kk = k if contrast == "response" else 1
    dev = input_device(model)
    H = {"tgt": {l: [] for l in layers}, "en": {l: [] for l in layers}}
    for name, seqs in (("tgt", seq_t), ("en", seq_e)):
        for chunk in batches(seqs, batch_size):
            ids, att = left_pad(chunk, tok.pad_token_id, dev)
            with capture(model, layers, last_k=kk) as st:
                model(input_ids=ids, attention_mask=att)
            for l in layers:
                H[name][l].append(st[l].to("cpu", torch.float16))
    Ht = {l: torch.cat(H["tgt"][l]) for l in layers}
    He = {l: torch.cat(H["en"][l]) for l in layers}
    return Ht, He, keep


def attribution_gradients(model, tok, seqs, ts, target, T, layers, batch_size):
    """G_l = mean over prompts of sum over the last T positions of d dM / d h_l  -> [d]."""
    dev = input_device(model)
    emb = model.get_input_embeddings()
    G = {l: None for l in layers}
    n = 0
    for chunk in batches(seqs, max(1, batch_size // 2)):
        ids, att = left_pad(chunk, tok.pad_token_id, dev)
        x = emb(ids).detach().requires_grad_(True)
        with capture(model, layers, keep_grad=True) as st:
            logits = model(inputs_embeds=x, attention_mask=att).logits[:, -T:]
            dM, _ = delta_m_from_logits(logits, ts, target)
            dM.sum().backward()
        for l in layers:
            g = st[l].grad[:, -T:].float().sum(1).sum(0).cpu()
            G[l] = g if G[l] is None else G[l] + g
        n += len(chunk)
        model.zero_grad(set_to_none=True)
    return {l: G[l] / n for l in layers}


@torch.no_grad()
def verify_lift(model, tok, seqs, ts, target, T, layer, dirs, mags, alphas, base_dM,
                batch_size):
    """Real additive interventions.  dirs [n,d] decoder rows, mags [n].  Returns slopes [n]."""
    slopes = []
    for j in range(dirs.shape[0]):
        vals = []
        for a in alphas:
            with add_vectors(model, {layer: a * mags[j] * dirs[j]}, last_k=T):
                dM, _ = delta_m_from_logits(tf_logits(model, tok, seqs, T, batch_size), ts, target)
            vals.append(float(dM.mean() - base_dM) / a)
        slopes.append(float(np.median(vals)))
    return torch.tensor(slopes)


def token_affinity(model, tok, dirs, script, top=10):
    norm = getattr(getattr(model, "model", model), "norm", None)
    W_U = model.get_output_embeddings().weight
    out = []
    with torch.no_grad():
        for v in dirs:
            v = v.to(W_U.device, W_U.dtype)[None]
            v = norm(v) if norm is not None else v
            ids = (v @ W_U.T)[0].float().topk(top).indices.tolist()
            toks = [tok.decode([i]) for i in ids]
            frac = float(np.mean([script in token_scripts(t) for t in toks]))
            out.append({"tokens": toks, "target_script_frac": frac})
    return out


# --------------------------------------------------------------------------------------
def run_stage1(model, tok, spec, ldata, ts, cfg, sae_loader, out_dir, device=None):
    os.makedirs(out_dir, exist_ok=True)
    target = ldata.target
    layers = cfg.layers or list(range(spec.n_layers))
    device = device or input_device(model)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    pairs = contrast_pairs(ldata.disc, cfg.contrast)
    Ht, He, keep = collect_contrast(model, tok, spec, pairs, layers, cfg.k_resp, cfg.contrast,
                                    cfg.batch_size)
    domains = [pairs[i].domain for i in keep]
    print(f"[stage1] {len(keep)} contrast pairs, {len(layers)} layers")

    lift_prompts = [chat_ids(tok, p, spec.chat_kwargs)
                    for p in weak_prompts(ldata.disc, "disc", cfg.n_lift_prompts)]
    seqs = tf_sequences(model, tok, lift_prompts, cfg.T, cfg.batch_size)
    base_dM_all, _ = delta_m_from_logits(tf_logits(model, tok, seqs, cfg.T, cfg.batch_size), ts, target)
    vseqs = seqs[: cfg.n_verify_prompts]
    base_dM_v = float(base_dM_all[: cfg.n_verify_prompts].mean())

    G = None
    if cfg.lift_mode in ("attribution", "verify"):
        G = attribution_gradients(model, tok, seqs, ts, target, cfg.T, layers, cfg.batch_size)

    result = {"layers": {}, "meta": {"target": target, "model": spec.key, "contrast": cfg.contrast,
                                     "n_pairs": len(keep), "base_dM_lift": float(base_dM_all.mean()),
                                     "domains": domains}}
    report = {}
    for l in layers:
        sae = sae_loader(l).to(device)
        xt, xe = Ht[l].to(device).float(), He[l].to(device).float()
        fv = self_check(sae, torch.cat([xt.flatten(0, 1), xe.flatten(0, 1)]))
        zt = sae.encode(xt).mean(1)          # [N, m]
        ze = sae.encode(xe).mean(1)
        mu_t, mu_e = zt.mean(0), ze.mean(0)
        sel = (mu_t - mu_e) / (zt.std(0) + ze.std(0) + 1e-6)
        sel[(mu_t == 0) & (mu_e == 0)] = 0
        n_scr = min(cfg.n_screen, sae.m)
        cand_t = sel.topk(n_scr).indices
        cand_e = (-sel).topk(n_scr).indices
        cand_t = cand_t[sel[cand_t] > 0]
        cand_e = cand_e[sel[cand_e] < 0]
        mag_t = mu_t[cand_t].clamp_min(1e-4)
        mag_e = mu_e[cand_e].clamp_min(1e-4)

        # ---- causal lift per natural-activation unit ----
        dec_t, dec_e = sae.dec_rows(cand_t), sae.dec_rows(cand_e)
        if G is not None:
            g = G[l].to(device)
            slope_t, slope_e = (dec_t @ g) * mag_t, (dec_e @ g) * mag_e
        else:
            slope_t, slope_e = torch.zeros(len(cand_t), device=device), torch.zeros(len(cand_e), device=device)
        if cfg.lift_mode in ("verify", "intervention"):
            for side in ("t", "e"):
                sl, sel_s = (slope_t, sel[cand_t]) if side == "t" else (-slope_e, -sel[cand_e])
                dec, mag = (dec_t, mag_t) if side == "t" else (dec_e, mag_e)
                if cfg.lift_mode == "verify":
                    pri = (sel_s.clamp_min(0) * sl.clamp_min(0)) + 1e-6 * sel_s
                    pick = pri.topk(min(cfg.n_verify, len(pri))).indices
                else:
                    pick = torch.arange(len(sl), device=device)
                if len(pick) == 0:
                    continue
                v = verify_lift(model, tok, vseqs, ts, target, cfg.T, l, dec[pick], mag[pick],
                                cfg.lift_alphas, base_dM_v, cfg.batch_size).to(device)
                if side == "t":
                    slope_t[pick] = v
                else:
                    slope_e[pick] = v
        score_t = sel[cand_t].clamp_min(0) * slope_t.clamp_min(0)
        score_e = (-sel[cand_e]).clamp_min(0) * (-slope_e).clamp_min(0)
        # tie-break: larger lift, then larger selectivity, then smaller index
        ord_t = sorted(range(len(cand_t)), key=lambda i: (-float(score_t[i]), -float(slope_t[i]),
                                                          -float(sel[cand_t[i]]), int(cand_t[i])))
        ord_e = sorted(range(len(cand_e)), key=lambda i: (-float(score_e[i]), float(slope_e[i]),
                                                          float(sel[cand_e[i]]), int(cand_e[i])))
        ord_t = [i for i in ord_t if score_t[i] > 0][: cfg.K_tgt]
        ord_e = [i for i in ord_e if score_e[i] > 0][: cfg.K_en]
        if not ord_t:
            print(f"[stage1] layer {l}: no feature passes both criteria")
            report[l] = {"fvu": fv, "n_tgt": 0, "n_en": 0}
            del sae
            continue
        idx_t, idx_e = cand_t[ord_t], cand_e[ord_e]
        idx = torch.cat([idx_t, idx_e])

        alive = torch.nonzero((mu_t > 0) | (mu_e > 0)).flatten().tolist()
        rand_idx = torch.tensor(rng.sample(alive, min(len(idx), len(alive))), device=device)

        def pack(ix):
            thr = (sae.threshold[ix] if sae.act == "jumprelu"
                   else torch.full((len(ix),), sae.topk_threshold(torch.cat([xt, xe]).flatten(0, 1)),
                                   device=device))
            return {"idx": ix.cpu(), "Zt": zt[:, ix].cpu(), "Ze": ze[:, ix].cpu(),
                    "dec": sae.dec_rows(ix).cpu(), "enc_W": sae.W_enc[:, ix].cpu(),
                    "enc_b": sae.b_enc[ix].cpu(), "enc_thr": thr.cpu()}

        L = pack(idx)
        L.update({"n_tgt": len(idx_t), "n_en": len(idx_e),
                  "sel": sel[idx].cpu(), "slope": torch.cat([slope_t[ord_t], slope_e[ord_e]]).cpu(),
                  "score": torch.cat([score_t[ord_t], score_e[ord_e]]).cpu(),
                  "in_scale": float(sae.in_scale),
                  "in_bias": (sae.in_bias.cpu() if sae.in_bias is not None
                              else torch.zeros(sae.d)),
                  "rand": pack(rand_idx),
                  "caa": (xt - xe).mean((0, 1)).cpu(),          # residual diff-in-means baseline
                  "fvu": fv})
        if sae.act == "topk":   # agreement of the threshold approximation on the support
            exact = torch.cat([zt, ze])[:, idx]
            x = torch.cat([xt, xe]).mean(1)
            pre = sae.pre(x)[:, idx]
            approx = pre * (pre > L["enc_thr"].to(device))
            L["topk_approx_corr"] = float(torch.corrcoef(torch.stack([
                exact.flatten(), approx.flatten()]))[0, 1])
        L["affinity"] = token_affinity(model, tok, L["dec"][: len(idx_t)], SCRIPT[target])
        result["layers"][l] = L
        report[l] = {"fvu": fv, "n_tgt": len(idx_t), "n_en": len(idx_e),
                     "mean_target_script_frac": float(np.mean([a["target_script_frac"]
                                                               for a in L["affinity"]]))}
        print(f"[stage1] layer {l}: |N_tgt|={len(idx_t)} |N_en|={len(idx_e)} "
              f"FVU={fv['fvu']:.3f}")
        del sae, zt, ze
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["meta"]["report"] = report
    torch.save(result, os.path.join(out_dir, "stage1.pt"))
    json.dump({"meta": {k: v for k, v in result["meta"].items() if k != "domains"},
               "affinity": {l: L["affinity"][:10] for l, L in result["layers"].items()}},
              open(os.path.join(out_dir, "stage1_report.json"), "w"), indent=1, default=str)
    return result
