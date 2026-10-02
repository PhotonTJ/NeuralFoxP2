#!/usr/bin/env python
"""Every paper figure and table from the pipeline outputs, for all models x languages.

    python scripts/make_figures.py --runs runs --ckpts checkpoints --lmeval results --out figures

Each figure answers one reviewer question (see FIGURES below).  Missing inputs are skipped
with a note, so the script can be run at any point.  Every figure is written as PDF + PNG,
and every figure has a CSV with the exact numbers behind it (the table view).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from foxp2.config import LANG_NAME, MODELS, TARGETS  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FIGURES = {
    "fig01": "Gain vs cost frontier per language: does FOXP2 beat CAA and prompting at equal cost?",
    "fig02": "Per-layer causal effect + selected window: is the decision localized, and where?",
    "fig03": "Spectral evidence vs nulls: is the language shift genuinely low-rank?",
    "fig04": "SAE FVU on the instruction-tuned models: does the base-model dictionary transfer?",
    "fig05": "Share of the residual language shift carried by the SAE support",
    "fig06": "Held-out ablation heatmap (all methods x languages, gain and utility cost)",
    "fig07": "Held-out macro comparison of all methods with 95% CIs",
    "fig08": "Response-level: % responses entirely in target + mid-response switching",
    "fig09": "Response-language leakage matrix under steering",
    "fig10": "Explicit-instruction conflict test + gate accuracy",
    "fig11": "Dose-response in lambda (gain, utility, KL) - controllability",
    "fig12": "English suppression (beta) interaction with promotion",
    "fig13": "Utility preservation on benchmarks (translation chrF++, lm-eval target/English)",
    "fig14": "Selected intervention windows across languages and models",
    "fig15": "Interpretability: share of selected features that are target-script detectors",
}

# ---- fixed visual system (reference palette; validated: CVD/normal-vision pass, contrast WARN
#      on aqua/yellow/magenta -> every series also has a marker shape, legend and CSV) ----
INK, MUTED, GRID, NEUTRAL = "#1f1f1e", "#6b6a64", "#e6e5e0", "#8a8984"
METHODS = [("full", "FOXP2", "#2a78d6", "o"), ("caa", "CAA", "#eb6834", "s"),
           ("prompt", "Prompting", "#1baf7a", "^"), ("pos_only", "No English suppression", "#eda100", "D"),
           ("neg_only", "Suppression only", "#e87ba4", "v"), ("sparse_only", "Mean shift only", "#008300", "P"),
           ("outside_window", "Outside window", "#4a3aa7", "X"), ("random", "Random support", "#e34948", "h")]
MLAB = {k: l for k, l, _, _ in METHODS}
MCOL = {k: c for k, _, c, _ in METHODS}
MMRK = {k: m for k, _, _, m in METHODS}
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # 4 models / 4 series
MARKS = ["o", "s", "^", "D"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fd", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#b8302f", "#e88c8b", "#f0efec", "#86b6ef", "#1c5cab"])


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "savefig.dpi": 200})


# --------------------------------------------------------------------------------------
def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def csv_rows(p):
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return []


def f(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def ci(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return [float("nan")] * 2
    return x if isinstance(x, (list, tuple)) and len(x) == 2 else [float("nan")] * 2


class Data:
    def __init__(self, runs, ckpts, lmeval):
        self.models = [m for m in MODELS if os.path.isdir(os.path.join(runs, m))]
        self.d = {}
        for m in self.models:
            for l in TARGETS:
                rd, cd = os.path.join(runs, m, l), os.path.join(ckpts, f"{m}-foxp2-{l}")
                x = {"s2": jload(os.path.join(rd, "stage2.json")),
                     "s1": jload(os.path.join(rd, "stage1_report.json")),
                     "grid": csv_rows(os.path.join(rd, "stage3_grid.csv")),
                     "held": jload(os.path.join(rd, "stage3_heldout.json")),
                     "art": jload(os.path.join(rd, "artifact.json")),
                     "eval": jload(os.path.join(cd, f"eval_{l}.json")),
                     "lm": {k: lm_results(os.path.join(lmeval, m, l, k)) for k in
                            ("target/gamma0", "target/gamma1", "english/gamma0", "english/gamma1")}}
                if any(v for k, v in x.items() if k != "lm") or any(x["lm"].values()):
                    self.d[(m, l)] = x
        self.langs = [l for l in TARGETS if any((m, l) in self.d for m in self.models)]

    def get(self, m, l, k):
        return self.d.get((m, l), {}).get(k)


def lm_results(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "results_*.json"), recursive=True))
    if not files:
        return None
    res = {}
    for fp in files:                      # several runs (log-likelihood + generative); latest wins
        res.update((jload(fp) or {}).get("results", {}))
    vals = []
    for task, r in res.items():
        for key in ("acc,none", "acc_norm,none", "exact_match,flexible-extract",
                    "exact_match,strict-match", "exact_match,none"):
            if key in r and isinstance(r[key], (int, float)):
                vals.append(float(r[key]))
                break
    return float(np.mean(vals)) if vals else None


def save(fig, out, name, rows=None, header=None):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    if rows:
        with open(os.path.join(out, f"{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    print(f"[fig] {name}")


def skip(name, why):
    print(f"[skip] {name}: {why}")


def depth_series(D, getter):
    """{model: (rel_depth array, median, q25, q75)} of a per-layer quantity, across languages."""
    out = {}
    for m in D.models:
        n = MODELS[m].n_layers
        per = {}
        for l in D.langs:
            for layer, v in (getter(m, l) or {}).items():
                if v is not None and np.isfinite(v):
                    per.setdefault(int(layer), []).append(v)
        if per:
            ks = sorted(per)
            a = [np.array(per[k]) for k in ks]
            out[m] = (np.array(ks) / max(n - 1, 1), np.array([np.median(x) for x in a]),
                      np.array([np.percentile(x, 25) for x in a]), np.array([np.percentile(x, 75) for x in a]))
    return out


def lang_ticks(ax, langs):
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANG_NAME[l] for l in langs], rotation=45, ha="right")


def pareto(pts):
    out, best = [], -1e9
    for c, g in sorted(pts, key=lambda p: (p[0], -p[1])):
        if g > best:
            out.append((c, g))
            best = g
    return out


def symlog_x(ax, kl=False, ticks=(-0.1, 0, 0.01, 0.03, 0.1, 0.3, 1, 3)):
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    lo, hi = ax.dataLim.x0, ax.dataLim.x1
    t = [x for x in ticks if (not kl or x >= 0) and lo - 1e-9 <= x <= hi * 1.5 + 1e-9]
    ax.set_xlim(min(lo, 0) - 0.002, hi * 1.6 + 0.005)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{x:g}" for x in t])
    ax.minorticks_off()


# --------------------------------------------------------------------------------------
def fig01(D, out):
    for m in D.models:
        langs = [l for l in D.langs if D.get(m, l, "grid")]
        if not langs:
            skip(f"fig01_frontier_{m}", "no stage3_grid.csv")
            continue
        nc = 4
        nr = math.ceil(len(langs) / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.3 * nr), sharex=True, sharey=True,
                                 squeeze=False, constrained_layout=True)
        rows = []
        for ax, l in zip(axes.flat, langs):
            g = D.get(m, l, "grid")
            for key in ("full", "caa"):
                pts = [(f(r["util_nll"]), f(r["gain_DEFAULT"])) for r in g if r["variant"] == key]
                if not pts:
                    continue
                ax.scatter(*zip(*pts), s=12, color=MCOL[key], alpha=0.3, marker=MMRK[key], lw=0)
                fr = pareto(pts)
                ax.plot(*zip(*fr), color=MCOL[key], lw=1.8, marker=MMRK[key], ms=4.5,
                        mec="white", mew=1, label=MLAB[key])
                rows += [[l, key, c, y] for c, y in fr]
            s2 = D.get(m, l, "s2") or {}
            pr, eps = s2.get("prompt_reference"), s2.get("guardrail_eps", {})
            if pr:
                ax.scatter([pr["util_nll"]], [pr["gain_DEFAULT"]], s=55, facecolors="none",
                           edgecolors=INK, lw=1.5, zorder=4, label="Prompting")
                rows.append([l, "prompt", pr["util_nll"], pr["gain_DEFAULT"]])
            if "util" in eps:
                ax.axvline(eps["util"], color=MUTED, lw=0.9, ls=(0, (4, 3)))
            ax.set_title(LANG_NAME[l])
            symlog_x(ax, ticks=(-0.1, 0, 0.01, 0.1, 1))
            ax.set_ylim(-0.05, 1.05)
        for ax in axes.flat[len(langs):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("DEFAULT gain (dev)")
        for ax in axes[-1, :]:
            ax.set_xlabel("Utility cost (Δ NLL)")
        h, lb = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, lb, loc="outside lower center", ncol=3)
        fig.suptitle(f"{m}: gain vs. utility cost (Pareto frontier; dashed = guardrail)")
        save(fig, out, f"fig01_frontier_{m}", rows, ["lang", "method", "util_nll", "gain_DEFAULT"])


def fig02(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "s2") for l in D.langs)]
    if not ms:
        return skip("fig02_causal_curve", "no stage2.json")
    fig, axes = plt.subplots(len(ms), 1, figsize=(9, 2.2 * len(ms) + 0.6), squeeze=False,
                             constrained_layout=True)
    rows = []
    for ax, m in zip(axes[:, 0], ms):
        n = MODELS[m].n_layers
        langs = [l for l in D.langs if D.get(m, l, "s2")]
        M = np.full((len(langs), n), np.nan)
        for i, l in enumerate(langs):
            s2 = D.get(m, l, "s2")
            for k, v in s2.get("curve", {}).items():
                M[i, int(k)] = v
                rows.append([m, l, int(k), v])
            W = s2.get("window") or []
            if W:
                ax.add_patch(Rectangle((W[0] - 0.5, i - 0.5), len(W), 1, fill=False, ec=INK, lw=1.6))
        vmax = float(np.nanpercentile(M, 97)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, aspect="auto", cmap=SEQ, interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels([LANG_NAME[l] for l in langs])
        ax.set_xlabel("Layer")
        ax.set_title(f"{m}: single-layer causal effect on dM (box = selected window)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("dM gain (scale clipped at 97th pct.)", color=MUTED)
    save(fig, out, "fig02_causal_curve", rows, ["model", "lang", "layer", "tf_gain"])


def _window_median(s2, key, sub):
    W = s2.get("window") or []
    vals = []
    for l in W:
        d = s2.get("diagnostics", {}).get(str(l)) or s2.get("diagnostics", {}).get(l)
        if d and d.get(key):
            vals.append(d[key][sub] if sub else d[key])
    return float(np.median(vals)) if vals else float("nan")


def fig03(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "s2") for l in D.langs)]
    if not ms:
        return skip("fig03_low_rank", "no stage2.json")
    series = [("spectral", "Language shift", 0), ("null_domain", "Null: topic contrast", 1),
              ("null_colshuffle", "Null: column shuffle", 2), ("null_random_support", "Null: random support", 3)]
    fig, axes = plt.subplots(2, len(ms), figsize=(3.3 * len(ms), 5.2), squeeze=False,
                             constrained_layout=True)
    rows = []
    for j, m in enumerate(ms):
        langs = [l for l in D.langs if D.get(m, l, "s2")]
        for i, (metric, ylab) in enumerate([("top1_var", "Top-1 variance share"), ("r_eff", "Effective rank")]):
            ax = axes[i, j]
            for key, lab, s in series + [("centred", "Language shift, mean removed", 0)]:
                y = [_window_median(D.get(m, l, "s2"), key, metric) for l in langs]
                hollow = key == "centred"
                x = np.arange(len(langs)) + ((-2 if hollow else s) - 1.0) * 0.12
                ax.scatter(x, y, marker=MARKS[s], s=26, label=lab, zorder=3, lw=1.2,
                           facecolors="none" if hollow else SLOTS[s], edgecolors=SLOTS[s])
                rows += [[m, l, metric, key, v] for l, v in zip(langs, y)]
            lang_ticks(ax, langs)
            ax.set_ylabel(ylab) if j == 0 else None
            ax.set_title(m if i == 0 else "")
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=5)
    fig.suptitle("Low-rank evidence (median over the selected window's layers); compare the hollow "
                 "points with the column-shuffle null, which is also mean-removed")
    save(fig, out, "fig03_low_rank", rows, ["model", "lang", "metric", "series", "value"])


def _depth_fig(D, out, name, panels, title, ref=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False,
                             constrained_layout=True)
    rows, any_data = [], False
    for ax, (getter, ylab) in zip(axes[0], panels):
        ds = depth_series(D, getter)
        for s, (m, (x, med, q1, q3)) in enumerate(ds.items()):
            any_data = True
            ax.fill_between(x, q1, q3, color=SLOTS[s % 4], alpha=0.15, lw=0)
            ax.plot(x, med, color=SLOTS[s % 4], lw=2, marker=MARKS[s % 4], ms=3.5, markevery=4, label=m)
            rows += [[m, ylab, float(a), float(b)] for a, b in zip(x, med)]
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.set_xlabel("Relative depth (layer / last layer)")
        ax.set_ylabel(ylab)
    if not any_data:
        plt.close(fig)
        return skip(name, "no per-layer data")
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=4)
    fig.suptitle(title)
    save(fig, out, name, rows, ["model", "quantity", "rel_depth", "median_over_languages"])


def fig04(D, out):
    def g(m, l):
        s1 = D.get(m, l, "s1") or {}
        rep = (s1.get("meta") or {}).get("report") or {}
        return {k: (v.get("fvu") or {}).get("fvu") for k, v in rep.items()}
    _depth_fig(D, out, "fig04_sae_fvu", [(g, "FVU (lower is better)")],
               "Base-model SAE reconstruction on the instruction-tuned checkpoints (median, IQR band)",
               ref=0.5)


def fig05(D, out):
    def gk(key):
        def g(m, l):
            s2 = D.get(m, l, "s2") or {}
            return {k: v.get(key) for k, v in (s2.get("diagnostics") or {}).items()}
        return g
    _depth_fig(D, out, "fig05_support_coverage",
               [(gk("support_vs_caa_norm"), "‖support shift‖ / ‖residual shift‖"),
                (gk("support_vs_caa_cos"), "cos(support shift, residual shift)")],
               "How much of the residual-stream language shift the SAE support carries")


def _held(D, m, l, key, field="gain_DEFAULT"):
    h = (D.get(m, l, "held") or {}).get(key)
    return (f(h.get(field)), ci(h.get(field + "_ci"))) if h else (float("nan"), [float("nan")] * 2)


def fig06(D, out):
    for m in D.models:
        langs = [l for l in D.langs if D.get(m, l, "held")]
        if not langs:
            skip(f"fig06_heldout_heatmap_{m}", "no stage3_heldout.json")
            continue
        keys = [k for k, *_ in METHODS]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
        rows = []
        for ax, (field, lab, cmap, div) in zip(axes, [("gain_DEFAULT", "DEFAULT gain", DIV, True),
                                                     ("util_nll", "Utility cost (Δ NLL)", SEQ, False)]):
            M = np.array([[_held(D, m, l, k, field)[0] for l in langs] for k in keys])
            M = np.c_[M, np.nanmean(M, 1)]
            v = np.nanmax(np.abs(M)) or 1
            norm = TwoSlopeNorm(0, -v, v) if div else None
            im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, vmin=None if div else 0,
                           vmax=None if div else max(v, 1e-6))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        rgba = im.cmap(im.norm(M[i, j]))
                        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                        ax.text(j, i, f"{M[i, j]:.2f}" if div else f"{M[i, j]:.3f}", ha="center",
                                va="center", fontsize=6.5,
                                color="white" if lum < 0.5 else INK)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels([MLAB[k] for k in keys])
            ax.set_xticks(range(len(langs) + 1))
            ax.set_xticklabels([LANG_NAME[l] for l in langs] + ["Mean"], rotation=45, ha="right")
            ax.axvline(len(langs) - 0.5, color="white", lw=2)
            ax.grid(False)
            ax.set_title(lab)
            fig.colorbar(im, ax=ax, pad=0.01)
            rows += [[k, l, field, M[i, j]] for i, k in enumerate(keys) for j, l in enumerate(langs)]
        fig.suptitle(f"{m}: held-out (FLORES+ devtest, unseen templates), each method at its dev-selected point")
        save(fig, out, f"fig06_heldout_heatmap_{m}", rows, ["method", "lang", "metric", "value"])


def fig07(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "held") for l in D.langs)]
    if not ms:
        return skip("fig07_heldout_macro", "no stage3_heldout.json")
    keys = [k for k, *_ in METHODS]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 3.6), sharey=True, squeeze=False,
                             constrained_layout=True)
    rows = []
    for ax, m in zip(axes[0], ms):
        langs = [l for l in D.langs if D.get(m, l, "held")]
        for i, k in enumerate(keys[::-1]):
            vals = np.array([_held(D, m, l, k)[0] for l in langs])
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            mu = vals.mean()
            hw = 1.96 * vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)),
                       s=9, color=MCOL[k], alpha=0.35, lw=0)
            ax.errorbar(mu, i, xerr=hw, fmt=MMRK[k], color=MCOL[k], ms=6, mec="white", mew=1,
                        capsize=3, lw=1.6)
            rows.append([m, k, mu, mu - hw, mu + hw, len(vals)])
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([MLAB[k] for k in keys[::-1]])
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.set_xlabel("Held-out DEFAULT gain")
        ax.set_title(m)
    fig.suptitle("Mean over languages (dots = languages; bars = 95% CI across languages)")
    save(fig, out, "fig07_heldout_macro", rows, ["model", "method", "mean", "ci_lo", "ci_hi", "n_langs"])


def fig08(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "eval") for l in D.langs)]
    if not ms:
        return skip("fig08_response_level", "no eval_<lang>.json (run scripts/evaluate.py)")
    series = [("unedited", "Unedited", NEUTRAL, "o", lambda e: e.get("response_gamma0")),
              ("prompt_en", "Prompt: “Answer in <L>.”", MCOL["prompt"], "^",
               lambda e: (e.get("prompt_baselines") or {}).get("instr_end")),
              ("prompt_native", "Prompt in the target language", MCOL["prompt"], "v",
               lambda e: (e.get("prompt_baselines") or {}).get("instr_native")),
              ("foxp2", "FOXP2 (weak prompt)", MCOL["full"], "o", lambda e: e.get("response_gamma1"))]
    fig, axes = plt.subplots(2, len(ms), figsize=(4.2 * len(ms), 6), squeeze=False, sharey="row",
                             constrained_layout=True)
    rows = []
    for j, m in enumerate(ms):
        langs = [l for l in D.langs if D.get(m, l, "eval")]
        for r, (field, cfield, ylab) in enumerate([("entire_target", "entire_target_ci",
                                                    "% responses entirely in target"),
                                                   ("switch_away_rate", "switch_away_ci",
                                                    "Mid-response switch-away rate")]):
            ax = axes[r, j]
            for s, (key, lab, col, mk, get) in enumerate(series):
                if r == 1 and key.startswith("prompt"):
                    continue
                xs, ys, lo, hi = [], [], [], []
                for i, l in enumerate(langs):
                    v = get(D.get(m, l, "eval")) or {}
                    if field in v:
                        c = ci(v.get(cfield))
                        xs.append(i + (s - 1.5) * 0.17)
                        ys.append(100 * v[field])
                        lo.append(100 * (v[field] - c[0]) if np.isfinite(c[0]) else 0)
                        hi.append(100 * (c[1] - v[field]) if np.isfinite(c[1]) else 0)
                        rows.append([m, l, field, key, v[field], c[0], c[1]])
                if xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=6.5, lw=1.1, capsize=0,
                                mec=col, mew=1.4, label=lab, elinewidth=1.1,
                                mfc="white" if key == "prompt_native" else col)
            lang_ticks(ax, langs)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_title(m if r == 0 else "")
            ax.set_ylim(-3, 103)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=4)
    fig.suptitle("Response-level language (whole-response LID; Wilson 95% CIs)")
    save(fig, out, "fig08_response_level", rows, ["model", "lang", "metric", "condition", "value",
                                                  "ci_lo", "ci_hi"])


def fig09(D, out):
    for m in D.models:
        langs = [l for l in D.langs if D.get(m, l, "eval")]
        if not langs:
            skip(f"fig09_leakage_matrix_{m}", "no eval json")
            continue
        cols = ["en"] + [l for l in TARGETS] + ["ur", "ne", "other"]
        M = np.zeros((len(langs), len(cols)))
        for i, l in enumerate(langs):
            dist = (D.get(m, l, "eval").get("response_gamma1") or {}).get("response_lang_dist") or {}
            tot = sum(dist.values()) or 1
            for k, v in dist.items():
                M[i, cols.index(k) if k in cols else cols.index("other")] += v / tot
        fig, ax = plt.subplots(figsize=(1 + 0.55 * len(cols), 0.9 + 0.42 * len(langs)),
                               constrained_layout=True)
        im = ax.imshow(M * 100, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.55 else INK)
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels([f"→ {LANG_NAME[l]}" for l in langs])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([LANG_NAME.get(c, c.capitalize()) for c in cols], rotation=45, ha="right")
        ax.set_xlabel("Detected response language")
        ax.set_ylabel("Steering target")
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.01).set_label("% of responses", color=MUTED)
        ax.set_title(f"{m}: response languages under steering (held-out weak prompts)")
        save(fig, out, f"fig09_leakage_matrix_{m}",
             [[m, l, c, M[i, j]] for i, l in enumerate(langs) for j, c in enumerate(cols)],
             ["model", "target", "response_lang", "share"])


def fig10(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "eval") or D.get(m, l, "art") for l in D.langs)]
    if not ms:
        return skip("fig10_conflict_gate", "no eval/artifact json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    rows = []
    for s, m in enumerate(ms):
        langs = D.langs
        for g, mfc in (("gamma0", "none"), ("gamma1", SLOTS[s % 4])):
            xs, ys = [], []
            for i, l in enumerate(langs):
                c = ((D.get(m, l, "eval") or {}).get("conflict_english") or {})
                if g in c:
                    xs.append(i + (s - 1.5) * 0.15)
                    ys.append(100 * c[g])
                    rows.append([m, l, f"conflict_{g}", c[g]])
            if xs:
                axes[0].scatter(xs, ys, marker=MARKS[s % 4], s=30, facecolors=mfc,
                                edgecolors=SLOTS[s % 4], lw=1.3,
                                label=f"{m} ({'unedited' if g == 'gamma0' else 'steered'})")
        xs, ys = [], []
        for i, l in enumerate(langs):
            gt = ((D.get(m, l, "art") or {}).get("gate") or {}).get("heldout_acc")
            if gt is not None:
                xs.append(i + (s - 1.5) * 0.15)
                ys.append(100 * gt)
                rows.append([m, l, "gate_heldout_acc", gt])
        if xs:
            axes[1].scatter(xs, ys, marker=MARKS[s % 4], color=SLOTS[s % 4], s=30, label=m)
    for ax, lab in zip(axes, ["% English kept when the user asks for English",
                              "Gate accuracy on held-out templates (%)"]):
        lang_ticks(ax, D.langs)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 102)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    axes[0].set_title("Conflict test: hollow = unedited, filled = steered")
    axes[1].set_title("Explicit-instruction gate")
    save(fig, out, "fig10_conflict_gate", rows, ["model", "lang", "metric", "value"])


def _grid_stat(D, m, field, xkey, fix_key, fix_val_fn):
    """median/IQR across languages of `field` vs xkey for the full method."""
    per = {}
    for l in D.langs:
        g = [r for r in (D.get(m, l, "grid") or []) if r["variant"] == "full"]
        art = D.get(m, l, "art") or {}
        if not g or not art:
            continue
        fv = fix_val_fn(art)
        for r in g:
            if abs(f(r[fix_key]) - fv) < 1e-9:
                per.setdefault(f(r[xkey]), []).append(f(r[field]))
    xs = sorted(per)
    return (np.array(xs), np.array([np.median(per[x]) for x in xs]),
            np.array([np.percentile(per[x], 25) for x in xs]), np.array([np.percentile(per[x], 75) for x in xs]))


def fig11(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "grid") for l in D.langs)]
    if not ms:
        return skip("fig11_dose_response", "no grid")
    panels = [("gain_DEFAULT", "DEFAULT gain"), ("util_nll", "Utility cost"), ("kl_offdecision", "Off-decision KL")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    rows = []
    for s, m in enumerate(ms):
        for ax, (field, lab) in zip(axes, panels):
            x, med, q1, q3 = _grid_stat(D, m, field, "lam", "beta", lambda a: a["config"]["beta"])
            if not len(x):
                continue
            ax.fill_between(x, q1, q3, color=SLOTS[s % 4], alpha=0.15, lw=0)
            ax.plot(x, med, color=SLOTS[s % 4], marker=MARKS[s % 4], lw=2, ms=5, label=m)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("λ (total strength)")
            ax.set_ylabel(lab)
            rows += [[m, field, a, b] for a, b in zip(x, med)]
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Dose-response at the selected β (median over languages, IQR band)")
    save(fig, out, "fig11_dose_response", rows, ["model", "metric", "lam", "median"])


def fig12(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "grid") for l in D.langs)]
    if not ms:
        return skip("fig12_beta_interaction", "no grid")
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 3.3), sharey=True, squeeze=False,
                             constrained_layout=True)
    rows = []
    for ax, m in zip(axes[0], ms):
        for s, (lab, fn) in enumerate([("selected λ", lambda a: a["config"]["lam"]),
                                       ("selected λ / 2", lambda a: a["config"]["lam"] / 2)]):
            x, med, q1, q3 = _grid_stat(D, m, "gain_DEFAULT", "beta", "lam", fn)
            if not len(x):
                continue
            ax.fill_between(x, q1, q3, color=SLOTS[s], alpha=0.15, lw=0)
            ax.plot(x, med, color=SLOTS[s], marker=MARKS[s], lw=2, ms=5, label=lab)
            rows += [[m, lab, a, b] for a, b in zip(x, med)]
        vals = [f(r["gain_DEFAULT"]) for l in D.langs for r in (D.get(m, l, "grid") or [])
                if r["variant"] == "neg_only" and f(r["beta"]) == 1.0]
        if vals:
            ax.scatter([1.0], [np.median(vals)], marker=MMRK["neg_only"], color=MCOL["neg_only"], s=40,
                       zorder=4, label="Suppression only (β=1)")
        ax.set_xlabel("β (English suppression)")
        ax.set_title(m)
    axes[0, 0].set_ylabel("DEFAULT gain (dev)")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("English suppression only works together with promotion")
    save(fig, out, "fig12_beta_interaction", rows, ["model", "series", "beta", "median_gain_DEFAULT"])


def fig13(D, out):
    """Change (steered - unedited, same weights) per language, on a zero-centred axis with a
    floor on the range, so that small noise is not drawn as a large effect."""
    ms = D.models
    empty = {"target/gamma0": None, "target/gamma1": None, "english/gamma0": None, "english/gamma1": None}
    metrics = [("translation", "Δ FLORES+ chrF++ (eng→target)", 3.0,
                lambda x: ((x.get("eval") or {}).get("translation_chrf++") or {})),
               ("target", "Δ accuracy, target-language tasks", 0.03,
                lambda x: {"gamma0": x["lm"]["target/gamma0"], "gamma1": x["lm"]["target/gamma1"]}),
               ("english", "Δ accuracy, English tasks", 0.03,
                lambda x: {"gamma0": x["lm"]["english/gamma0"], "gamma1": x["lm"]["english/gamma1"]})]
    have = [mt for mt in metrics if any(mt[3](D.d[k]).get("gamma1") is not None for k in D.d)]
    if not have:
        return skip("fig13_benchmarks", "no translation or lm-eval results")
    fig, axes = plt.subplots(len(have), len(ms), figsize=(3.6 * len(ms), 2.6 * len(have)),
                             squeeze=False, sharey="row", constrained_layout=True)
    rows = []
    for i, (k, lab, floor, g) in enumerate(have):
        lim = floor
        for j, m in enumerate(ms):
            ax = axes[i, j]
            xs, ds = [], []
            for x, l in enumerate(D.langs):
                v = g(D.d.get((m, l), {"eval": None, "lm": empty}))
                a, b = v.get("gamma0"), v.get("gamma1")
                if a is None or b is None:
                    continue
                xs.append(x)
                ds.append(b - a)
                rows.append([m, l, k, a, b, b - a])
            if ds:
                ax.vlines(xs, 0, ds, color=MCOL["full"], lw=2)
                ax.scatter(xs, ds, color=MCOL["full"], s=24, zorder=3)
                lim = max(lim, 1.15 * max(abs(d) for d in ds))
            ax.axhline(0, color=MUTED, lw=1)
            lang_ticks(ax, D.langs)
            ax.set_title(m if i == 0 else "")
            if j == 0:
                ax.set_ylabel(lab)
        for ax in axes[i]:
            ax.set_ylim(-lim, lim)
    fig.suptitle("Utility change from steering (steered − unedited, same weights; 0 = no change)")
    save(fig, out, "fig13_benchmarks", rows, ["model", "lang", "metric", "unedited", "foxp2", "delta"])


def fig14(D, out):
    ms = [m for m in D.models if any(D.get(m, l, "s2") for l in D.langs)]
    if not ms:
        return skip("fig14_windows", "no stage2.json")
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 3.6), sharey=True, squeeze=False,
                             constrained_layout=True)
    rows = []
    for ax, m in zip(axes[0], ms):
        n = MODELS[m].n_layers
        for i, l in enumerate(D.langs):
            s2 = D.get(m, l, "s2")
            if not s2 or not s2.get("window"):
                continue
            W = s2["window"]
            ax.barh(i, (len(W)) / n, left=W[0] / n, height=0.6, color=MCOL["full"], zorder=3)
            rows.append([m, l, W[0], W[-1], n])
        ax.axvspan(0, 1 / 3, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.axvspan(2 / 3, 1, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.set_xlim(0, 1)
        ax.set_xlabel("Relative depth (shaded = first / last third)")
        ax.set_yticks(range(len(D.langs)))
        ax.set_yticklabels([LANG_NAME[l] for l in D.langs])
        ax.set_title(f"{m} ({n} layers)")
    fig.suptitle("Selected intervention windows")
    save(fig, out, "fig14_windows", rows, ["model", "lang", "first_layer", "last_layer", "n_layers"])


def fig15(D, out):
    def g(m, l):
        s1 = D.get(m, l, "s1") or {}
        rep = (s1.get("meta") or {}).get("report") or {}
        return {k: v.get("mean_target_script_frac") for k, v in rep.items()}
    _depth_fig(D, out, "fig15_feature_script", [(g, "Share of top-10 tokens in target script")],
               "Selected target features: script detectors (high) vs. language-level features (low)")


def tables(D, out):
    """Main LaTeX table: held-out DEFAULT gain [95% CI] and utility cost for FOXP2 / CAA / prompting."""
    os.makedirs(out, exist_ok=True)
    lines = [r"\begin{tabular}{ll" + "c" * 3 + "}", r"\toprule",
             r"Model & Language & FOXP2 & CAA & Prompting \\", r"\midrule"]
    rows = []
    for m in D.models:
        for l in D.langs:
            if not D.get(m, l, "held"):
                continue
            cells = []
            for k in ("full", "caa", "prompt"):
                v, c = _held(D, m, l, k)
                u, _ = _held(D, m, l, k, "util_nll")
                cells.append(f"{v:.2f} [{c[0]:.2f}, {c[1]:.2f}] / {u:.3f}" if np.isfinite(v) else "--")
                rows.append([m, l, k, v, c[0], c[1], u])
            lines.append(f"{m} & {LANG_NAME[l]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if rows:
        open(os.path.join(out, "table_main_heldout.tex"), "w").write("\n".join(lines) + "\n")
        with open(os.path.join(out, "table_main_heldout.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "lang", "method", "gain_DEFAULT", "ci_lo", "ci_hi", "util_nll"])
            w.writerows(rows)
        print("[table] table_main_heldout.tex / .csv  (cell = DEFAULT gain [95% CI] / utility cost)")
    else:
        skip("table_main_heldout", "no stage3_heldout.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ckpts", default="checkpoints")
    ap.add_argument("--lmeval", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default=None, help="comma list, e.g. fig01,fig06")
    args = ap.parse_args()
    style()
    D = Data(args.runs, args.ckpts, args.lmeval)
    print(f"models: {D.models}\nlanguages: {D.langs}\n")
    fns = {k: globals()[k] for k in FIGURES}
    for k, fn in fns.items():
        if args.only and k not in args.only.split(","):
            continue
        fn(D, args.out)
    tables(D, args.out)
    with open(os.path.join(args.out, "README.txt"), "w") as fh:
        for k, v in FIGURES.items():
            fh.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
