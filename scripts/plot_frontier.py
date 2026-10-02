#!/usr/bin/env python
"""Gain-vs-cost frontier for FOXP2, its ablations, CAA and plain prompting (one model/language).

    python scripts/plot_frontier.py runs/llama31-8b/hi            # -> frontier.png/.pdf + frontier_table.csv

Two panels share the y-axis (DEFAULT gain on D_dev): x = in-language utility cost (NLL
increase, nats/token, at deployed positions) and x = off-decision KL.  Faded marks are every grid
point; the line is each method's Pareto frontier (best gain at no more cost).  Identity is
carried by colour + marker shape + legend; the exact frontier values are in frontier_table.csv.  The ring marks
the explicit-instruction prompt baseline; dashed lines are the guardrails.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-4 of the reference palette, fixed order; validated (CVD/normal pass,
# contrast WARN for slots 3-4 -> every series is also direct-labelled and has its own marker)
SERIES = [("full", "FOXP2", "#2a78d6", "o"), ("caa", "CAA (residual diff-in-means)", "#eb6834", "s"),
          ("outside_window", "FOXP2, outside window", "#1baf7a", "^"),
          ("pos_only", "FOXP2, no English suppression", "#eda100", "D")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
COSTS = [("util_nll", "Utility cost  (Δ NLL, nats/token)", "util"),
         ("kl_offdecision", "Off-decision KL", "kl")]


def pareto(points):
    """points: [(cost, gain, row)] -> frontier sorted by cost (max gain so far)."""
    out, best = [], -1e9
    for c, g, r in sorted(points, key=lambda p: (p[0], -p[1])):
        if g > best:
            out.append((c, g, r))
            best = g
    return out


def main(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "stage3_grid.csv"))))
    s2 = json.load(open(os.path.join(run_dir, "stage2.json")))
    pref, eps = s2.get("prompt_reference"), s2.get("guardrail_eps", {})
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    table = []
    for ax, (ckey, clabel, ekey) in zip(axes, COSTS):
        for key, label, color, marker in SERIES:
            pts = [(float(r[ckey]), float(r["gain_DEFAULT"]), r) for r in rows if r["variant"] == key]
            if not pts:
                continue
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, color=color, alpha=0.35,
                       marker=marker, linewidths=0, zorder=2)
            fr = pareto(pts)
            ax.plot([p[0] for p in fr], [p[1] for p in fr], color=color, lw=2, marker=marker, ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
            if ckey == "util_nll":
                table += [{"method": label, "lam": p[2]["lam"], "beta": p[2]["beta"],
                           "gain_DEFAULT": p[1], "gain_dM": p[2]["gain_dM"], "util_nll": p[0],
                           "kl_offdecision": p[2]["kl_offdecision"], "leak_max": p[2]["leak_max"],
                           "feasible": p[2]["feasible"]} for p in fr]
        if pref:
            ax.scatter([pref[ckey]], [pref["gain_DEFAULT"]], s=90, facecolors="none", edgecolors=INK,
                       linewidths=1.8, zorder=4, label="Prompt: “Answer in <L>.”")
            ax.annotate("prompting", (pref[ckey], pref["gain_DEFAULT"]), xytext=(6, -12),
                        textcoords="offset points", color=INK, fontsize=8.5)
        if ekey in eps:
            ax.axvline(eps[ekey], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.annotate("guardrail", (eps[ekey], 0.02), xytext=(4, 0), textcoords="offset points",
                        color=MUTED, fontsize=8, rotation=90, va="bottom")
        ax.set_xlabel(clabel)
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
        ticks = [t for t in (-0.1, 0, 0.01, 0.03, 0.1, 0.3, 1, 3) if ckey != "kl_offdecision" or t >= 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        if ckey == "kl_offdecision":
            ax.set_xlim(left=0)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("DEFAULT gain (dev)")
    axes[0].set_ylim(-0.05, 1.05)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, fontsize=8.5)
    meta = os.path.normpath(run_dir).split(os.sep)[-2:]
    fig.suptitle(f"Language steering: gain vs. cost  ({' / '.join(meta)})", color=INK, fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(run_dir, f"frontier.{ext}"), dpi=200)
    with open(os.path.join(run_dir, "frontier_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
        if pref:
            w.writerow({"method": "prompt", "lam": "", "beta": "", "gain_DEFAULT": pref["gain_DEFAULT"],
                        "gain_dM": pref["gain_dM"], "util_nll": pref["util_nll"],
                        "kl_offdecision": pref["kl_offdecision"], "leak_max": pref["leak_max"],
                        "feasible": ""})
    print(f"[saved] {run_dir}/frontier.png, frontier.pdf, frontier_table.csv")


if __name__ == "__main__":
    main(sys.argv[1])
