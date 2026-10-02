#!/usr/bin/env python
"""SYNTHETIC outputs in the pipeline's exact file formats, for testing make_figures.py only.

    python tests/fake_results.py /tmp/fake && python scripts/make_figures.py \
        --runs /tmp/fake/runs --ckpts /tmp/fake/checkpoints --lmeval /tmp/fake/results --out /tmp/fake/figures

The numbers are random.  Never put figures made from this data in a paper.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from foxp2.config import MODELS, TARGETS, leak_langs  # noqa: E402

rng = np.random.default_rng(0)
root = sys.argv[1]


def spec(r_eff, top1):
    return {"r_eff": r_eff, "top1_var": top1, "eigengap_idx": 1, "eigengap": 2.0, "top4_var": top1 + .1,
            "spectrum": []}


def compare(g, gd, util, kl, leak):
    lo = lambda x: [x - .05, x + .05]
    return {"n": 300, "gain_dM": g, "gain_dM_ci": lo(g), "gain_DEFAULT": gd, "gain_DEFAULT_ci": lo(gd),
            "gain_lid": gd, "gain_lid_ci": lo(gd), "leak_max": leak, "kl_offdecision": kl,
            "util_nll": util, "util_nll_all": util * 3, "feasible": util < .065, "leak": {}}


for m, sp in MODELS.items():
    n = sp.n_layers
    for li, l in enumerate(TARGETS):
        rd = os.path.join(root, "runs", m, l)
        cd = os.path.join(root, "checkpoints", f"{m}-foxp2-{l}")
        os.makedirs(rd, exist_ok=True)
        os.makedirs(cd, exist_ok=True)
        a = int(n * .25) + rng.integers(-2, 3)
        W = list(range(a, a + int(n * .3)))
        diag = {str(k): {"spectral": spec(30 + 10 * rng.random(), .6 + .1 * rng.random()),
                         "centred": spec(40 + 8 * rng.random(), .1),
                         "null_colshuffle": spec(55 + 5 * rng.random(), .04),
                         "null_random_support": spec(50, .05),
                         "null_domain": spec(20 + 5 * rng.random(), .32),
                         "mean_share": .6, "Pmu_over_mu": .999, "stability": .97,
                         "support_vs_caa_norm": .1 + .5 * k / n + .05 * rng.random(),
                         "support_vs_caa_cos": .25 + .5 * k / n + .05 * rng.random()} for k in range(n)}
        curve = {str(k): float(.03 * np.exp(-((k - a - 5) / 5) ** 2) + (.1 if k == n - 1 else 0)
                               + .003 * rng.random()) for k in range(n)}
        pref = {"gain_dM": .6, "gain_DEFAULT": .7 + .1 * rng.random(), "leak_max": .02,
                "kl_offdecision": .29, "util_nll": .065}
        json.dump({"window": W, "curve": curve, "diagnostics": diag, "prompt_reference": pref,
                   "guardrail_eps": {"leak": .08, "kl": .29, "util": .065}},
                  open(os.path.join(rd, "stage2.json"), "w"))
        rep = {str(k): {"fvu": {"fvu": .15 + .25 * k / n + .05 * rng.random()}, "n_tgt": 64, "n_en": 32,
                        "mean_target_script_frac": .8 - .5 * k / n + .1 * rng.random()} for k in range(n)}
        json.dump({"meta": {"report": rep}}, open(os.path.join(rd, "stage1_report.json"), "w"))
        rows = []
        for v, lams, betas, scale in [("full", (1, 2, 3, 4, 6, 8, 12, 16), (0, .5, 1), 1),
                                      ("caa", (.25, .5, .75, 1, 1.5, 2, 3, 4), (0,), 4),
                                      ("neg_only", (1,), (0, .5, 1), 0)]:
            for lam in lams:
                for b in betas:
                    x = lam * scale / 8
                    gd = float(np.clip(1 / (1 + np.exp(-6 * (x - .6))) * (1 + .1 * b) - .02, 0, 1))
                    rows.append({"variant": v, "lam": lam, "beta": b, "gain_DEFAULT": gd,
                                 "gain_dM": .55 * gd, "util_nll": .02 * x ** 2 * (1.6 if v == "caa" else 1),
                                 "kl_offdecision": .03 * x ** 2, "leak_max": .03 * gd, "feasible": True})
        with open(os.path.join(rd, "stage3_grid.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        held = {"prompt": compare(.6, .72, .065, .29, .02), "full": compare(.52, .85, .03, .04, .03),
                "caa": compare(.55, .88, .045, .06, .03), "pos_only": compare(.5, .75, .025, .03, .03),
                "neg_only": compare(.02, .01, .01, .01, 0), "sparse_only": compare(.52, .85, .03, .04, .03),
                "outside_window": compare(.5, .8, .11, .6, .02), "random": compare(0, 0, .002, 0, 0)}
        for k in held:
            held[k]["gain_DEFAULT"] = float(np.clip(held[k]["gain_DEFAULT"] + .08 * rng.standard_normal(), -0.05, 1))
        json.dump(held, open(os.path.join(rd, "stage3_heldout.json"), "w"))
        json.dump({"config": {"lam": 8.0, "beta": 1.0, "window": W, "feasible": True},
                   "gate": {"heldout_acc": .95 + .05 * rng.random()}}, open(os.path.join(rd, "artifact.json"), "w"))
        p1 = .85 - (.4 if l == "am" else 0) + .05 * rng.random()
        dist = {l: int(300 * p1), "en": int(300 * (1 - p1) * .8), leak_langs(l)[0]: int(300 * (1 - p1) * .2)}
        resp = lambda p: {"entire_target": p, "entire_target_ci": [p - .05, p + .05],
                          "switch_away_rate": (1 - p) * .2, "switch_away_ci": [(1 - p) * .15, (1 - p) * .25],
                          "response_lang_dist": dist}
        json.dump({"response_gamma0": resp(.05), "response_gamma1": resp(p1),
                   "prompt_baselines": {"instr_end": resp(.7), "instr_native": resp(.8)},
                   "conflict_english": {"gamma0": .97, "gamma1": .95},
                   "translation_chrf++": {"gamma0": 40 + 5 * rng.random(), "gamma1": 40 + 5 * rng.random()}},
                  open(os.path.join(cd, f"eval_{l}.json"), "w"))
        for kind in ("target", "english"):
            for g in ("gamma0", "gamma1"):
                d = os.path.join(root, "results", m, l, kind, g, "model")
                os.makedirs(d, exist_ok=True)
                json.dump({"results": {"task": {"acc,none": .6 + .05 * rng.random()}}},
                          open(os.path.join(d, "results_2026.json"), "w"))
print("SYNTHETIC results written to", root)
