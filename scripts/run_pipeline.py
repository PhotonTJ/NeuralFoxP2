#!/usr/bin/env python
"""Run Neural FOXP2 Stages I-III for every (model, target language) and export steered checkpoints.

Examples
    # everything in the paper's grid (4 models x 5 languages)
    python scripts/run_pipeline.py --models all --langs all --out runs --export_dir checkpoints

    # one pair, quick settings for a first look
    python scripts/run_pipeline.py --models llama31-8b --langs hi --fast

Each stage caches its output under runs/<model>/<lang>/ and is skipped on re-run unless
--force is given.  FLORES+ is gated: accept the terms on the dataset page and log in first.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from foxp2.config import MODELS, TARGETS, FOXP2Config, leak_langs, resolve_langs   # noqa: E402
from foxp2.data import build_lang_data, build_token_sets            # noqa: E402
from foxp2.export import export_checkpoint                          # noqa: E402
from foxp2.metrics import TokenSets                                 # noqa: E402
from foxp2.model_io import load_model                               # noqa: E402
from foxp2.saes import load_sae                                     # noqa: E402
from foxp2.stage1 import run_stage1                                 # noqa: E402
from foxp2.stage2 import DevBench, run_stage2                       # noqa: E402
from foxp2.stage3 import build_steerer, run_stage3                  # noqa: E402


def parse_layers(s, n):
    if not s:
        return None
    a, b = s.split("-")
    return list(range(int(a), min(int(b), n - 1) + 1))


def fast(cfg: FOXP2Config) -> FOXP2Config:
    cfg.n_screen, cfg.n_verify, cfg.n_verify_prompts = 128, 16, 16
    cfg.n_boot, cfg.n_dev_prompts, cfg.n_lift_prompts = 50, 64, 48
    cfg.lam_grid, cfg.beta_grid = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0), (0.0, 0.5, 1.0)
    cfg.caa_lam_grid = (0.5, 1.0, 1.5, 2.0, 3.0)
    cfg.n_window_candidates = 4
    cfg.n_heldout_prompts = 128
    return cfg


def pair_done(args, spec, lang) -> bool:
    out = os.path.join(args.out, spec.key, lang)
    need = [os.path.join(out, f) for f in ("artifact.pt", "stage3_heldout.json")]
    if args.export_dir:
        need.append(os.path.join(args.export_dir, f"{spec.key}-foxp2-{lang}", "foxp2.safetensors"))
    return all(os.path.exists(p) for p in need)


def run_pair(model, tok, spec, lang, cfg, args):
    out = os.path.join(args.out, spec.key, lang)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    others = leak_langs(lang)
    ldata = build_lang_data(lang, cfg, extra_langs=others, cache_dir=args.hf_cache,
                            qc_model=args.qc_model)
    ts_path = os.path.join(out, "token_sets.json")
    if os.path.exists(ts_path) and not args.force:
        sets = json.load(open(ts_path))
    else:
        sets = build_token_sets(tok, ldata.texts, cfg.token_ratio, cfg.token_min_count)
        json.dump(sets, open(ts_path, "w"))
    print({l: len(v) for l, v in sets.items()}, "diagnostic tokens per language")
    ts = TokenSets(sets, len(tok))

    s1_path = os.path.join(out, "stage1.pt")
    if os.path.exists(s1_path) and not args.force:
        s1 = torch.load(s1_path, weights_only=False)
    else:
        s1 = run_stage1(model, tok, spec, ldata, ts, cfg, lambda l: load_sae(spec, l), out)

    bench = DevBench(model, tok, spec, ldata, ts, cfg, others)
    d = spec.d_model

    def make_steerer(W, r, lam, beta, diag):
        return build_steerer(model, s1, diag, W, r, lam, beta, cfg, d, "full", None)

    s2_path = os.path.join(out, "stage2.pt")
    if os.path.exists(s2_path) and not args.force:
        s2 = torch.load(s2_path, weights_only=False)
    else:
        s2 = run_stage2(model, tok, spec, ldata, ts, cfg, s1, out, bench, make_steerer)
        torch.save(s2, s2_path)

    artifact, _ = run_stage3(model, tok, spec, ldata, cfg, s1, s2, out, bench, d,
                             run_ablations=not args.skip_ablations)
    if not artifact["config"]["feasible"]:
        print("!" * 80 + f"\n[stage3] {spec.key}/{lang}: NO operating point satisfies the guardrails.\n"
              "The exported checkpoint (if any) is the best trade-off, flagged feasible=false in its\n"
              "manifest.  Do not report it as a guardrail-compliant result.\n" + "!" * 80)
    if args.export_dir and (artifact["config"]["feasible"] or not args.require_feasible):
        ck = os.path.join(args.export_dir, f"{spec.key}-foxp2-{lang}")
        export_checkpoint(artifact, spec.hf_id, ck, spec.arch, copy=args.copy_weights,
                          extra_manifest={"stage1_report": s1["meta"].get("report", {}),
                                          "token_set_sizes": {l: len(v) for l, v in sets.items()}},
                          token_sets=sets)
        print(f"[export] {ck}")
    print(f"[done] {spec.key}/{lang} in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="comma list of " + ",".join(MODELS))
    ap.add_argument("--langs", default="all",
                    help="all | core (paper's 5) | ext (sw,am,ar,mr,de,fr) | comma list of " + ",".join(TARGETS))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--export_dir", default="checkpoints")
    ap.add_argument("--copy_weights", action="store_true", help="copy base shards instead of symlinking")
    ap.add_argument("--contrast", default="response", choices=["response", "input"])
    ap.add_argument("--lift_mode", default="verify", choices=["attribution", "verify", "intervention"])
    ap.add_argument("--mode", default="decode_window", choices=["decode_window", "all"])
    ap.add_argument("--k_decode", type=int, default=8)
    ap.add_argument("--subspace", default="uncentered", choices=["uncentered", "mean"])
    ap.add_argument("--layers", default=None, help="e.g. 4-28 (default: all layers)")
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--skip_ablations", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--qc_model", default=None,
                    help="optional NLI model for bidirectional-entailment QC, e.g. "
                         "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    ap.add_argument("--hf_cache", default=None)
    ap.add_argument("--force", action="store_true", help="redo every stage, including Stage I")
    ap.add_argument("--redo_from", default=None, choices=["stage2", "stage3"],
                    help="keep cached Stage I (and Stage II for stage3) and redo the rest for every "
                         "selected pair, e.g. after a code fix in the later stages")
    ap.add_argument("--guardrail", default="prompt", choices=["prompt", "fixed"],
                    help="prompt: steering may cost no more than instructing the language (default); "
                         "fixed: the paper's eps (leak .08, KL .08, util .03)")
    ap.add_argument("--require_feasible", action="store_true",
                    help="skip export when no operating point satisfies the guardrails")
    args = ap.parse_args()

    models = list(MODELS) if args.models == "all" else args.models.split(",")
    langs = resolve_langs(args.langs)
    for mk in models:
        spec = MODELS[mk]
        cfg = FOXP2Config(contrast=args.contrast, lift_mode=args.lift_mode, mode=args.mode,
                          k_decode=args.k_decode, subspace=args.subspace,
                          use_gate=not args.no_gate, batch_size=args.batch_size,
                          guardrail=args.guardrail,
                          layers=parse_layers(args.layers, spec.n_layers))
        if args.fast:
            fast(cfg)
        if args.redo_from:
            for l in langs:
                rd = os.path.join(args.out, spec.key, l)
                drop = ["artifact.pt", "stage3_heldout.json"] + (["stage2.pt"] if args.redo_from == "stage2" else [])
                for fn in drop:
                    if os.path.exists(os.path.join(rd, fn)):
                        os.remove(os.path.join(rd, fn))
        todo = [l for l in langs if args.force or args.redo_from or not pair_done(args, spec, l)]
        for l in sorted(set(langs) - set(todo)):
            print(f"[resume] {mk}/{l} already complete, skipping (use --force to redo)")
        if not todo:
            continue
        model, tok = load_model(spec)
        for lang in todo:
            run_pair(model, tok, spec, lang, cfg, args)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
