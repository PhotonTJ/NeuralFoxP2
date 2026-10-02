#!/usr/bin/env python
"""Evaluate an exported FOXP2 checkpoint on held-out data (FLORES+ devtest, eval templates).

Reports, steered (gamma=1) vs unedited (gamma=0, same weights, bit-identical to the base):
  * early defaultness: dM gain, prefix-LID gain, DEFAULT gain, mass leakage per language
  * response level (primary metric): % responses entirely in the target language,
    mid-generation switch-away rate, full leakage matrix of response languages
  * prompt baselines at RESPONSE level: explicit instruction variants incl. the instruction
    written in the target language, for comparison with the steered weak-prompt setting
  * conflict test: weak prompt + "answer in English" style instruction -> % still English
  * optional FLORES+ devtest translation (eng -> target) chrF++ with --translation

    python scripts/evaluate.py --ckpt checkpoints/llama31-8b-foxp2-hi --model llama31-8b --lang hi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from foxp2.config import LANG_NAME, MODELS, leak_langs                    # noqa: E402
from foxp2.data import (EXPLICIT_TEMPLATES, Unit, _align, keywords, load_flores,  # noqa: E402,F401
                        weak_prompts)
from foxp2.metrics import (LID, TokenSets, compare_early, early_defaultness, segment_langs,
                           wilson_ci)  # noqa: E402
from foxp2.model_io import chat_ids, greedy, prepare_tokenizer             # noqa: E402

NATIVE_INSTR = {"hi": "कृपया हिंदी में उत्तर दें।", "zh": "请用中文回答。", "bn": "অনুগ্রহ করে বাংলায় উত্তর দিন।",
                "te": "దయచేసి తెలుగులో సమాధానం ఇవ్వండి.", "es": "Por favor, responde en español.",
                "sw": "Tafadhali jibu kwa Kiswahili.", "am": "እባክዎ በአማርኛ ይመልሱ።",
                "ar": "من فضلك أجب باللغة العربية.", "mr": "कृपया मराठीत उत्तर द्या.",
                "de": "Bitte antworte auf Deutsch.", "fr": "Veuillez répondre en français."}


def set_gamma(model, g):
    model.foxp2_steer.gamma = float(g)


def response_level(model, tok, spec, prompts, target, lid, max_new, bs):
    ids = [chat_ids(tok, p, spec.chat_kwargs) for p in prompts]
    gens, _ = greedy(model, tok, ids, max_new_tokens=max_new, batch_size=bs)
    texts = [tok.decode(g, skip_special_tokens=True) for g in gens]
    whole = [lid(t) for t in texts]
    segs = [segment_langs(t, lid) for t in texts]
    started = [s[0] == target for s in segs]
    switched = [st and any(x != target for x in s[1:]) for st, s in zip(started, segs)]
    k_all = sum(all(x == target for x in s) for s in segs)
    k_whole = sum(w == target for w in whole)
    return {"n": len(texts), "entire_target": k_all / max(1, len(texts)),
            "entire_target_ci": wilson_ci(k_all, len(texts)),
            "whole_lid_target": k_whole / max(1, len(texts)),
            "whole_lid_target_ci": wilson_ci(k_whole, len(texts)),
            "switch_away_rate": float(sum(switched) / max(1, sum(started))),
            "switch_away_ci": wilson_ci(sum(switched), sum(started)),
            "response_lang_dist": dict(Counter(whole)),
            "examples": [{"prompt": p, "response": t[:300]} for p, t in zip(prompts[:5], texts[:5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_glotlid", action="store_true")
    ap.add_argument("--translation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[args.model]
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(args.ckpt))
    kw = {"attn_implementation": spec.attn_implementation} if spec.attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map="auto", **kw).eval()
    summ = model.foxp2_steer.summary()
    assert any(v > 0 for v in summ["v_pos_norm"]), "foxp2 tensors were not loaded"
    print("[steer]", summ)

    lang, others = args.lang, leak_langs(args.lang)
    units = _align(load_flores("en", "devtest"), load_flores(lang, "devtest"))[: args.n]
    weak = weak_prompts(units, "eval")
    # token sets frozen at discovery time travel with the checkpoint
    ts_file = os.path.join(args.ckpt, "foxp2_token_sets.json")
    if not os.path.exists(ts_file):
        raise FileNotFoundError(f"{ts_file} missing: re-export with scripts/run_pipeline.py")
    sets = json.load(open(ts_file))
    ts = TokenSets(sets, len(tok))
    lid = LID(use_glotlid=not args.no_glotlid)
    if lid.model is None and lang in ("mr", "ar"):
        print(f"[warn] without GlotLID '{lang}' cannot be told apart from "
              f"{'Hindi/Nepali' if lang == 'mr' else 'Urdu'}; install fasttext-wheel")
    ids = [chat_ids(tok, p, spec.chat_kwargs) for p in weak]
    res = {"ckpt": args.ckpt, "lang": lang, "n": len(units), "steer": summ}

    set_gamma(model, 0.0)
    b = early_defaultness(model, tok, ids, ts, lang, batch_size=args.batch_size, lid=lid)
    set_gamma(model, 1.0)
    e = early_defaultness(model, tok, ids, ts, lang, batch_size=args.batch_size, lid=lid)
    res["early"] = compare_early(e, b, lang, others)
    print("[early]", {k: v for k, v in res["early"].items() if k != "leak"})

    for g in (0.0, 1.0):
        set_gamma(model, g)
        res[f"response_gamma{int(g)}"] = response_level(model, tok, spec, weak, lang, lid,
                                                        args.max_new, args.batch_size)
        print(f"[response gamma={g}]", {k: v for k, v in res[f"response_gamma{int(g)}"].items()
                                        if k != "examples"})

    # prompting baselines at response level (unedited model)
    set_gamma(model, 0.0)
    L = LANG_NAME[lang]
    strategies = {
        "instr_end": [f"{p} Answer in {L}." for p in weak],
        "instr_start": [f"Answer in {L}. {p}" for p in weak],
        "instr_native": [f"{p} {NATIVE_INSTR[lang]}" for p in weak] if lang in NATIVE_INSTR else None,
    }
    res["prompt_baselines"] = {}
    for name, ps in strategies.items():
        if ps:
            r = response_level(model, tok, spec, ps, lang, lid, args.max_new, args.batch_size)
            res["prompt_baselines"][name] = {k: v for k, v in r.items() if k != "examples"}
            print(f"[prompt:{name}]", res["prompt_baselines"][name])

    # conflict: explicit English instruction must win over steering
    conflict = [t.format(p=p, L="English") for t, p in
                zip(EXPLICIT_TEMPLATES["eval"] * len(weak), weak)]
    res["conflict_english"] = {}
    for g in (0.0, 1.0):
        set_gamma(model, g)
        r = response_level(model, tok, spec, conflict, "en", lid, 64, args.batch_size)
        res["conflict_english"][f"gamma{int(g)}"] = r["whole_lid_target"]
        res["conflict_english"][f"gamma{int(g)}_ci"] = r["whole_lid_target_ci"]
    print("[conflict] % English kept", res["conflict_english"])

    if args.translation:
        import sacrebleu
        src = [f"Translate into {L}:\n{u.en}" for u in units]
        ref = [u.tgt for u in units]
        res["translation_chrf++"] = {}
        for g in (0.0, 1.0):
            set_gamma(model, g)
            gens, _ = greedy(model, tok, [chat_ids(tok, s, spec.chat_kwargs) for s in src], 192,
                             args.batch_size)
            hyp = [tok.decode(x, skip_special_tokens=True).strip().split("\n")[0] for x in gens]
            res["translation_chrf++"][f"gamma{int(g)}"] = sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score
        print("[translation chrF++]", res["translation_chrf++"])

    out = args.out or os.path.join(args.ckpt, f"eval_{lang}.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
