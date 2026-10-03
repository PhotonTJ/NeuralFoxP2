#!/usr/bin/env python
"""End-to-end smoke test on a tiny random Llama with random JumpReLU and TopK SAEs.

Runs Stages I-III, exports a steered checkpoint with symlinked weights, reloads it with
AutoModelForCausalLM(trust_remote_code=True), and checks:
  * gamma = 0 reproduces the base model's logits exactly
  * gamma = 1 changes only the steered positions in decode_window mode
  * generate() works with the hooks and the KV cache
  * the explicit-instruction gate can switch steering off per row
No network access or GPU is needed.  Numbers are meaningless (random model); this only
checks the plumbing.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from foxp2.config import FOXP2Config, ModelSpec, leak_langs     # noqa: E402
from foxp2.data import LangData, Unit, build_token_sets          # noqa: E402
from foxp2.export import export_checkpoint                       # noqa: E402
from foxp2.metrics import TokenSets                              # noqa: E402
from foxp2.model_io import decoder_layers, prepare_tokenizer     # noqa: E402
from foxp2.saes import SAE                                       # noqa: E402
from foxp2.stage1 import run_stage1                              # noqa: E402
from foxp2.stage2 import DevBench, run_stage2                    # noqa: E402
from foxp2.stage3 import build_steerer, run_stage3               # noqa: E402

TARGET = os.environ.get("SMOKE_TARGET", "hi")   # any language key in ALPH

ALPH = {
    "en": "abcdefghijklmnopqrstuvwxyz", "es": "abcdefghijklmnopqrstuvwxyzñáéíóú",
    "hi": "कखगघचछजझटठडढणतथदधनपफबभमयरलवशसह", "mr": "कखगघचछजझटठडढणतथदधनपळ",
    "ne": "कखगघचछजझटठडढणतथदधनप", "bn": "কখগঘচছজঝটঠডঢণতথদধনপ", "te": "కఖగఘచఛజఝటఠడఢణతథదధనప",
    "zh": "的一是不了人我在有他这中大来上国个到说们", "ur": "ابتثجحخدذرزسشصضطظعغفٹڈ",
    "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي", "sw": "abcdefghijklmnoprstuvwyz",
    "am": "ሀለሐመሠረሰሸቀበተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ",
    "de": "abcdefghijklmnopqrstuvwxyzäöüß", "fr": "abcdefghijklmnopqrstuvwxyzàâçèéêëîôùû",
}


def words(lang, rng, n):
    a = ALPH[lang]
    return " ".join("".join(rng.choice(a) for _ in range(rng.randint(2, 6))) for _ in range(n))


def make_tokenizer(tmp):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    rng = random.Random(0)
    corpus = [words(l, rng, 30) for l in ALPH for _ in range(200)]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=900, special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<u>", "</u>", "<a>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  bos_token="<s>", eos_token="</s>")
    tok.chat_template = ("{{ bos_token }}{% for m in messages %}<u>{{ m['content'] }}</u>{% endfor %}"
                         "{% if add_generation_prompt %}<a>{% endif %}")
    return prepare_tokenizer(tok)


def make_data(rng, n_disc=40, n_dev=24, n_eval=24):
    def unit(i):
        en = "Topic " + words("en", rng, 8).title()
        return Unit(str(i), en, words(TARGET, rng, 8), "news",
                    "wikinews" if i % 2 else "wikivoyage", ", ".join(en.split()[:3]))
    us = [unit(i) for i in range(n_disc + n_dev + n_eval)]
    disc, dev, ev = us[:n_disc], us[n_disc:n_disc + n_dev], us[n_disc + n_dev:]
    texts = {"en": [u.en for u in disc], TARGET: [u.tgt for u in disc]}
    for l in leak_langs(TARGET):
        texts[l] = [words(l, rng, 8) for _ in disc]
    return LangData(TARGET, disc, dev, ev, texts)


def fake_sae(d, m, act, seed):
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(m, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    return SAE(W_enc=W_dec.T.clone() * 2, b_enc=torch.zeros(m), W_dec=W_dec, b_dec=torch.zeros(d),
               act=act, threshold=torch.full((m,), 0.05) if act == "jumprelu" else None,
               topk=16 if act == "topk" else None, meta={"family": "fake"})


def main():
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="foxp2_smoke_")
    tok = make_tokenizer(tmp)
    d, nL = 64, 6
    mcfg = LlamaConfig(vocab_size=len(tok), hidden_size=d, intermediate_size=128,
                       num_hidden_layers=nL, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
                       bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    model = LlamaForCausalLM(mcfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_dir = os.path.join(tmp, "base")
    model.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)

    spec = ModelSpec("tiny", base_dir, "llama", nL, d, "fake", "none")
    cfg = FOXP2Config(n_screen=32, n_verify=4, n_verify_prompts=8, K_tgt=8, K_en=6,
                      n_lift_prompts=16, n_dev_prompts=16, n_boot=10, stab_min=0.0,
                      min_width=2, max_width=3, n_window_candidates=2, lam_grid=(1.0, 2.0),
                      beta_grid=(0.0, 1.0), batch_size=8, token_min_count=1)
    ld = make_data(rng)
    if os.environ.get("SMOKE_INFEASIBLE"):      # exercise the no-feasible-point path
        cfg.guardrail, cfg.eps_leak, cfg.eps_kl, cfg.eps_util = "fixed", -1.0, -1.0, -1.0
    sets = build_token_sets(tok, ld.texts, cfg.token_ratio, cfg.token_min_count)
    print({l: len(v) for l, v in sets.items()})
    ts = TokenSets(sets, len(tok))
    out = os.path.join(tmp, "run")

    saes = {l: fake_sae(d, 256, "topk" if l % 2 else "jumprelu", l) for l in range(nL)}
    s1 = run_stage1(model, tok, spec, ld, ts, cfg, lambda l: saes[l], out, device="cpu")
    assert s1["layers"], "Stage I selected nothing"
    bench = DevBench(model, tok, spec, ld, ts, cfg, leak_langs(TARGET))

    def make_steerer(W, r, lam, beta, diag):
        return build_steerer(model, s1, diag, W, r, lam, beta, cfg, d, "full", None)

    s2 = run_stage2(model, tok, spec, ld, ts, cfg, s1, out, bench, make_steerer)
    artifact, gate = run_stage3(model, tok, spec, ld, cfg, s1, s2, out, bench, d)
    assert artifact["config"]["feasible"] == (not os.environ.get("SMOKE_INFEASIBLE")), artifact["config"]
    ck = export_checkpoint(artifact, base_dir, os.path.join(tmp, "ckpt"), "llama", token_sets=sets)
    print("exported files:", sorted(os.listdir(ck)))
    assert os.path.islink(os.path.join(ck, "model-base.safetensors"))

    m2 = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True).eval()
    st = m2.foxp2_steer
    print("steer summary:", st.summary())
    assert torch.allclose(st.v_pos.float(), artifact["state_dict"]["v_pos"].float()), "buffers not loaded"
    # force a strong, non-zero edit so the assertions below are informative
    with torch.no_grad():
        st.v_pos.add_(torch.randn_like(st.v_pos))
    st.invalidate()
    st.use_gate = False

    base = LlamaForCausalLM.from_pretrained(base_dir).eval()
    ids = tok(["<s><u>hello world</u><a>", "<s><u>abc</u><a>"], return_tensors="pt",
              padding=True, add_special_tokens=False)
    with torch.no_grad():
        ref = base(**ids).logits
        st.gamma = 0.0
        same = m2(**ids).logits
        st.gamma = 1.0
        edited = m2(**ids).logits
    assert torch.equal(ref, same), "gamma=0 must be bit-identical to the base model"
    diff = (edited - ref).abs().amax(-1)
    assert diff[:, -1].min() > 0, "prompt-final position must change"
    assert diff[:, :-1].max() == 0, "decode_window mode must leave earlier prompt positions alone"
    print("gamma=0 identical; gamma=1 edits only the prompt-final position in prefill")

    with torch.no_grad():
        g0 = base.generate(**ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        g1 = m2.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
    print("generate ok:", g0.shape, g1.shape, "steps seen:", st._step)
    assert st._step == 11, st._step

    # gate: a probe that fires on row 0 only switches steering off for that row
    with torch.no_grad():
        st.use_gate = True
        st.gate_w.zero_()
        st.gate_b.zero_()
        h_probe = {}
        hd = decoder_layers(m2)[st.layers[0]].register_forward_hook(
            lambda m, a, o: h_probe.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].clone()))
        st.gamma = 0.0
        m2(**ids)
        hd.remove()
        h = h_probe["h"].float()
        w = h[0] - h[1]
        st.gate_w.copy_(w)
        st.gate_b.fill_(-float(((h[0] + h[1]) / 2) @ w))
        st.invalidate()
        st.gamma = 1.0
        gated = m2(**ids).logits
    assert torch.equal(gated[0], ref[0]), "gated row must be unedited"
    assert (gated[1, -1] - ref[1, -1]).abs().max() > 0, "ungated row must be edited"
    print("gate switches steering off per row")

    os.environ["FOXP2_MODE"] = "all"
    with torch.no_grad():
        st.use_gate = False
        allpos = m2(**ids).logits
    os.environ.pop("FOXP2_MODE")
    assert (allpos - ref).abs().amax(-1)[:, :-1].max() > 0, "'all' mode must edit every position"
    print("FOXP2_MODE=all edits every position")

    # attention-sink positions (huge residual norm) are never edited, in prefill or decoding
    from foxp2.foxp2_steer import FOXP2Steerer
    s = FOXP2Steerer([0], d, 1, mode="all", sink_factor=5.0)
    s.v_pos[0] = 1.0
    s.begin_forward(6, 0)
    h = torch.randn(2, 6, d)
    h[:, 0] *= 100
    o = s.edit(0, h)
    assert torch.equal(o[:, 0], h[:, 0]) and (o[:, 1:] - h[:, 1:]).abs().amin() > 0
    s.begin_forward(1, 6)
    h1 = torch.randn(2, 1, d) * 100
    assert torch.equal(s.edit(0, h1), h1)
    print("attention-sink positions are left untouched")
    shutil.rmtree(tmp)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
