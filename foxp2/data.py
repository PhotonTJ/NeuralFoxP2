"""FLORES+ loading, weak-prompt families, response-language contrast pairs and token sets.

Design (review fixes):
  * Contrast the RESPONSE language with the prompt held fixed (input language is not a
    confound any more).  The original input-language contrast is kept as `contrast="input"`
    for the ablation.
  * Discovery and tuning use FLORES+ `dev`; evaluation uses FLORES+ `devtest`.  Prompt
    templates are also disjoint across disc / dev / eval, so no template or sentence used to
    build the artifact appears at evaluation time.  Downstream FLORES translation scores
    must therefore be computed on `devtest` only.
"""
from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .config import FLORES_CODE, LANG_NAME, SCRIPT

FLORES_REPO = "openlanguagedata/flores_plus"

# --------------------------------------------------------------------------------------
# Prompt families.  None of them names a language; template sets are disjoint per split.
# {kw} = content keywords of the meaning unit, {topic} = FLORES+ topic label.
# --------------------------------------------------------------------------------------
WEAK_TEMPLATES = {
    "disc": ["Tell me about {kw}.", "{kw}?", "One sentence on {kw}, please.",
             "Something about {kw}.", "Quick fact: {kw}", "Write a line about {kw}."],
    "dev": ["{topic}: {kw}", "Thoughts on {kw}?", "Can you say a bit about {kw}?"],
    "eval": ["Short note on {kw}.", "What do you know about {kw}?", "{kw}",
             "A few words about {kw}, please."],
}
# Explicit-instruction families (gate training on dev, conflict test on eval).
EXPLICIT_TEMPLATES = {
    # gate training (dev): varied positions and phrasings; eval phrasings stay unseen
    "dev": ["{p} Answer in {L}.", "Please reply in {L}. {p}", "{p} ({L} please)",
            "{p} Write the answer in {L}.", "Reply using {L}: {p}", "{p}\nLanguage of reply: {L}",
            "Can you answer in {L}? {p}", "{p} -- respond in {L}"],
    "eval": ["In {L}: {p}", "{p} Respond only in {L}.", "Use {L} for your answer. {p}"],
}
EXPLICIT_LANGS = ["English", "Hindi", "Chinese", "Bengali", "Telugu", "Spanish", "Swahili",
                  "Amharic", "Arabic", "Marathi", "French", "German", "Japanese"]

_STOP = set("""a an the and or but of to in on for with at by from as is are was were be been
being this that these those it its into over under than then there their they them he she his
her we our you your i me my not no yes also which who whom whose what when where why how all any
some such more most other can could would should will may might must have has had do does did
about after before during while because said says""".split())


def keywords(text: str, k: int = 4) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text)
    cands = [w for w in words if w.lower() not in _STOP and len(w) >= 4]
    seen, out = set(), []
    for w in sorted(cands, key=lambda w: (-(w[0].isupper()), -len(w))):
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
        if len(out) == k:
            break
    out = out or words[:k]
    return ", ".join(out)


# --------------------------------------------------------------------------------------
# FLORES+
# --------------------------------------------------------------------------------------
def load_flores(lang: str, split: str, cache_dir: str | None = None) -> list[dict]:
    """FLORES+ is gated: accept the terms on the dataset page and `huggingface-cli login`."""
    from datasets import load_dataset
    ds = load_dataset(FLORES_REPO, FLORES_CODE[lang], split=split, cache_dir=cache_dir)
    rows = []
    for r in ds:
        if r.get("variant", "") not in ("", None):
            continue
        rows.append({"id": str(r["id"]), "text": r["text"], "topic": r.get("topic", "") or "",
                     "domain": r.get("domain", "") or ""})
    return rows


@dataclass
class Unit:
    """One meaning unit: aligned English / target text plus metadata."""
    id: str
    en: str
    tgt: str
    topic: str
    domain: str
    kw: str


@dataclass
class LangData:
    target: str
    disc: list[Unit]
    dev: list[Unit]
    eval: list[Unit]
    texts: dict[str, list[str]] = field(default_factory=dict)   # discovery texts per language


def _align(en_rows, tgt_rows) -> list[Unit]:
    tgt_by_id = {r["id"]: r for r in tgt_rows}
    units = []
    for r in en_rows:
        t = tgt_by_id.get(r["id"])
        if t is None:
            continue
        units.append(Unit(r["id"], r["text"], t["text"], r["topic"], r["domain"], keywords(r["text"])))
    return units


def build_lang_data(target: str, cfg, extra_langs: list[str], cache_dir: str | None = None,
                    qc_model: str | None = None) -> LangData:
    rng = random.Random(cfg.seed)
    en_dev, en_test = load_flores("en", "dev", cache_dir), load_flores("en", "devtest", cache_dir)
    t_dev, t_test = load_flores(target, "dev", cache_dir), load_flores(target, "devtest", cache_dir)
    dev_units, test_units = _align(en_dev, t_dev), _align(en_test, t_test)
    if qc_model:
        dev_units = entailment_filter(dev_units, qc_model)
        test_units = entailment_filter(test_units, qc_model)
    rng.shuffle(dev_units)
    rng.shuffle(test_units)
    disc = dev_units[: cfg.n_disc]
    dev = dev_units[cfg.n_disc: cfg.n_disc + cfg.n_dev]
    ev = test_units[: cfg.n_eval]
    disc_ids = {u.id for u in disc}
    texts = {"en": [u.en for u in disc], target: [u.tgt for u in disc]}
    for l in extra_langs:
        if l in texts:
            continue
        rows = load_flores(l, "dev", cache_dir)
        texts[l] = [r["text"] for r in rows if r["id"] in disc_ids]
    return LangData(target, disc, dev, ev, texts)


def entailment_filter(units: list[Unit], model_name: str, tau: float = 0.85) -> list[Unit]:
    """Optional bidirectional-entailment QC with a multilingual NLI model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    ent = [i for i, l in m.config.id2label.items() if "entail" in l.lower()][0]
    keep = []
    with torch.no_grad():
        for u in units:
            enc = tok([u.en, u.tgt], [u.tgt, u.en], return_tensors="pt", padding=True, truncation=True)
            p = m(**enc).logits.softmax(-1)[:, ent]
            if bool((p >= tau).all()):
                keep.append(u)
    print(f"[qc] kept {len(keep)}/{len(units)} pairs at tau={tau}")
    return keep


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
def weak_prompt(u: Unit, split: str, i: int) -> str:
    tpls = WEAK_TEMPLATES[split]
    return tpls[i % len(tpls)].format(kw=u.kw, topic=u.topic or "news")


def weak_prompts(units: list[Unit], split: str, n: int | None = None) -> list[str]:
    units = units if n is None else units[:n]
    return [weak_prompt(u, split, i) for i, u in enumerate(units)]


def explicit_prompts(units: list[Unit], split: str, langs=None, n: int | None = None,
                     seed: int = 0) -> list[tuple[str, str]]:
    """Returns (prompt, instructed language name)."""
    rng = random.Random(seed)
    langs = langs or EXPLICIT_LANGS
    base = weak_prompts(units, "dev" if split == "dev" else "eval", n)
    tpls = EXPLICIT_TEMPLATES[split]
    out = []
    for i, p in enumerate(base):
        L = rng.choice(langs)
        out.append((tpls[i % len(tpls)].format(p=p, L=L), L))
    return out


@dataclass
class ContrastPair:
    prompt_en: str      # user turn used with the English realization
    prompt_tgt: str     # user turn used with the target realization (== prompt_en for "response")
    resp_en: str
    resp_tgt: str
    domain: str


def contrast_pairs(units: list[Unit], contrast: str) -> list[ContrastPair]:
    pairs = []
    for i, u in enumerate(units):
        if contrast == "response":
            p = weak_prompt(u, "disc", i)
            pairs.append(ContrastPair(p, p, u.en, u.tgt, u.domain))
        elif contrast == "input":
            pairs.append(ContrastPair(u.en, u.tgt, "", "", u.domain))
        else:
            raise ValueError(contrast)
    return pairs


# --------------------------------------------------------------------------------------
# Token sets (token-id space, pinned to the tokenizer).  w(u)=1 diagnostic, 0 otherwise.
# --------------------------------------------------------------------------------------
_RANGES = {
    "Deva": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Beng": [(0x0980, 0x09FF)],
    "Telu": [(0x0C00, 0x0C7F)],
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Hani": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "Ethi": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
    "Latn": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}


def char_script(ch: str) -> str | None:
    cp = ord(ch)
    for s, rs in _RANGES.items():
        if any(a <= cp <= b for a, b in rs):
            return s
    return None


def token_scripts(s: str) -> set[str]:
    out = set()
    for ch in s:
        if unicodedata.category(ch)[0] in ("L", "M"):
            out.add(char_script(ch) or "Other")
    return out


def build_token_sets(tokenizer, texts: dict[str, list[str]], ratio: float = 4.0,
                     min_count: int = 3) -> dict[str, list[int]]:
    """One-vs-rest diagnostic token sets for every language in `texts`.

    A token is diagnostic for language a if (i) it is frequent in a and >= `ratio` times more
    frequent (per token) in a than in any other language, or (ii) all its letters are in a
    script used by no other language under consideration.  Digits, punctuation, whitespace and
    tokens without letters are in the shared pool (weight 0) for every language.
    """
    langs = list(texts)
    counts, totals = {}, {}
    for l in langs:
        c = Counter()
        for t in texts[l]:
            c.update(tokenizer(t, add_special_tokens=False)["input_ids"])
        counts[l], totals[l] = c, max(1, sum(c.values()))
    script_users = Counter(SCRIPT[l] for l in langs)
    vocab = len(tokenizer)
    decoded = {}

    def dec(i):
        if i not in decoded:
            try:
                decoded[i] = tokenizer.decode([i])
            except Exception:
                decoded[i] = ""
        return decoded[i]

    sets = {l: set() for l in langs}
    all_seen = set().union(*[set(c) for c in counts.values()])
    for u in all_seen:
        if not token_scripts(dec(u)):
            continue            # shared pool: no letters
        freqs = {l: (counts[l][u] + 0.5) / totals[l] for l in langs}
        for a in langs:
            if counts[a][u] < min_count:
                continue
            other = max(freqs[b] for b in langs if b != a)
            if freqs[a] / other >= ratio:
                sets[a].add(u)
    # script rule (covers rare tokens never seen in the sample)
    for u in range(vocab):
        sc = token_scripts(dec(u))
        if len(sc) != 1:
            continue
        s = next(iter(sc))
        owners = [l for l in langs if SCRIPT[l] == s]
        if len(owners) == 1 and script_users[s] == 1:
            sets[owners[0]].add(u)
    # a token may only be diagnostic for one language
    owner = {}
    for l in langs:
        for u in sets[l]:
            owner.setdefault(u, []).append(l)
    return {l: sorted(u for u in sets[l] if len(owner[u]) == 1) for l in langs}
