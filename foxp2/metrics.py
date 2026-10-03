"""Defaultness, leakage, drift, utility and response-level language identification."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from .data import token_scripts
from .model_io import batches, greedy, input_device, left_pad


# --------------------------------------------------------------------------------------
class TokenSets:
    def __init__(self, sets: dict[str, list[int]], vocab_size: int):
        self.langs = list(sets)
        self.idx = {l: torch.tensor(v, dtype=torch.long) for l, v in sets.items()}
        self.vocab_size = vocab_size

    def masses(self, probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """probs [..., V] -> {lang: [...]} weighted class mass (w=1 diagnostic, 0 otherwise)."""
        out = {}
        for l, ix in self.idx.items():
            ix = ix[ix < probs.shape[-1]].to(probs.device)
            out[l] = probs.index_select(-1, ix).sum(-1)
        return out

    def to_json(self):
        return {l: v.tolist() for l, v in self.idx.items()}


def delta_m_from_logits(logits: torch.Tensor, ts: TokenSets, target: str):
    """logits [B, T, V] -> (dM [B], masses {lang: [B, T]})."""
    p = logits.float().softmax(-1)
    m = ts.masses(p)
    return (m[target] - m["en"]).mean(-1), m


# --------------------------------------------------------------------------------------
# Language identification on generated text.
# --------------------------------------------------------------------------------------
_ES = set("el la los las de que y en un una por para con no se es del al lo como más pero sus "
          "su este esta también fue ha son".split())
_EN = set("the and of to in is that for it with as was on are be by this from at an or have "
          "has were which".split())
_SW = set("na ya wa kwa katika ni za la cha vya kuwa hii pia au lakini kama hiyo yake wake "
          "zaidi sana".split())
_DE = set("der die das und ist nicht ein eine zu den von mit sich des auf für im dem auch es "
          "wird wurde sind".split())
_FR = set("le la les de des et est un une du en que qui dans pour pas sur au avec il elle sont "
          "été".split())
_LID_MAP = {"deu_Latn": "de", "fra_Latn": "fr","swh_Latn": "sw", "swa_Latn": "sw", "amh_Ethi": "am", "arb_Arab": "ar","eng_Latn": "en", "hin_Deva": "hi", "cmn_Hans": "zh", "zho_Hans": "zh",
            "ben_Beng": "bn", "tel_Telu": "te", "spa_Latn": "es", "urd_Arab": "ur",
            "mar_Deva": "mr", "npi_Deva": "ne", "hin_Latn": "hi-Latn"}


class LID:
    """GlotLID (fastText) when available, else a script + stopword heuristic.

    Heuristic limits: Devanagari defaults to Hindi (cannot separate hi/mr/ne) and Arabic
    script to Arabic (cannot separate ar/ur).  Install `fasttext` + GlotLID for Marathi and
    Arabic targets and for the full leakage matrix.
    """

    def __init__(self, use_glotlid: bool = True):
        self.model = None
        if use_glotlid:
            try:
                import fasttext
                from huggingface_hub import hf_hub_download
                self.model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
            except Exception as e:  # pragma: no cover
                print(f"[lid] GlotLID unavailable ({e}); using script heuristic")

    def __call__(self, text: str) -> str:
        text = text.replace("\n", " ").strip()
        if not re.search(r"\w", text):
            return "none"
        if self.model is not None:
            lab = self.model.predict(text)[0][0].replace("__label__", "")
            return _LID_MAP.get(lab, lab)
        counts = {}
        for ch in text:
            for s in token_scripts(ch):
                counts[s] = counts.get(s, 0) + 1
        if not counts:
            return "none"
        s = max(counts, key=counts.get)
        if s == "Latn":
            w = re.findall(r"[a-zß-ÿ]+", text.lower())
            sc = {"es": sum(x in _ES for x in w) + 2 * len(re.findall(r"[ñ¿¡áéíóú]", text.lower())),
                  "en": sum(x in _EN for x in w), "sw": sum(x in _SW for x in w),
                  "de": sum(x in _DE for x in w) + 2 * len(re.findall(r"[äöüß]", text.lower())),
                  "fr": sum(x in _FR for x in w) + 2 * len(re.findall(r"[àâçèêëîïôùûœ]", text.lower()))}
            return max(sc, key=sc.get)
        # Deva cannot be split into hi/mr/ne, nor Arab into ar/ur, without GlotLID
        return {"Deva": "hi", "Hani": "zh", "Beng": "bn", "Telu": "te", "Arab": "ar",
                "Ethi": "am"}.get(s, s)


def segment_langs(text: str, lid: LID, n_seg: int = 4) -> list[str]:
    words = text.split()
    if len(words) < 2 * n_seg:
        return [lid(text)]
    k = max(1, len(words) // n_seg)
    return [lid(" ".join(words[i:i + k])) for i in range(0, len(words), k)]


# --------------------------------------------------------------------------------------
# Closed-loop early defaultness (+ prefix LID for the DEFAULT conjunction).
# --------------------------------------------------------------------------------------
@dataclass
class Early:
    dM: np.ndarray                 # [N] mean_t (M_tgt - M_en)
    mass: dict                     # lang -> [N, T]
    prefix_lid: list[str]
    prefixes: list[str]


def early_defaultness(model, tok, prompt_ids, ts: TokenSets, target: str, T: int = 3,
                      prefix_len: int = 8, batch_size: int = 16, lid: LID | None = None) -> Early:
    gens, logits = greedy(model, tok, prompt_ids, max_new_tokens=max(T, prefix_len),
                          batch_size=batch_size, logits=True)
    dM, m = delta_m_from_logits(logits[:, :T], ts, target)
    lid = lid or LID(use_glotlid=False)
    prefixes = [tok.decode(g[:prefix_len], skip_special_tokens=True) for g in gens]
    return Early(dM.numpy(), {l: v.numpy() for l, v in m.items()}, [lid(p) for p in prefixes],
                 prefixes)


def boot_ci(x, B: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile bootstrap CI of the mean, prompt as the resampling unit."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def compare_early(e: Early, b: Early, target: str, others: list[str]) -> dict:
    """Paired comparison on the same prompts (row i of e and b is the same item)."""
    d_e = np.array([(d > 0) and (l == target) for d, l in zip(e.dM, e.prefix_lid)], dtype=float)
    d_b = np.array([(d > 0) and (l == target) for d, l in zip(b.dM, b.prefix_lid)], dtype=float)
    l_e = np.array([l == target for l in e.prefix_lid], dtype=float)
    l_b = np.array([l == target for l in b.prefix_lid], dtype=float)
    leak = {l: float(e.mass[l].mean() - b.mass[l].mean()) for l in others if l in e.mass}
    return {"n": int(len(e.dM)), "dM_base": float(b.dM.mean()), "dM_edit": float(e.dM.mean()),
            "gain_dM": float((e.dM - b.dM).mean()), "gain_dM_ci": boot_ci(e.dM - b.dM),
            "lid_base": float(l_b.mean()), "lid_edit": float(l_e.mean()),
            "gain_lid": float((l_e - l_b).mean()), "gain_lid_ci": boot_ci(l_e - l_b),
            "DEFAULT_base": float(d_b.mean()), "DEFAULT_edit": float(d_e.mean()),
            "gain_DEFAULT": float((d_e - d_b).mean()), "gain_DEFAULT_ci": boot_ci(d_e - d_b),
            "leak": leak, "leak_max": max([0.0] + list(leak.values()))}


# --------------------------------------------------------------------------------------
# Teacher-forced probes (cheap, used for screening and per-layer causal curves).
# --------------------------------------------------------------------------------------
def tf_sequences(model, tok, prompt_ids, T: int, batch_size: int = 16):
    """prompt + the unedited model's first T-1 greedy tokens.  The last T positions of each
    sequence predict response tokens 1..T."""
    gens, _ = greedy(model, tok, prompt_ids, max_new_tokens=max(T - 1, 1), batch_size=batch_size)
    return [p + g[: T - 1] if T > 1 else p for p, g in zip(prompt_ids, gens)]


def tf_logits(model, tok, seqs, T: int, batch_size: int = 16) -> torch.Tensor:
    dev, out = input_device(model), []
    with torch.no_grad():
        for chunk in batches(seqs, batch_size):
            ids, att = left_pad(chunk, tok.pad_token_id, dev)
            out.append(model(input_ids=ids, attention_mask=att).logits[:, -T:].float().cpu())
    return torch.cat(out)


def tf_masses(model, tok, seqs, ts, target, T, batch_size=16):
    return delta_m_from_logits(tf_logits(model, tok, seqs, T, batch_size), ts, target)


# --------------------------------------------------------------------------------------
# Guardrails that do not penalise the intended language change.
# --------------------------------------------------------------------------------------
def committed_contexts(tok, prompt_ids, tgt_texts, n_commit: int = 4):
    """prompt + first `n_commit` tokens of a reference target-language response."""
    from .model_io import text_ids
    return [p + text_ids(tok, t)[:n_commit] for p, t in zip(prompt_ids, tgt_texts)]


def last_logprobs(model, tok, seqs, batch_size=16) -> torch.Tensor:
    dev, out = input_device(model), []
    with torch.no_grad():
        for chunk in batches(seqs, batch_size):
            ids, att = left_pad(chunk, tok.pad_token_id, dev)
            out.append(model(input_ids=ids, attention_mask=att).logits[:, -1].float()
                       .log_softmax(-1).cpu())
    return torch.cat(out)


def kl(base_lp: torch.Tensor, edit_lp: torch.Tensor) -> float:
    """Mean KL(p_base || p_edit) over contexts."""
    return float((base_lp.exp() * (base_lp - edit_lp)).sum(-1).mean())


def response_nll(model, tok, prompt_ids, resp_texts, skip: int = 4, max_resp: int = 48,
                 batch_size: int = 16) -> float:
    """Mean NLL (nats/token) of reference responses after the first `skip` tokens."""
    from .model_io import text_ids
    dev = input_device(model)
    items = []
    for p, r in zip(prompt_ids, resp_texts):
        rid = text_ids(tok, r)[:max_resp]
        if len(rid) > skip + 1:
            items.append((p, rid))
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for chunk in batches(items, batch_size):
            seqs = [p + r for p, r in chunk]
            ids, att = left_pad(seqs, tok.pad_token_id, dev)
            lp = model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
            L = ids.shape[1]
            for b, (p, r) in enumerate(chunk):
                start = L - len(r) + skip
                tgt = ids[b, start:]
                pred = lp[b, start - 1:L - 1]
                tot -= float(pred.gather(-1, tgt[:, None]).sum())
                cnt += len(tgt)
    return tot / max(cnt, 1)


def downstream_scores(model, tok, prompt_ids, refs, k: int, n_score: int, batch_size: int = 16,
                      steer=None):
    """Utility cost of an edit, free of the "it makes target tokens likelier" confound.

    Each context is prompt + the first R = k + 1 + n_score tokens of a target-language
    reference.  When `steer` is given, the edit is applied exactly where it is applied at
    deployment (the prompt-final position and the first k response positions); the scored
    tokens k+1 .. R-1 are predicted from positions the edit never touches, so any change in
    them is collateral damage carried through the KV cache, not the intended language push.

    Returns (mean NLL in nats/token over scored tokens, log-probs at the first scored
    position [N, V] on CPU for a KL comparison, number of contexts used)."""
    from .model_io import text_ids
    R = k + 1 + n_score
    items = [(p, text_ids(tok, r)[:R]) for p, r in zip(prompt_ids, refs)]
    items = [(p, r) for p, r in items if len(r) == R]
    if not items:
        raise ValueError(f"no reference has >= {R} tokens; lower n_score")
    dev, tot, cnt, first = input_device(model), 0.0, 0, []
    if steer is not None:
        steer.force_range = (R + 1, R - k)
    try:
        with torch.no_grad():
            for chunk in batches(items, batch_size):
                ids, att = left_pad([p + r for p, r in chunk], tok.pad_token_id, dev)
                L = ids.shape[1]
                lp = model(input_ids=ids, attention_mask=att).logits[:, L - R + k: L - 1].float() \
                    .log_softmax(-1)
                tgt = ids[:, L - R + k + 1: L]
                tot -= float(lp.gather(-1, tgt[..., None]).sum())
                cnt += tgt.numel()
                first.append(lp[:, 0].cpu())
    finally:
        if steer is not None:
            steer.force_range = None
    return tot / cnt, torch.cat(first), len(items)
