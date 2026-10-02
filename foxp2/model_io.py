"""Model loading, chat formatting, batching and residual-stream capture."""
from __future__ import annotations

import contextlib

import torch


def load_model(spec, dtype=torch.bfloat16, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    kw = dict(dtype=dtype, device_map=device_map)
    if spec.attn_implementation:
        kw["attn_implementation"] = spec.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw).eval()
    prepare_tokenizer(tok)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def prepare_tokenizer(tok):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot find decoder layers")


def input_device(model):
    return model.get_input_embeddings().weight.device


def chat_ids(tok, user_text: str, chat_kwargs: dict | None = None) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                add_generation_prompt=True, **(chat_kwargs or {}))
    return tok(s, add_special_tokens=False)["input_ids"]


def text_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def left_pad(id_lists: list[list[int]], pad_id: int, device):
    L = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
        att[i, L - len(x):] = 1
    return ids.to(device), att.to(device)


def batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _h(out):
    return out[0] if isinstance(out, tuple) else out


@contextlib.contextmanager
def capture(model, layers, last_k: int | None = None, keep_grad: bool = False):
    """Capture residual-stream outputs of `layers`.  Stores [B, last_k, d] (or full seq)."""
    store, handles, dl = {}, [], decoder_layers(model)
    for l in layers:
        def hook(module, args, output, l=l):
            h = _h(output)
            if keep_grad:
                h.retain_grad()
                store[l] = h
            else:
                store[l] = (h[:, -last_k:] if last_k else h).detach()
        handles.append(dl[l].register_forward_hook(hook))
    try:
        yield store
    finally:
        for hd in handles:
            hd.remove()


@contextlib.contextmanager
def add_vectors(model, vecs: dict[int, torch.Tensor], last_k: int):
    """Add a fixed vector to the last `last_k` positions at given layers (teacher-forced probes)."""
    handles, dl = [], decoder_layers(model)
    for l, v in vecs.items():
        def hook(module, args, output, v=v):
            h = _h(output)
            h2 = h.clone()
            h2[:, -last_k:] += v.to(h.device, h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        handles.append(dl[l].register_forward_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def greedy(model, tok, id_lists, max_new_tokens: int, batch_size: int = 16, logits: bool = False):
    """Greedy decoding.  Returns (generated id lists, per-step logits [B, T, V] or None)."""
    dev, gens, all_logits = input_device(model), [], []
    for chunk in batches(id_lists, batch_size):
        ids, att = left_pad(chunk, tok.pad_token_id, dev)
        out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
                             output_logits=logits)
        seq = out.sequences[:, ids.shape[1]:]
        for row in seq.tolist():
            gens.append(row)
        if logits:
            lg = torch.stack(out.logits, dim=1).float()
            if lg.shape[1] < max_new_tokens:     # all rows stopped early: pad with last step
                lg = torch.cat([lg, lg[:, -1:].expand(-1, max_new_tokens - lg.shape[1], -1)], 1)
            all_logits.append(lg.cpu())
    return gens, (torch.cat(all_logits) if logits else None)
