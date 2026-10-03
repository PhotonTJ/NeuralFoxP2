"""Remote-code model class for FOXP2-steered checkpoints.

Loaded with
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)

The class is the stock HF CausalLM for the architecture plus a `foxp2_steer` submodule whose
buffers are stored in `foxp2.safetensors`.  Config overrides at load time:
    foxp2_gamma (float), foxp2_mode ("decode_window" | "all"), foxp2_k_decode (int),
    foxp2_use_gate (bool), foxp2_sink_factor (float, 0 = edit attention-sink positions too)
and the env vars FOXP2_GAMMA / FOXP2_MODE / FOXP2_KDECODE / FOXP2_GATE / FOXP2_SINK override at
runtime.
"""
from .foxp2_steer import FOXP2Steerer


def _decoder_layers(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError("decoder layers not found")


def _make(base_cls, name):
    def __init__(self, config):
        base_cls.__init__(self, config)
        fc = dict(getattr(config, "foxp2", {}) or {})
        self.foxp2_steer = FOXP2Steerer(
            layers=fc["layers"], d_model=config.hidden_size, k_en=fc["k_en"],
            mode=getattr(config, "foxp2_mode", fc.get("mode", "decode_window")),
            k_decode=getattr(config, "foxp2_k_decode", fc.get("k_decode", 8)),
            beta=fc.get("beta", 1.0),
            gamma=getattr(config, "foxp2_gamma", fc.get("gamma", 1.0)),
            use_gate=getattr(config, "foxp2_use_gate", fc.get("use_gate", True)),
            sink_factor=getattr(config, "foxp2_sink_factor", fc.get("sink_factor", 5.0)))
        self.foxp2_steer.attach(self, _decoder_layers(self))

    cls = type(name, (base_cls,), {"__init__": __init__})
    cls.__module__ = __name__
    return cls


try:
    from transformers import LlamaForCausalLM
    FOXP2LlamaForCausalLM = _make(LlamaForCausalLM, "FOXP2LlamaForCausalLM")
except ImportError:  # pragma: no cover
    pass
try:
    from transformers import Gemma2ForCausalLM
    FOXP2Gemma2ForCausalLM = _make(Gemma2ForCausalLM, "FOXP2Gemma2ForCausalLM")
except ImportError:  # pragma: no cover
    pass
try:
    from transformers import Qwen3ForCausalLM
    FOXP2Qwen3ForCausalLM = _make(Qwen3ForCausalLM, "FOXP2Qwen3ForCausalLM")
except ImportError:  # pragma: no cover
    pass
