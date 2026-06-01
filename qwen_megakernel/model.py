"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 4096
VOCAB_SIZE = 3072

# MRoPE interleaved sections for Qwen3-TTS talker
# mrope_section = [24, 20, 20] means each section uses that many (cos,sin) pairs
# giving 24+20+20 = 64 pairs = 128 dims per head
MROPE_SECTION = [24, 20, 20]  # text, time, audio
ROPE_THETA = 1_000_000.0

_decode = torch.ops.qwen_megakernel_C.decode


def _build_mrope_tables(max_seq_len: int):
    """Build MRoPE cos/sin tables for the Qwen3-TTS talker.

    The talker uses interleaved MRoPE with sections [24, 20, 20] (text, time, audio).
    Each section uses its own position index; during single-token decode we pass
    the same scalar position for all three sections (time position dominates).
    The kernel reads cos_table[position * HEAD_DIM : (position+1) * HEAD_DIM]
    so we pre-bake the full interleaved layout into the table rows.
    """
    import torch

    # Build inv_freq for each section independently
    # Section dims: 24, 20, 20 pairs → 48, 40, 40 elements per head
    section_dims = [s * 2 for s in MROPE_SECTION]  # [48, 40, 40]
    inv_freqs = []
    offset = 0
    for dim in section_dims:
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / HEAD_DIM)
        )
        inv_freqs.append(inv_freq)
        offset += dim

    positions = torch.arange(max_seq_len, dtype=torch.float32)

    cos_rows = []
    sin_rows = []
    for pos_idx in range(max_seq_len):
        cos_parts = []
        sin_parts = []
        for inv_freq in inv_freqs:
            freqs = positions[pos_idx] * inv_freq  # [dim/2]
            cos_parts.append(torch.cos(freqs))
            sin_parts.append(torch.sin(freqs))
        # Interleaved layout: concatenate section cos values, then repeat for
        # the second half (standard rotate-half convention the kernel uses)
        cos_row = torch.cat(cos_parts)  # [64]
        sin_row = torch.cat(sin_parts)  # [64]
        cos_rows.append(torch.cat([cos_row, cos_row]))  # [128]
        sin_rows.append(torch.cat([sin_row, sin_row]))  # [128]

    cos_table = torch.stack(cos_rows).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.stack(sin_rows).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors."""
    if not verbose:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging

            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    # MRoPE tables for Qwen3-TTS talker
    cos_table, sin_table = _build_mrope_tables(MAX_SEQ_LEN)

    # Per-layer weight list (11 tensors per layer, flattened)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,  # tied embeddings
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Qwen Megakernel torch ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            MAX_SEQ_LEN,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        # bmax scratch: must be >= LDG_LM_NUM_BLOCKS; 4096 covers any tuning
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one token. Returns the next token id."""
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        _gen = torch.ops.qwen_megakernel_C.generate_nosync
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 100, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)


def load_talker_weights(model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
    """Load Qwen3-TTS talker backbone weights into the megakernel weight dict.

    Only loads the talker transformer (28 layers, hidden 2048) — not the
    code_predictor, text_projection, or speech tokenizer.
    Returns (weights_dict, None) — no tokenizer needed for TTS decode.
    """
    import os
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    if verbose:
        print(f"Loading {model_name} talker weights...")

    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    shard = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    if verbose:
        print("Loading safetensors shard...")
    sd = load_file(shard, device="cuda")

    cos_table, sin_table = _build_mrope_tables(MAX_SEQ_LEN)

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend([
            sd[p + "input_layernorm.weight"].contiguous(),
            sd[p + "self_attn.q_proj.weight"].contiguous(),
            sd[p + "self_attn.k_proj.weight"].contiguous(),
            sd[p + "self_attn.v_proj.weight"].contiguous(),
            sd[p + "self_attn.q_norm.weight"].contiguous(),
            sd[p + "self_attn.k_norm.weight"].contiguous(),
            sd[p + "self_attn.o_proj.weight"].contiguous(),
            sd[p + "post_attention_layernorm.weight"].contiguous(),
            sd[p + "mlp.gate_proj.weight"].contiguous(),
            sd[p + "mlp.up_proj.weight"].contiguous(),
            sd[p + "mlp.down_proj.weight"].contiguous(),
        ])

    embed_weight = sd["talker.model.codec_embedding.weight"].contiguous()  # [3072, 2048]
    lm_head_weight = sd["talker.codec_head.weight"].contiguous()

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=sd["talker.model.norm.weight"].contiguous(),
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        # Keep full sd alive so tensors aren't GC'd
        _sd_ref=sd,
    )

    if verbose:
        print("Talker weights loaded.")
    return weights, None


class TalkerDecoder(Decoder):
    """Megakernel decoder loaded with Qwen3-TTS talker weights.

    Identical kernel to Decoder — only weights differ.
    step(token_id) takes a codec token id, returns next codec token id.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
        weights, _ = load_talker_weights(model_name, verbose=verbose)
        super().__init__(weights=weights, tokenizer=None)
        if verbose:
            print("TalkerDecoder ready.")
