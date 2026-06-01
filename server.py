"""Streaming inference server for Qwen3-TTS talker decode via megakernel.

Interface:
    POST /synthesize
        Body: {"text": "...", "max_tokens": 2048}
        Response: newline-delimited JSON stream, one codec token id per line
            {"token": 42, "position": 0}
            {"token": 17, "position": 1}
            ...
            {"done": true, "total_tokens": N, "elapsed_ms": X}

The server exposes raw codec token ids (codebook 0).
The caller is responsible for running code_predictor (codebooks 1-15)
and the speech tokenizer decoder to get PCM audio.

Run:
    python server.py [--host 0.0.0.0] [--port 8000] [--model Qwen/Qwen3-TTS]
"""

import argparse
import json
import time
from typing import Iterator

import torch
from flask import Flask, Response, request, jsonify

from qwen_megakernel.model import TalkerDecoder

# Codec EOS token id from config
CODEC_EOS_ID = 2150
DEFAULT_MAX_TOKENS = 2048

app = Flask(__name__)
_decoder: TalkerDecoder | None = None


def get_decoder() -> TalkerDecoder:
    global _decoder
    if _decoder is None:
        raise RuntimeError("Decoder not initialised — call init_decoder() first")
    return _decoder


def init_decoder(model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
    global _decoder
    _decoder = TalkerDecoder(model_name=model_name, verbose=verbose)


def _stream_tokens(token_ids: list[int], max_tokens: int) -> Iterator[str]:
    """Feed prefill token ids into the decoder, then stream generated tokens."""
    dec = get_decoder()
    dec.reset()

    t0 = time.perf_counter()

    # Prefill: step through all but the last token without streaming
    for tid in token_ids[:-1]:
        dec.step(tid)

    # Decode: stream each generated token as it comes
    token = token_ids[-1]
    position = dec.position
    total = 0

    for _ in range(max_tokens):
        next_token = dec.step(token)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        yield json.dumps({"token": token, "position": position}) + "\n"

        position += 1
        total += 1
        token = next_token

        if token == CODEC_EOS_ID:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000
    tok_per_sec = total / max(elapsed_ms / 1000, 1e-6)
    yield json.dumps({
        "done": True,
        "total_tokens": total,
        "elapsed_ms": round(elapsed_ms, 2),
        "tokens_per_sec": round(tok_per_sec, 1),
    }) + "\n"


@app.post("/synthesize")
def synthesize():
    body = request.get_json(force=True)
    token_ids: list[int] = body.get("token_ids")
    max_tokens: int = int(body.get("max_tokens", DEFAULT_MAX_TOKENS))

    if not token_ids or not isinstance(token_ids, list):
        return jsonify({"error": "token_ids (list[int]) required"}), 400

    return Response(
        _stream_tokens(token_ids, max_tokens),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},  # disable nginx buffering
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "decoder_ready": _decoder is not None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    init_decoder(model_name=args.model, verbose=not args.no_verbose)

    print(f"Server ready on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
