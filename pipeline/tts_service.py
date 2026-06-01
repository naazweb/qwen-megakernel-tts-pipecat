"""Pipecat TTSService — Qwen3-TTS with megakernel talker decode.

Architecture per utterance:
  1. HF prefill        — one-time setup: build input embeddings, run one
                         forward pass to get first codec token + KV cache.
  2. Megakernel decode — TalkerDecoder.step() is the hot loop that produces
                         codec token ids at ~1000 tok/s (replaces HF generate).
  3. code_predictor    — separate 5-layer model fills codebooks 1-15 per token.
  4. speech_tokenizer  — decodes CHUNK_TOKENS codec rows → PCM, pushed
                         immediately as a TTSAudioRawFrame (true streaming).
"""

import asyncio
import os
import queue
import sys
import time
from typing import AsyncGenerator

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService, TextAggregationMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_megakernel.model import TalkerDecoder

SAMPLE_RATE = 24000
CHUNK_TOKENS = 6      # tokens per audio chunk (~0.5 s at 12 Hz)
MAX_NEW_TOKENS = 2048


class QwenTTSEngine:
    """
    Owns:
      - HF Qwen3TTS model  (prefill, code_predictor, speech_tokenizer)
      - TalkerDecoder      (megakernel — the hot decode loop)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS", device: str = "cuda",
                 verbose: bool = True):
        from qwen_tts import Qwen3TTSModel

        if verbose:
            logger.info(f"Loading {model_name}...")

        self.device = device
        self._qwen = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.hf = self._qwen.model
        self.hf.eval()
        self.processor = self._qwen.processor

        # Megakernel decoder — loads talker backbone weights from same checkpoint
        if verbose:
            logger.info("Loading megakernel TalkerDecoder...")
        self.decoder = TalkerDecoder(self._qwen, verbose=verbose)

        if verbose:
            logger.info("QwenTTSEngine ready.")

    # ------------------------------------------------------------------
    def synthesize(self, text: str, pcm_queue: queue.Queue) -> None:
        """
        Called in a background thread.
        Puts np.ndarray float32 PCM chunks into pcm_queue, then None sentinel.
        """
        t0 = time.perf_counter()
        talker = self.hf.talker
        cfg = self.hf.config
        talker_cfg = cfg.talker_config
        num_code_groups = talker_cfg.num_code_groups   # 16
        device = self.device
        dtype = torch.bfloat16
        eos_id = talker_cfg.codec_eos_token_id
        language_id = talker_cfg.codec_language_id["english"]

        # ── 1. HF prefill ────────────────────────────────────────────────────
        formatted = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = self.processor(text=formatted, return_tensors="pt", padding=True)
        input_ids = inp["input_ids"].to(device)

        with torch.inference_mode():
            tts_special = talker.text_projection(
                talker.get_text_embeddings()(
                    torch.tensor(
                        [[cfg.tts_bos_token_id,
                          cfg.tts_eos_token_id,
                          cfg.tts_pad_token_id]],
                        device=device,
                    )
                )
            ).chunk(3, dim=1)
            tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_special

            codec_prefill_ids = [[
                talker_cfg.codec_think_id,
                talker_cfg.codec_think_bos_id,
                language_id,
                talker_cfg.codec_think_eos_id,
            ]]
            codec_emb0 = talker.get_input_embeddings()(
                torch.tensor(codec_prefill_ids, device=device)
            )
            codec_emb1 = talker.get_input_embeddings()(
                torch.tensor(
                    [[talker_cfg.codec_pad_id, talker_cfg.codec_bos_id]],
                    device=device,
                )
            )
            codec_input_emb = torch.cat([codec_emb0, codec_emb1], dim=1)

            role_embed = talker.text_projection(
                talker.get_text_embeddings()(input_ids[:, :3])
            )
            pad_part = tts_pad_embed.expand(-1, codec_input_emb.shape[1] - 2, -1)
            prefill_codec = (
                torch.cat([pad_part, tts_bos_embed], dim=1) + codec_input_emb[:, :-1]
            )
            talker_input_embed = torch.cat([
                role_embed,
                prefill_codec,
                talker.text_projection(
                    talker.get_text_embeddings()(input_ids[:, 3:4])
                ) + codec_input_emb[:, -1:],
            ], dim=1)

            trailing_text_hidden = torch.cat((
                talker.text_projection(
                    talker.get_text_embeddings()(input_ids[:, 4:-5])
                ),
                tts_eos_embed,
            ), dim=1)

            prefill_out = talker(
                inputs_embeds=talker_input_embed,
                attention_mask=torch.ones(
                    1, talker_input_embed.shape[1], device=device, dtype=torch.long
                ),
                use_cache=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=-1,
            )

        first_token = int(prefill_out.logits[0, -1].argmax())
        logger.info(
            f"Prefill: {(time.perf_counter() - t0)*1000:.1f} ms  "
            f"first_token={first_token}"
        )

        # ── 2. Copy HF KV cache → megakernel KV cache ───────────────────────
        self.decoder.reset()
        past_kv = prefill_out.past_key_values
        prefill_len = past_kv.get_seq_length()
        for i in range(len(past_kv)):
            k_hf, v_hf = past_kv[i]
            self.decoder._k_cache[i, :, :prefill_len, :] = k_hf[0].to(dtype)
            self.decoder._v_cache[i, :, :prefill_len, :] = v_hf[0].to(dtype)
        self.decoder._position = prefill_len

        # ── 3. Megakernel decode loop (codebook 0) ───────────────────────────
        #    code_predictor fills codebooks 1-15 per token.
        code_predictor = talker.code_predictor
        chunk_codes: list[torch.Tensor] = []
        token = first_token
        first_audio = True

        with torch.inference_mode():
            for _ in range(MAX_NEW_TOKENS):
                if token == eos_id:
                    break

                # codebooks 1-15
                tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                tok_embed = talker.get_input_embeddings()(tok_tensor)
                past_hidden = torch.zeros(
                    1, 1, talker_cfg.hidden_size, device=device, dtype=dtype
                )
                pred = code_predictor.generate(
                    inputs_embeds=torch.cat([past_hidden, tok_embed], dim=1),
                    max_new_tokens=num_code_groups - 1,
                    do_sample=False,
                )
                full_row = torch.cat(
                    [tok_tensor[0, 0:1], pred[0, 1:]], dim=0
                )  # [16]
                chunk_codes.append(full_row)

                # ── megakernel: advance one step ─────────────────────────────
                token = self.decoder.step(token)

                # ── stream chunk ─────────────────────────────────────────────
                if len(chunk_codes) >= CHUNK_TOKENS:
                    pcm = self._vocoder(chunk_codes)
                    if first_audio:
                        logger.info(
                            f"TTFC: {(time.perf_counter() - t0)*1000:.1f} ms"
                        )
                        first_audio = False
                    pcm_queue.put(pcm)
                    chunk_codes = []

        # flush tail
        if chunk_codes:
            pcm_queue.put(self._vocoder(chunk_codes))

        logger.info(f"Synthesis done: {(time.perf_counter() - t0)*1000:.1f} ms")
        pcm_queue.put(None)

    def _vocoder(self, codes: list[torch.Tensor]) -> np.ndarray:
        codes_tensor = torch.stack(codes, dim=0).unsqueeze(0)  # [1, T, 16]
        wavs, _ = self.hf.speech_tokenizer.decode(
            [{"audio_codes": codes_tensor}]
        )
        return wavs[0].astype(np.float32)


# ── Pipecat service ──────────────────────────────────────────────────────────

class QwenTTSService(TTSService):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            push_stop_frames=False,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            **kwargs,
        )
        self._model_name = model_name
        self._device = device
        self._engine: QwenTTSEngine | None = None
        self._settings.model = model_name
        self._settings.voice = None
        self._settings.language = None

    def _ensure_loaded(self):
        if self._engine is None:
            self._engine = QwenTTSEngine(
                model_name=self._model_name,
                device=self._device,
            )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"QwenTTSService: {text!r}")
        self._ensure_loaded()
        try:
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            loop = asyncio.get_running_loop()
            pcm_queue: queue.Queue = queue.Queue()

            # synthesis runs in a thread — event loop stays unblocked
            fut = loop.run_in_executor(
                None, self._engine.synthesize, text, pcm_queue
            )

            first = True
            frame_count = 0
            while True:
                pcm = await loop.run_in_executor(None, pcm_queue.get)
                if pcm is None:
                    break
                if first:
                    await self.stop_ttfb_metrics()
                    first = False
                frame_count += 1
                yield TTSAudioRawFrame(
                    audio=(np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

            await fut  # surface any thread exception
            logger.info(f"QwenTTSService: {frame_count} frames pushed")
            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)

        except Exception as e:
            logger.error(f"QwenTTSService error: {e}", exc_info=True)
            yield ErrorFrame(str(e))
