"""FineART-VLA torch frontend on RTX / Jetson Orin (SM87).

FineART-VLA = π0.5 + System 2 language head (Scale AI × HF LeRobot).
Inference-time structural delta vs π0.5, proven against the released
``lerobot/FineART-VLA-subtask-midtrained`` checkpoint (813-tensor
safetensors header), is exactly two new tensors::

    paligemma_with_expert.paligemma.lm_head.weight    [257152, 2048]
    paligemma_with_expert.gemma_expert.lm_head.weight [257152, 1024]

System 1 (flow-matching action chunk) therefore runs the π0.5 path
unchanged: this subclass inherits :class:`Pi05TorchFrontendRtx` as-is.
``convert_pi05_safetensors`` pulls keys strictly by name, so the extra
lm_head tensors in a FineART-VLA checkpoint are simply never read —
loading a fineart checkpoint through the π0.5 path is safe, and the
lerobot preprocessor/postprocessor norm-stats pair ships in the same
release dir (loaded natively via ``flash_rt.core.utils.norm_stats``).

System 2 (subtask text decode) is what this subclass adds: greedy
autoregressive decode over the gemma_2b prefix KV cache against
``paligemma.lm_head``. Lerobot-side reference =
``FineARTVLAPolicy.select_message`` (greedy temperature=0, KV cache,
suppress ``<locXXXXX>`` ids for subtask generation); FlashRT-side
AR-decode machinery precedent = ``frontends/torch/pi0fast.py``.

Status (2026-10-27): System 1 works end-to-end through
``load_model(config="fineart_vla", num_views=3)``; System 2 subtask
decode lives in ``flash_rt.models.fineart_vla.text_decode_rtx``
(pure-torch PaliGemma port — no transformers dependency), lazily
built on the first ``generate_subtask`` call.

Checkpoint = the lerobot HF release directory as-is (config.json +
model.safetensors + pre/postprocessor + action_tokenizer); base is
``lerobot/pi052_base`` per train_config.json, chunk_size 50 /
num_steps 10 / QUANTILES norm — all π0.5 defaults already.
"""

from __future__ import annotations

import logging
import pathlib

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx, _PromptEmbedCache

logger = logging.getLogger(__name__)


class FineARTVLATorchFrontendRtx(Pi05TorchFrontendRtx):
    """π0.5-compatible frontend + (planned) System 2 text decode."""

    # Key inside the raw lerobot safetensors (the ``model.`` wrap is
    # auto-stripped the same way convert_pi05_safetensors does).
    TEXT_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"

    def __init__(self, checkpoint_dir, num_views=3, action_dim=14,
                 num_steps=None, **kwargs):
        # FineART-VLA trains with 3 camera views (cam_high + both
        # wrists); π0.5's num_views=2 default must not leak in here.
        # action_dim/num_steps MUST stay explicit in the signature:
        # load_model forwards only parameters inspect.signature() can
        # see — anything swallowed by **kwargs is silently dropped
        # (found the hard way in the 2026-10-27 smoke, action_dim=14
        # arrived as the LIBERO 7 default).
        super().__init__(checkpoint_dir, num_views=num_views,
                         action_dim=action_dim,
                         **({"num_steps": num_steps}
                            if num_steps is not None else {}), **kwargs)
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._text_lm_head = None
        self._subtask_decoder = None

    def _ensure_text_lm_head(self):
        """Lazy-load paligemma.lm_head [257152, 2048] → CUDA bf16.

        Kept out of __init__ so System 1 users never pay for the
        (currently unused) 1 GB head; Stage 2b's generate_subtask
        consumes this.
        """
        if self._text_lm_head is None:
            import torch
            from safetensors import safe_open
            from flash_rt.executors.torch_weights import (
                _autodetect_strip_prefix,
            )

            path = self._checkpoint_dir / "model.safetensors"
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = set(f.keys())
                strip = _autodetect_strip_prefix(keys)
                key = (strip + self.TEXT_LM_HEAD_KEY) if strip \
                    else self.TEXT_LM_HEAD_KEY
                if key not in keys:
                    raise KeyError(
                        f"{self.TEXT_LM_HEAD_KEY} not found in {path} "
                        "(is this a FineART-VLA checkpoint?)")
                self._text_lm_head = f.get_tensor(key).to(
                    "cuda", torch.bfloat16, non_blocking=True)
            logger.info("FineART-VLA text lm_head loaded: %s",
                        tuple(self._text_lm_head.shape))
        return self._text_lm_head

    def generate_subtask(self, prompt_text: str, images=None, *,
                         max_new_tokens: int = 128, temperature: float = 0.0,
                         top_p: float = 1.0) -> str:
        """System 2: greedy subtask decode, lerobot ``select_message``
        parity target (greedy temperature=0, KV cache, ``<loc…>`` id
        suppression, eos stop).

        ``images``: optional observation dict — same keys the flow path
        consumes (``image``, ``wrist_image``, ``wrist_image_right`` if
        num_views>=3), np uint8 HWC at any resolution (resized to 224²
        with padding internally). ``None`` → text-only prefix (all view
        masks False). Decode cadence is caller-controlled: System 1
        keeps running π0.5-style on fixed subtask text between calls.
        """
        decoder = self._ensure_subtask_decoder()
        return decoder.generate(
            prompt_text, self._order_views(images),
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p)

    def _order_views(self, images):
        """Same view order the π0.5 flow path uses (pi05_rtx img_list)."""
        if images is None:
            return [None] * self.num_views
        if isinstance(images, dict):
            keys = ["image", "wrist_image"]
            if self.num_views >= 3:
                keys.append("wrist_image_right")
            return [images.get(k) for k in keys[: self.num_views]]
        return list(images)[: self.num_views]

    def _ensure_subtask_decoder(self):
        """Lazy-build the ~8 GB SubtaskDecoder on first System 2 call —
        System 1 users never pay for it (mirrors _ensure_text_lm_head)."""
        if self._subtask_decoder is None:
            from flash_rt.models.fineart_vla.text_decode_rtx import SubtaskDecoder

            head = self._ensure_text_lm_head()
            self._subtask_decoder = SubtaskDecoder(
                self._checkpoint_dir, device=str(head.device), lm_head=head)
        return self._subtask_decoder

    # ── System 1 prompt format ─────────────────────────────────────
    # The inherited π0.5 path tokenizes with the openpi template
    # ``Task: {task}, State: {bins};\nAction: `` (pi05_prompt.
    # format_pi05_prompt). This checkpoint was trained on the fineart
    # role-delimited recipe instead (lerobot processor chain: task
    # fallback in FineARTVLATextTokenizerStep → stream "low_level" →
    # state injected, no "Assistant:" tail — that suffix belongs to the
    # System 2 query path): ``User: {task}, State: {bins};\n``. State
    # discretization math is identical on both sides (digitize on
    # linspace(-1,1,257)[:-1] minus 1), so only the wrapper text
    # differs — hence this override of _embed_prompt_cached.

    def _fineart_prompt_ids(self, prompt_text: str, state=None) -> list:
        """Fineart action-path prompt token ids: ``[bos] + Encode(text)``.

        ``state`` = *normalized* 14-dim state (quantile-normalized to
        [−1,1] by the caller — the engine has no separate state input,
        pi05-style state lives only in the prompt).
        """
        import torch.nn.functional as F  # noqa: F401  (parity with inherited embed path)
        from flash_rt.core.utils.pi05_prompt import discretize_pi05_state
        from flash_rt.utils.paligemma_tokenizer import (
            load_paligemma_sentencepiece,
        )

        sp = load_paligemma_sentencepiece()
        cleaned = str(prompt_text).strip().replace("_", " ").replace("\n", " ")
        if state is None:
            text = cleaned
        else:
            bins = discretize_pi05_state(state)
            text = f"User: {cleaned}, State: {' '.join(map(str, bins.tolist()))};\n"
        return list(sp.Encode(text, add_bos=True))

    def _embed_prompt_cached(self, prompt_text, max_len, state=None):
        """Fineart override: fineart template + inherited embedding math."""
        import numpy as np
        import torch
        import torch.nn.functional as F  # noqa: N812

        cache = getattr(self, "_prompt_embed_cache", None)
        if cache is None:
            cache = self._prompt_embed_cache = _PromptEmbedCache()
        key = _PromptEmbedCache.key(prompt_text, max_len, state)
        hit = cache.get(key) if state is None else None
        if hit is not None:
            return hit
        token_ids = torch.tensor(self._fineart_prompt_ids(prompt_text, state),
                                 dtype=torch.long, device="cuda")
        weight = self.embedding_weight
        if weight.device.type != "cuda":
            weight = weight.to("cuda")
        embeds = (F.embedding(token_ids, weight)
                  * float(weight.shape[-1] ** 0.5)).contiguous()
        host = np.ascontiguousarray(embeds.view(torch.uint16).cpu().numpy())
        hit = (embeds, int(token_ids.numel()), host)
        if state is None:
            cache.put(key, hit)
        return hit
