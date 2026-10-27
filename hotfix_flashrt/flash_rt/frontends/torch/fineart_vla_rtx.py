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

Status (2026-10-27): registration skeleton. System 1 works end-to-end
through ``load_model(config="fineart_vla", num_views=3)``;
``generate_subtask`` is not ported yet (Stage 2b).

Checkpoint = the lerobot HF release directory as-is (config.json +
model.safetensors + pre/postprocessor + action_tokenizer); base is
``lerobot/pi052_base`` per train_config.json, chunk_size 50 /
num_steps 10 / QUANTILES norm — all π0.5 defaults already.
"""

from __future__ import annotations

import logging
import pathlib

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

logger = logging.getLogger(__name__)


class FineARTVLATorchFrontendRtx(Pi05TorchFrontendRtx):
    """π0.5-compatible frontend + (planned) System 2 text decode."""

    # Key inside the raw lerobot safetensors (the ``model.`` wrap is
    # auto-stripped the same way convert_pi05_safetensors does).
    TEXT_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"

    def __init__(self, checkpoint_dir, num_views=3, **kwargs):
        # FineART-VLA trains with 3 camera views (cam_high + both
        # wrists); π0.5's num_views=2 default must not leak in here.
        super().__init__(checkpoint_dir, num_views=num_views, **kwargs)
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._text_lm_head = None

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

    def generate_subtask(self, prompt_text: str,
                         max_new_tokens: int = 128,
                         temperature: float = 0.0) -> str:
        """Stage 2b (not ported): greedy subtask decode, lerobot
        ``select_message`` parity target."""
        raise NotImplementedError(
            "FineART-VLA System 2 text decode is not ported yet; "
            "System 1 (action chunks) works via the inherited π0.5 path. "
            "Port plan: prefix embed via the existing prompt path → "
            "greedy AR loop on paligemma KV cache against "
            "_ensure_text_lm_head(), <loc> id suppression per lerobot "
            "select_message; decode-graph precedent = pi0fast.py.")
