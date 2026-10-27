"""FineART-VLA System 2 text decoder — pure-torch PaliGemma port (RTX/Orin).

Ports the paligemma half of the π0.5 VLM (SigLIP 27L + projector + Gemma
gemma_2b 18L + untied ``paligemma.lm_head``) for greedy subtask-text
decoding, mirroring lerobot ``FineARTVLAPolicy.select_message``:

  - prompt format  ``User: {task}\nAssistant:``  (fineart runtime recipe:
    ``_format_messages`` + the no-target ``Assistant:`` tail)
  - prefix = 3 views × 256 image tokens + ≤128 language tokens (right-pad 0),
    fully bidirectional over valid positions (embed_prefix att_masks all 0)
  - AR loop: KV cache, additive finite-negative masks (NOT -inf — fully
    masked rows must softmax to uniform, never NaN), greedy temperature=0,
    ``<loc…>`` suppression [256000, 257024), eos stop
  - decode via the same paligemma SentencePiece FlashRT prompts use

Source of the exact math (all read locally, v5.4.0 pinned by lerobot):
transformers v5.4.0 ``modeling_gemma.py`` / ``modeling_siglip.py`` /
``modeling_paligemma.py`` + lerobot forks (``pi_gemma.py`` PiGemmaRMSNorm,
``modeling_pi05.py`` embed_prefix/_prepare_attention_masks_4d).

Weights load straight from the lerobot-format checkpoint (zero key
remapping; ``model.`` wrap stripped the same way the pi05 frontend
does). Untied heads: input embedding = ``language_model.embed_tokens``,
output logits = ``paligemma.lm_head`` (proven diverged in
``FineART-VLA-subtask-midtrained`` — max|Δ| ≈ 2.06 vs embed_tokens).

Deliberately NOT kernelized (pi0fast-style fp8/graphs): subtask decode
runs rarely (caller-throttled), ~seconds eager on Orin is fine for
bring-up; optimize later if a workload demands it.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)

# ── openpi gemma_2b / SigLIP so400m constants (lerobot PaligemmaConfig
#    defaults + openpi get_gemma_config("gemma_2b")) ──
GEMMA_WIDTH, GEMMA_DEPTH, GEMMA_HEADS, GEMMA_KV_HEADS = 2048, 18, 8, 1
GEMMA_HEAD_DIM, GEMMA_MLP_DIM, GEMMA_EPS, GEMMA_THETA = 256, 16384, 1e-6, 10000.0
SIG_HIDDEN, SIG_DEPTH, SIG_HEADS, SIG_MLP = 1152, 27, 16, 4304
SIG_PATCH, SIG_IMAGE, SIG_EPS = 14, 224, 1e-6
NUM_IMAGE_EMB = (SIG_IMAGE // SIG_PATCH) ** 2  # 256 per view
VOCAB_SIZE = 257152
LOC_LO, LOC_HI = 256000, 257024          # <loc0000>..<loc1023>
FAST_VOCAB, FAST_SKIP = 1024, 128        # fast_lo = 256000 → empty slice (kept for fidelity)
MAX_LANG_TOKENS = 128                    # fineart tokenizer_max_length


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class _RMSNorm(nn.Module):
    """PiGemmaRMSNorm standard path (lerobot pi_gemma.py:86): fp32 variance,
    ×(1+w), cast back. Weight is stored zero-init (gemma convention)."""

    def __init__(self, dim: int, eps: float = GEMMA_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class _GemmaAttention(nn.Module):
    """transformers v5.4.0 GemmaAttention: GQA, no bias, explicit
    ``scale=head_dim**-0.5`` handed to SDPA."""

    def __init__(self):
        super().__init__()
        self.head_dim = GEMMA_HEAD_DIM
        self.n_rep = GEMMA_HEADS // GEMMA_KV_HEADS
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(GEMMA_WIDTH, GEMMA_HEADS * self.head_dim, bias=False)
        self.k_proj = nn.Linear(GEMMA_WIDTH, GEMMA_KV_HEADS * self.head_dim, bias=False)
        self.v_proj = nn.Linear(GEMMA_WIDTH, GEMMA_KV_HEADS * self.head_dim, bias=False)
        self.o_proj = nn.Linear(GEMMA_HEADS * self.head_dim, GEMMA_WIDTH, bias=False)
        inv = 1.0 / (GEMMA_THETA ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D); pos: (S,) int64 absolute positions
        freqs = torch.outer(pos.float(), self.inv_freq.to(x.device)).to(x.dtype)  # (S, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)                                   # (S, D)
        cos, sin = emb.cos(), emb.sin()
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, mask4d, position_ids, cache):
        b, s, _ = x.shape
        q = self._rope(self.q_proj(x).view(b, s, GEMMA_HEADS, self.head_dim).transpose(1, 2), position_ids)
        k = self._rope(self.k_proj(x).view(b, s, GEMMA_KV_HEADS, self.head_dim).transpose(1, 2), position_ids)
        v = self.v_proj(x).view(b, s, GEMMA_KV_HEADS, self.head_dim).transpose(1, 2)
        k, v = cache.append(self.layer_idx, k, v)  # (B, KV, S_tot, D)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask4d, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))

    layer_idx: int = 0  # bound by _GemmaLayer


class _GemmaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(GEMMA_WIDTH, GEMMA_MLP_DIM, bias=False)
        self.up_proj = nn.Linear(GEMMA_WIDTH, GEMMA_MLP_DIM, bias=False)
        self.down_proj = nn.Linear(GEMMA_MLP_DIM, GEMMA_WIDTH, bias=False)

    def forward(self, x):
        return self.down_proj(_gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class _GemmaLayer(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.input_layernorm = _RMSNorm(GEMMA_WIDTH)
        self.post_attention_layernorm = _RMSNorm(GEMMA_WIDTH)
        self.self_attn = _GemmaAttention()
        self.self_attn.layer_idx = idx
        self.mlp = _GemmaMLP()

    def forward(self, x, mask4d, position_ids, cache):
        x = x + self.self_attn(self.input_layernorm(x), mask4d, position_ids, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _KVCache:
    """Per-layer K/V list with torch.cat appends — 18×128 cats of
    (1, KV, ≤1024, 256) are negligible next to the GEMMs."""

    def __init__(self, n_layers: int):
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers

    def append(self, idx: int, k: torch.Tensor, v: torch.Tensor):
        self.k[idx] = k if self.k[idx] is None else torch.cat((self.k[idx], k), dim=2)
        self.v[idx] = v if self.v[idx] is None else torch.cat((self.v[idx], v), dim=2)
        return self.k[idx], self.v[idx]


class _SiglipVision(nn.Module):
    """transformers v5.4.0 SiglipVisionTransformer (no pooling head use):
    conv patch → +pos → 27×(LN1-attn + LN2-MLP, all bias) → post_layernorm."""

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, SIG_HIDDEN, SIG_PATCH, stride=SIG_PATCH, bias=True)
        self.position_embedding = nn.Embedding(NUM_IMAGE_EMB, SIG_HIDDEN)
        self.register_buffer("position_ids", torch.arange(NUM_IMAGE_EMB).expand(1, -1), persistent=False)
        self.layers = nn.ModuleList(_SiglipLayer() for _ in range(SIG_DEPTH))
        self.post_layernorm = nn.LayerNorm(SIG_HIDDEN, eps=SIG_EPS)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # (B,3,224,224)
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.position_embedding(self.position_ids)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


class _SiglipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(SIG_HIDDEN, eps=SIG_EPS)
        self.layer_norm2 = nn.LayerNorm(SIG_HIDDEN, eps=SIG_EPS)
        self.q_proj = nn.Linear(SIG_HIDDEN, SIG_HIDDEN, bias=True)
        self.k_proj = nn.Linear(SIG_HIDDEN, SIG_HIDDEN, bias=True)
        self.v_proj = nn.Linear(SIG_HIDDEN, SIG_HIDDEN, bias=True)
        self.out_proj = nn.Linear(SIG_HIDDEN, SIG_HIDDEN, bias=True)
        self.fc1 = nn.Linear(SIG_HIDDEN, SIG_MLP, bias=True)
        self.fc2 = nn.Linear(SIG_MLP, SIG_HIDDEN, bias=True)
        self.scale = (SIG_HIDDEN // SIG_HEADS) ** -0.5

    def forward(self, x):
        residual = x
        h = self.layer_norm1(x)
        b, s, _ = h.shape
        hd = SIG_HIDDEN // SIG_HEADS
        q = h.view(b, s, SIG_HEADS, hd).transpose(1, 2)
        k = h.view(b, s, SIG_HEADS, hd).transpose(1, 2)
        v = h.view(b, s, SIG_HEADS, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = residual + self.out_proj(attn.transpose(1, 2).reshape(b, s, -1))
        return x + self.fc2(_gelu_tanh(self.fc1(self.layer_norm2(x))))


class _PaligemmaText(nn.Module):
    """Embed(+sqrt(d) scale inside, HF GemmaTextScaledWordEmbedding) →
    18 layers → final norm → (caller applies untied lm_head)."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, GEMMA_WIDTH)
        self.embed_scale = math.sqrt(GEMMA_WIDTH)
        self.layers = nn.ModuleList(_GemmaLayer(i) for i in range(GEMMA_DEPTH))
        self.norm = _RMSNorm(GEMMA_WIDTH)


class SubtaskDecoder:
    """Self-contained System 2 decoder over one checkpoint's paligemma half.

    Lazy-constructed by the frontend on first ``generate_subtask``; holds a
    private bf16/fp32 copy of the vision tower + language model (~8 GB on
    Orin iGPU) independent of the kernelized flow-path buffers.
    """

    def __init__(self, checkpoint_dir, device="cuda", lm_head: Optional[torch.Tensor] = None):
        self.device = torch.device(device)
        self.model = _PaligemmaText().to(self.device, torch.bfloat16)
        self.vision = _SiglipVision().to(self.device, torch.float32)
        self.projector = nn.Linear(SIG_HIDDEN, GEMMA_WIDTH, bias=True).to(self.device, torch.float32)
        # ``lm_head`` injection lets a frontend reuse its already-loaded
        # head instead of paying a second 1 GB copy.
        self._injected_lm_head = lm_head
        self.lm_head: torch.Tensor = torch.empty(0)
        self._load_weights(pathlib.Path(checkpoint_dir))
        self._sp = None  # sentencepiece, loaded on first use
        self.model.eval().requires_grad_(False)
        self.vision.eval().requires_grad_(False)
        self.projector.eval().requires_grad_(False)

    # ── weight loading (lerobot key names, ``model.`` stripped) ──
    def _load_weights(self, ckpt_dir: pathlib.Path):
        from safetensors.torch import safe_open
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix

        path = ckpt_dir / "model.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as f:
            strip = _autodetect_strip_prefix(set(f.keys()))

            def g(key, dtype=None, to_dev=True):
                t = f.get_tensor((strip + key) if strip else key)
                if dtype is not None:
                    t = t.to(dtype)
                return t.to(self.device) if to_dev else t

            pv = "paligemma_with_expert.paligemma.model"
            # vision tower (fp32 master weights; forward runs bf16-autocast,
            # mirroring to_bfloat16_for_selected_params + _vision_autocast)
            self.vision.patch_embedding.weight.data = g(
                f"{pv}.vision_tower.vision_model.embeddings.patch_embedding.weight", torch.float32)
            self.vision.patch_embedding.bias.data = g(
                f"{pv}.vision_tower.vision_model.embeddings.patch_embedding.bias", torch.float32)
            self.vision.position_embedding.weight.data = g(
                f"{pv}.vision_tower.vision_model.embeddings.position_embedding.weight", torch.float32)
            for i, layer in enumerate(self.vision.layers):
                lp = f"{pv}.vision_tower.vision_model.encoder.layers.{i}"
                layer.layer_norm1.weight.data = g(f"{lp}.layer_norm1.weight", torch.float32)
                layer.layer_norm1.bias.data = g(f"{lp}.layer_norm1.bias", torch.float32)
                layer.layer_norm2.weight.data = g(f"{lp}.layer_norm2.weight", torch.float32)
                layer.layer_norm2.bias.data = g(f"{lp}.layer_norm2.bias", torch.float32)
                for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    getattr(layer, nm).weight.data = g(f"{lp}.self_attn.{nm}.weight", torch.float32)
                    getattr(layer, nm).bias.data = g(f"{lp}.self_attn.{nm}.bias", torch.float32)
                layer.fc1.weight.data = g(f"{lp}.mlp.fc1.weight", torch.float32)
                layer.fc1.bias.data = g(f"{lp}.mlp.fc1.bias", torch.float32)
                layer.fc2.weight.data = g(f"{lp}.mlp.fc2.weight", torch.float32)
                layer.fc2.bias.data = g(f"{lp}.mlp.fc2.bias", torch.float32)
            self.vision.post_layernorm.weight.data = g(
                f"{pv}.vision_tower.vision_model.post_layernorm.weight", torch.float32)
            self.vision.post_layernorm.bias.data = g(
                f"{pv}.vision_tower.vision_model.post_layernorm.bias", torch.float32)
            self.projector.weight.data = g(f"{pv}.multi_modal_projector.linear.weight", torch.float32)
            self.projector.bias.data = g(f"{pv}.multi_modal_projector.linear.bias", torch.float32)

            # language model (bf16; norms fp32 like params_to_keep_float32)
            self.model.embed_tokens.weight.data = g(f"{pv}.language_model.embed_tokens.weight", torch.bfloat16)
            for i, layer in enumerate(self.model.layers):
                lp = f"{pv}.language_model.layers.{i}"
                layer.self_attn.q_proj.weight.data = g(f"{lp}.self_attn.q_proj.weight", torch.bfloat16)
                layer.self_attn.k_proj.weight.data = g(f"{lp}.self_attn.k_proj.weight", torch.bfloat16)
                layer.self_attn.v_proj.weight.data = g(f"{lp}.self_attn.v_proj.weight", torch.bfloat16)
                layer.self_attn.o_proj.weight.data = g(f"{lp}.self_attn.o_proj.weight", torch.bfloat16)
                layer.mlp.gate_proj.weight.data = g(f"{lp}.mlp.gate_proj.weight", torch.bfloat16)
                layer.mlp.up_proj.weight.data = g(f"{lp}.mlp.up_proj.weight", torch.bfloat16)
                layer.mlp.down_proj.weight.data = g(f"{lp}.mlp.down_proj.weight", torch.bfloat16)
                layer.input_layernorm.weight.data = g(f"{lp}.input_layernorm.weight", torch.float32)
                layer.post_attention_layernorm.weight.data = g(f"{lp}.post_attention_layernorm.weight", torch.float32)
            self.model.norm.weight.data = g(f"{pv}.language_model.norm.weight", torch.float32)
            if self._injected_lm_head is not None:
                self.lm_head = self._injected_lm_head.to(device=self.device, dtype=torch.bfloat16)
            else:
                self.lm_head = g("paligemma_with_expert.paligemma.lm_head.weight", torch.bfloat16)
        logger.info("SubtaskDecoder weights loaded from %s", path)

    # ── tokenizer ──
    def _tokenizer(self):
        if self._sp is None:
            from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece
            self._sp = load_paligemma_sentencepiece()
        return self._sp

    def _prompt_tokens(self, task_text: str):
        """``User: {task}\nAssistant:`` → bos + sp pieces, right-pad 0 to 128
        (mirrors FineARTVLATextTokenizerStep runtime path + gemma bos)."""
        sp = self._tokenizer()
        cleaned = task_text.strip().replace("_", " ").replace("\n", " ")
        ids = [sp.bos_id()] + sp.Encode(f"User: {cleaned}\nAssistant:")[: MAX_LANG_TOKENS - 1]
        ids = ids[:MAX_LANG_TOKENS]
        pad = MAX_LANG_TOKENS - len(ids)
        ids_padded = ids + [0] * pad
        mask = [True] * len(ids) + [False] * pad
        return ids_padded, mask

    # ── image preprocessing (mirror _preprocess_images: [0,1] → resize+pad → ×2−1;
    #    missing views = −1 fill with mask False) ──
    def _preprocess_images(self, images) -> tuple[torch.Tensor, torch.Tensor]:
        resized = []
        for img in images:
            if img is None:
                resized.append(None)
                continue
            arr = np.asarray(img)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = np.asarray(arr, dtype=np.float32)
            t = torch.from_numpy(arr).to(self.device)
            if t.ndim == 3 and t.shape[-1] == 3:      # HWC → CHW
                t = t.permute(2, 0, 1)
            resized.append(self._resize_with_pad(t.unsqueeze(0)).squeeze(0))
        n = len(resized)
        out = torch.stack([r if r is not None else torch.full((3, SIG_IMAGE, SIG_IMAGE), -1.0, device=self.device)
                           for r in resized])                              # (V,3,224,224) fp32 [−1,1]
        masks = torch.tensor([r is not None for r in resized], dtype=torch.bool, device=self.device)
        return out, masks

    @staticmethod
    def _resize_with_pad(img: torch.Tensor, size: int = SIG_IMAGE) -> torch.Tensor:
        h, w = img.shape[-2:]
        if (h, w) == (size, size):
            return img
        scale = max(size / h, size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear",
                            align_corners=False, antialias=True).squeeze(0)
        return F.pad(img, (0, size - nw, 0, size - nh))

    # ── forward pieces ──
    def _embed_images(self, images: torch.Tensor) -> torch.Tensor:
        # reference runs fp32 weights under torch.autocast(bf16) on cuda eval
        use_ac = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            feats = self.projector(self.vision(images))
        return feats.to(torch.bfloat16)  # (V, 256, 2048)

    @staticmethod
    def _mask4d(bool_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(B,1,Q,K)→(B,1,Q,K) additive; finite min (never -inf: fully-masked
        rows must softmax to uniform, matching prepare_attention_masks_4d)."""
        neg = torch.finfo(dtype).min
        return torch.where(bool_2d, 0.0, neg).to(dtype)

    def _prefix_forward(self, images, img_masks, lang_ids, lang_mask):
        """Full bidirectional prefix pass → (last hidden, cache, pad, n_pref)."""
        m = self.model
        img_emb = self._embed_images(images)                       # (V,256,2048)
        v = img_emb.shape[0]
        img_pad = img_masks[:, None].expand(v, NUM_IMAGE_EMB)      # (V,256)
        lang = m.embed_tokens(torch.tensor(lang_ids, device=self.device)) * m.embed_scale
        lang_pad = torch.tensor(lang_mask, device=self.device)[None, :].expand(1, len(lang_ids))
        pad = torch.cat((img_pad.reshape(1, -1), lang_pad), dim=1)  # (1,S)
        embs = torch.cat((img_emb.reshape(1, -1, GEMMA_WIDTH), lang[None]), dim=1)
        s = pad.shape[1]
        # all att_masks = 0 → every valid token attends every valid token
        bool_2d = pad[:, None, :] & pad[:, :, None]                # (1,S,S)
        mask4d = self._mask4d(bool_2d[:, None], embs.dtype)
        pos = (torch.cumsum(pad, dim=1) - 1).clamp_min(0)[0]        # pads share last valid id
        cache = _KVCache(GEMMA_DEPTH)
        x = embs
        for layer in m.layers:
            x = layer(x, mask4d, pos, cache)
        x = m.norm(x)
        last_valid = int(pad[0].sum().item())
        return x[0, last_valid - 1], cache, pad, last_valid

    def _step_forward(self, tok_emb, pad_row, position, cache):
        """One AR step: append token embedding, attend cache + self."""
        m = self.model
        x = tok_emb[None, None]
        bool_2d = pad_row[None, None, None, :]                     # (1,1,1,S_tot)
        mask4d = self._mask4d(bool_2d, x.dtype)
        pos = torch.tensor([position], device=self.device)
        for layer in m.layers:
            x = layer(x, mask4d, pos, cache)
        return m.norm(x[0, 0])

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_ix = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_p = sorted_p.masked_fill(mask, 0.0)
            sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            choice = torch.multinomial(sorted_p, num_samples=1)
            return sorted_ix.gather(-1, choice).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── public entry ──
    @torch.no_grad()
    def generate(self, task_text: str, images, *, max_new_tokens: int = 128,
                 temperature: float = 0.0, top_p: float = 1.0,
                 min_new_tokens: int = 0, suppress_loc_tokens: bool = True) -> str:
        """Greedy subtask decode, lerobot ``select_message`` parity target.

        ``images``: per-view camera frames (np uint8 HWC / None for absent),
        any resolution — resized with padding to 224² here.
        """
        sp = self._tokenizer()
        lang_ids, lang_mask = self._prompt_tokens(task_text)
        imgs, img_masks = self._preprocess_images(images)

        last_hidden, cache, pad, n_pref = self._prefix_forward(imgs, img_masks, lang_ids, lang_mask)
        special_ids = {int(sp.unk_id()), int(sp.bos_id()), int(sp.eos_id()), int(sp.pad_id())}
        special_ids = {i for i in special_ids if i >= 0}
        eos = int(sp.eos_id())

        generated: list[int] = []
        cur_pad = pad
        fast_lo = VOCAB_SIZE - 1 - FAST_SKIP - (FAST_VOCAB - 1)
        for _ in range(max_new_tokens):
            logits = (self.lm_head @ last_hidden).float()          # (V,) vocab
            if special_ids and len(generated) < min_new_tokens:
                for sid in special_ids:
                    logits[sid] = float("-inf")
            if 0 < fast_lo < LOC_LO:                               # dead slice w/ these constants; kept for fidelity
                logits[fast_lo:LOC_LO] = float("-inf")
            if suppress_loc_tokens:
                logits[LOC_LO:LOC_HI] = float("-inf")
            tok_id = int(self._sample_next_token(logits, temperature, top_p).item())
            generated.append(tok_id)
            if tok_id == eos:
                break
            tok_emb = self.model.embed_tokens(torch.tensor([tok_id], device=self.device)) * self.model.embed_scale
            position = int(cur_pad[0].sum().item())
            cur_pad = torch.cat((cur_pad, torch.ones((1, 1), dtype=cur_pad.dtype, device=self.device)), dim=1)
            last_hidden = self._step_forward(tok_emb[0], cur_pad[0], position, cache)

        text = sp.decode(generated).strip()
        if not text and generated:
            logger.debug("FineART-VLA System 2 produced empty text; ids[:16]=%s", generated[:16])
        return text
