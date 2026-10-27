"""FineART-VLA model-side modules (System 2 decode lands here).

System 1 reuses ``flash_rt.models.pi05.pipeline_rtx`` unchanged —
FineART-VLA's only inference-time structural delta is the language-head
decode (see ``flash_rt/frontends/torch/fineart_vla_rtx.py``). The AR
text decoder port (greedy + prefix KV cache, pi0fast-style machinery)
will live in this package as ``decode_text_rtx.py`` (Stage 2b).
"""
