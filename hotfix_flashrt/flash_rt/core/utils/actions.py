"""FlashRT — Action/state normalization utilities.

Normalization semantics are selected by a top-level ``"norm_mode"`` key
in the norm-stats dict:

  - ``"q01_q99"`` (default, backward compatible): openpi quantile
    semantics — normalized = (x - q01) / (q99 - q01) * 2 - 1.
  - ``"mean_std"``: lerobot-style MEAN_STD semantics —
    normalized = (x - mean) / std.

The marker travels inside norm_stats.json so a checkpoint is
self-describing (G1 fine-tuned deployment writes it via
scripts/inference/g1_ckpt_prep.py). Without the key, behavior is
identical to the original quantile-only implementation.
"""

import numpy as np

LIBERO_ACTION_DIM = 7


def _stat_pair(norm_stats, entry, mode):
    """Return (lo, hi) tensors for the requested semantics.

    ``entry`` is ``norm_stats["actions"]`` or ``norm_stats["state"]``.
    For ``q01_q99`` → (q01, q99); for ``mean_std`` → (mean, std).
    Raises a clear error if the needed stats are missing.
    """
    if mode == "q01_q99":
        if "q01" not in entry or "q99" not in entry:
            raise KeyError(
                f"norm_mode=q01_q99 but stats entry lacks q01/q99 "
                f"(has {sorted(entry)})")
        lo = np.array(entry["q01"], dtype=np.float32)
        hi = np.array(entry["q99"], dtype=np.float32)
    elif mode == "mean_std":
        if "mean" not in entry or "std" not in entry:
            raise KeyError(
                f"norm_mode=mean_std but stats entry lacks mean/std "
                f"(has {sorted(entry)})")
        lo = np.array(entry["mean"], dtype=np.float32)
        hi = np.array(entry["std"], dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown norm_mode {mode!r} (expected 'q01_q99' or 'mean_std')")
    return lo, hi


def _resolve_mode(norm_stats) -> str:
    mode = norm_stats.get("norm_mode", "q01_q99") if norm_stats else "q01_q99"
    if mode not in ("q01_q99", "mean_std"):
        raise ValueError(
            f"Unknown norm_mode {mode!r} in norm_stats "
            f"(expected 'q01_q99' or 'mean_std')")
    return mode


def unnormalize_actions(actions, norm_stats):
    """Map normalized model output actions back to data space (pure numpy).

    ``q01_q99`` mode matches openpi.transforms.Unnormalize._unnormalize_quantile
    exactly: unnorm = (x + 1) / 2 * (q99 - q01) + q01, with NO clipping of
    the raw model output to [-1, 1] beforehand. Clipping first (as an
    earlier version of this function did) silently saturates any action
    dimension whose model output exceeds the training-data quantile range,
    which is common for pi0.5 policies and produces materially wrong actions.

    ``mean_std`` mode (norm_stats["norm_mode"] == "mean_std", lerobot
    MEAN_STD training): unnorm = x * std + mean, also unclipped.
    """
    mode = _resolve_mode(norm_stats)
    lo, hi = _stat_pair(norm_stats, norm_stats["actions"], mode)
    dim = min(actions.shape[-1], len(lo))
    unnorm = actions.copy()
    if mode == "q01_q99":
        unnorm[..., :dim] = (
            (actions[..., :dim] + 1.0) / 2.0 * (hi[:dim] - lo[:dim] + 1e-6)
            + lo[:dim]
        )
    else:  # mean_std
        unnorm[..., :dim] = actions[..., :dim] * (hi[:dim] + 1e-6) + lo[:dim]
    return unnorm


def normalize_state(state, norm_stats, mode=None):
    """Normalize a proprioceptive state vector for pi0.5 state-in-prompt.

    FlashRT does NOT normalize the state internally —
    ``discretize_pi05_state`` bins whatever it receives onto [-1, 1], so
    the caller must hand over an already-normalized vector (this is the
    openpi convention: Normalize → discretize → 256 language bins).
    Feeding raw joint values silently saturates every bin.

    Inverse of :func:`unnormalize_actions` applied to a state-shaped
    vector, using ``norm_stats["state"]`` (falls back to
    ``norm_stats["actions"]`` when state stats are absent — same
    statistics in openpi checkpoints).
    """
    mode = mode or _resolve_mode(norm_stats)
    entry = norm_stats.get("state") or norm_stats.get("actions")
    if entry is None:
        raise KeyError("norm_stats has neither 'state' nor 'actions' entry")
    lo, hi = _stat_pair(norm_stats, entry, mode)
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != len(lo):
        raise ValueError(
            f"state dim {state.shape[-1]} != stats dim {len(lo)} — "
            "state layout must match training")
    if mode == "q01_q99":
        return (state - lo) / (hi - lo + 1e-6) * 2.0 - 1.0
    return (state - lo) / (hi + 1e-6)
