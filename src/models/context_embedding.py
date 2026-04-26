"""Pool per-utterance embeddings into a single context vector (matches EmotionModel.predict)."""

from __future__ import annotations

import numpy as np


def pool_utterance_embeddings(
    embeddings: np.ndarray,
    context_window: int,
    context_method_id: int,
) -> np.ndarray:
    """
    Pool a sequence of utterance embeddings using the same rules as EmotionModel.predict.

    *embeddings* is shape (n, d) for n utterances in conversation order up to and including
    the target. If n > context_window + 1, only the last context_window + 1 rows are used.

    context_method_id: 0 = mean, 1 = exponential weights (then L2-normalize).
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if context_window < len(arr) - 1:
        arr = arr[-(context_window + 1) :]

    if context_method_id == 0:
        return np.mean(arr, axis=0)

    if context_method_id == 1:
        weights = np.array([2**i for i in range(len(arr))], dtype=float)
        out = np.average(arr, axis=0, weights=weights)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    raise ValueError(f"Context method not recognized: {context_method_id!r}")
