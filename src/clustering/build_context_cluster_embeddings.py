"""
Rebuild saved_clusters CSV embeddings using context pooling (same as EmotionModel.predict).

After GoalEX export writes ``{variation}.csv`` with ``label``, ``conversation_ID``,
``utterance_ID``, and target-only ``emb_*``, run this script to produce
``{variation}_cw{w}_cm{m}.csv`` with pooled embeddings for a given context window
and context method.

Use the same ``SELECTED_CLUSTER_VARIATIONS`` name (e.g. ``goalex_per_emotion_c5_cw2_cm0``)
and matching ``CONTEXT_WINDOWS`` / ``SELECTED_CONTEXT_METHODS`` in ``configuration.py``
so train and test pooling align.

Example::

    python -m src.clustering.build_context_cluster_embeddings \\
        --variation goalex_per_emotion_c5 --context-window 2 --context-method 0

Then set ``SELECTED_CLUSTER_VARIATIONS = ["goalex_per_emotion_c5_cw2_cm0"]`` and
``CONTEXT_WINDOWS = [2]``, ``SELECTED_CONTEXT_METHODS = [0]`` for evaluation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.ecfdataset import ECFDataset
from src.models.context_embedding import pool_utterance_embeddings

TRAIN_EMBEDDINGS_PICKLE = _project_root / "saved_data" / "train_embeddings.pkl"


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    emb_cols = [c for c in df.columns if c.startswith("emb_") and c[4:].isdigit()]
    return sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))


def _load_train_with_embeddings(split: str) -> pd.DataFrame:
    dataset = ECFDataset()
    df = dataset.load_split(split)
    if "embedding" not in df.columns:
        if not TRAIN_EMBEDDINGS_PICKLE.is_file():
            raise FileNotFoundError(
                f"No embedding column and no {TRAIN_EMBEDDINGS_PICKLE}; "
                "run generate_reports once or build train_embeddings.pkl."
            )
        import pickle

        with open(TRAIN_EMBEDDINGS_PICKLE, "rb") as f:
            data = pickle.load(f)
        n = len(df)
        df = df.copy()
        df["embedding"] = list(data[:n])
    return df


def _pooled_embedding_for_row(
    training_set: pd.DataFrame,
    conversation_id: int,
    utterance_id: int,
    context_window: int,
    context_method_id: int,
) -> np.ndarray:
    conv = training_set[training_set["conversation_ID"] == conversation_id].sort_values(
        "utterance_ID"
    )
    row = conv[(conv["utterance_ID"] == utterance_id)]
    if row.empty:
        raise ValueError(
            f"No train row for conversation_ID={conversation_id}, utterance_ID={utterance_id}"
        )
    utterances = conv.loc[conv["utterance_ID"] <= utterance_id]
    emb_seq = np.asarray(list(utterances.embedding), dtype=np.float64)
    return pool_utterance_embeddings(emb_seq, context_window, context_method_id)


def build_context_cluster_embeddings(
    variation: str,
    context_window: int,
    context_method_id: int,
    *,
    saved_clusters_dir: Path | None = None,
    train_split: str = "train",
) -> Path:
    """
    Read ``{variation}.csv``, write ``{variation}_cw{w}_cm{m}.csv`` with pooled emb_*.
    Copy emotion proportions to the matching suffixed name.
    Returns the path to the written CSV.
    """
    base_dir = Path(saved_clusters_dir) if saved_clusters_dir else _project_root / "saved_clusters"
    in_path = base_dir / f"{variation}.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"cluster CSV not found: {in_path}")

    prop_src = base_dir / f"{variation}_emotion_proportions.csv"
    if not prop_src.is_file():
        raise FileNotFoundError(f"emotion proportions not found: {prop_src}")

    df = pd.read_csv(in_path)
    for col in ("label", "conversation_ID", "utterance_ID"):
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r} in {in_path}")
    emb_cols = _embedding_columns(df)
    if not emb_cols:
        raise ValueError(f"No emb_* columns in {in_path}")

    training_set = _load_train_with_embeddings(train_split)

    out_rows: list[dict] = []
    for _, row in df.iterrows():
        cid = int(row["conversation_ID"])
        uid = int(row["utterance_ID"])
        lab = int(row["label"])
        src_emotion = str(row.get("source_emotion", ""))
        pooled = _pooled_embedding_for_row(
            training_set, cid, uid, context_window, context_method_id
        )
        rec = {
            "label": lab,
            "conversation_ID": cid,
            "utterance_ID": uid,
            "source_emotion": src_emotion,
        }
        for j, v in enumerate(pooled):
            rec[f"emb_{j}"] = v
        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    suffix = f"_cw{context_window}_cm{context_method_id}"
    out_var = f"{variation}{suffix}"
    out_path = base_dir / f"{out_var}.csv"
    out_prop = base_dir / f"{out_var}_emotion_proportions.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    shutil.copy2(prop_src, out_prop)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(f"Copied emotion proportions to {out_prop}")
    print(f"Use variation {out_var!r} in SELECTED_CLUSTER_VARIATIONS.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild cluster CSV embeddings with context pooling (matches EmotionModel).",
    )
    parser.add_argument(
        "--variation",
        type=str,
        required=True,
        help="Base variation name (reads saved_clusters/{variation}.csv).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        required=True,
        help="Same as CONTEXT_WINDOWS / EmotionModel.context_window (e.g. 2 for two prior utterances).",
    )
    parser.add_argument(
        "--context-method",
        type=int,
        required=True,
        help="0 = mean pool, 1 = exponential weights (same as CONTEXT_METHODS keys).",
    )
    parser.add_argument(
        "--saved-clusters-dir",
        type=Path,
        default=None,
        help="Default: project root / saved_clusters",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="ECF split to load embeddings from (default: train).",
    )
    args = parser.parse_args()

    build_context_cluster_embeddings(
        args.variation,
        args.context_window,
        args.context_method,
        saved_clusters_dir=args.saved_clusters_dir,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
