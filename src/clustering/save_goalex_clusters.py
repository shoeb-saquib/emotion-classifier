"""
Save GoalEx experiment output (cluster_result.json) into saved_clusters in the format
used by the emotion classifier: embeddings+labels CSV and emotion proportions per cluster.

Usage:
  python -m src.clustering.save_goalex_clusters path/to/cluster_result.json [--variation NAME]
  python -m src.clustering.save_goalex_clusters goalex/experiments/ecf_t5/2026-03-02-21-34-13
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.configuration import EMOTIONS
from src.data.ecfdataset import ECFDataset

SAVED_CLUSTERS_DIR = _project_root / "saved_clusters"


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def load_cluster_result(path: Path) -> dict:
    """Load cluster_result.json; return dict mapping cluster_description -> list of text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"cluster_result.json must be a dict, got {type(data)}")
    return data


def text_to_cluster_id(cluster_result: dict) -> dict:
    """
    Build mapping from normalized text to cluster_id (0, 1, 2, ...).
    Order follows iteration over cluster_result keys.
    """
    text_to_id = {}
    for cluster_id, (description, texts) in enumerate(cluster_result.items()):
        for t in texts:
            key = _normalize_text(t)
            if key not in text_to_id:  # first cluster wins if text appears in multiple
                text_to_id[key] = cluster_id
    return text_to_id


def ensure_embeddings(training_set: pd.DataFrame) -> pd.DataFrame:
    """Add 'embedding' column to training_set if missing (in-place on copy, returns copy)."""
    if "embedding" in training_set.columns:
        return training_set
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    training_set = training_set.copy()
    training_set["embedding"] = list(model.encode(training_set["text"].astype(str).tolist()))
    return training_set


def save_embeddings_labels_csv(
    out_path: Path,
    training_set: pd.DataFrame,
    text_to_cluster: dict,
) -> None:
    """Write CSV with columns label, emb_0, emb_1, ... for each row whose text is in a cluster."""
    rows = []
    for _, row in training_set.iterrows():
        key = _normalize_text(str(row["text"]))
        if key not in text_to_cluster:
            continue
        label = text_to_cluster[key]
        emb = np.asarray(row["embedding"], dtype=np.float64)
        rows.append((label, emb))
    if not rows:
        raise ValueError("No training texts matched any cluster; check that cluster_result used the same data.")
    labels = np.array([r[0] for r in rows], dtype=np.int64)
    embeddings = np.array([r[1] for r in rows], dtype=np.float64)
    n, dim = embeddings.shape
    data = {"label": labels}
    for j in range(dim):
        data[f"emb_{j}"] = embeddings[:, j]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Saved {n} rows to {out_path}")


def save_emotion_proportions_csv(
    out_path: Path,
    training_set: pd.DataFrame,
    text_to_cluster: dict,
) -> None:
    """Write CSV with index=cluster_id, columns=EMOTIONS, values=proportions."""
    training_set = training_set.copy()
    training_set["_text_norm"] = training_set["text"].astype(str).map(_normalize_text)
    training_set["_cluster_id"] = training_set["_text_norm"].map(text_to_cluster)
    # Only rows that were assigned to a cluster
    assigned = training_set[training_set["_cluster_id"].notna()]
    if assigned.empty:
        raise ValueError("No training texts matched any cluster.")
    assigned = assigned.astype({"_cluster_id": int})
    prop = (
        assigned.groupby("_cluster_id")["emotion"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    prop = prop.reindex(columns=EMOTIONS, fill_value=0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prop.to_csv(out_path)
    print(f"Saved emotion proportions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save GoalEx cluster_result.json into saved_clusters for the emotion classifier.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to cluster_result.json or to the experiment directory that contains it",
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="goalex",
        help="Variation name for output files (default: goalex). Outputs: {variation}.csv and {variation}_emotion_proportions.csv",
    )
    parser.add_argument(
        "--saved-clusters-dir",
        type=Path,
        default=SAVED_CLUSTERS_DIR,
        help=f"Directory to write outputs (default: {SAVED_CLUSTERS_DIR})",
    )
    args = parser.parse_args()

    path = args.path.resolve()
    if path.is_dir():
        path = path / "cluster_result.json"
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    cluster_result = load_cluster_result(path)
    text_to_cluster = text_to_cluster_id(cluster_result)

    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    training_set = ensure_embeddings(training_set)

    out_dir = Path(args.saved_clusters_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.variation}.csv"
    prop_path = out_dir / f"{args.variation}_emotion_proportions.csv"

    save_embeddings_labels_csv(csv_path, training_set, text_to_cluster)
    save_emotion_proportions_csv(prop_path, training_set, text_to_cluster)

    print(f"Done. Use variation {args.variation!r} in configuration (SELECTED_CLUSTER_VARIATIONS) to use these clusters.")


if __name__ == "__main__":
    main()
