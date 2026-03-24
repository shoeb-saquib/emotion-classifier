"""Fit one BERTopic clustering run from a config and save embeddings + cluster labels to disk."""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Find project root: walk up from this file until we find a directory containing "src"
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir
while _project_root != _project_root.parent and (_project_root / "src").is_dir() is False:
    _project_root = _project_root.parent
if (_project_root / "src").is_dir():
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
else:
    _project_root = _this_dir.parent.parent.parent.parent  # fallback: bertopic -> clustering -> src -> root

import numpy as np
import pandas as pd

from bertopic import BERTopic

from src.clustering.bertopic.bertopic_config import BERTopicConfig, BERTOPIC_VARIATIONS
from src.data.ecfdataset import ECFDataset
from src.configuration import EMOTIONS

# saved_clusters at project root (same root we use for sys.path)
SAVED_CLUSTERS_DIR = _project_root / "saved_clusters"


def fit_cluster(
    config: BERTopicConfig,
    docs: list,
    saved_clusters_dir: Optional[Path] = None,
    training_set: Optional[pd.DataFrame] = None,
) -> Tuple[list, BERTopic]:
    """
    Build model from config, fit on documents, save training embeddings + cluster labels to disk.

    Args:
        config: BERTopicConfig to build and fit.
        docs: List of document strings.
        saved_clusters_dir: Directory to save outputs; default is project_root / saved_clusters.
        training_set: Optional DataFrame with same row order as docs. If it has "emotion",
            saves per-cluster emotion proportions. If it has "embedding", saves a CSV with
            columns label, emb_0, emb_1, ... for use by ClusterKNNClassifier.

    Returns:
        (topics, fitted_model): topics is a list of topic IDs (int) per document;
        fitted_model is the fitted BERTopic instance (for get_topic etc.).
    """
    out_dir = saved_clusters_dir if saved_clusters_dir is not None else SAVED_CLUSTERS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_model = config.build(use_llm=False)
    topics, _ = topic_model.fit_transform(docs)

    # Save training embeddings + cluster labels as one CSV (label, emb_0, emb_1, ...)
    if training_set is not None:
        if "embedding" not in training_set.columns:
            embedder = topic_model.embedding_model
            embed_fn = getattr(embedder, "embed_documents", None) or getattr(embedder, "embed", None)
            if embed_fn is None:
                raise RuntimeError("BERTopic embedding_model has no embed_documents or embed method")
            emb = np.asarray(embed_fn(docs), dtype=np.float64)
            training_set = training_set.copy()
            training_set["embedding"] = [emb[i] for i in range(emb.shape[0])]
        embeddings = np.asarray(training_set["embedding"].tolist(), dtype=np.float64)
        n, dim = embeddings.shape
        data = {"label": np.asarray(topics, dtype=np.int64)[:n]}
        for j in range(dim):
            data[f"emb_{j}"] = embeddings[:, j]
        pd.DataFrame(data).to_csv(out_dir / f"{config.code_name()}.csv", index=False)
        print(f"Saved embeddings and cluster labels to {out_dir / config.code_name()}.csv")

    if training_set is not None and "emotion" in training_set.columns:
        df = training_set.copy()
        df["topic"] = topics
        # Proportion of each emotion in each cluster (rows=topic_id, columns=emotion)
        prop = df.groupby("topic")["emotion"].value_counts(normalize=True).unstack(fill_value=0.0)
        prop = prop.reindex(columns=EMOTIONS, fill_value=0.0)
        prop_path = out_dir / f"{config.code_name()}_emotion_proportions.csv"
        prop.to_csv(prop_path)
        print(f"Saved emotion proportions to {prop_path}")

    return topics, topic_model


def main() -> None:
    """
    Load training set, fit all variations in BERTOPIC_VARIATIONS, and save
    embeddings + cluster labels to saved_clusters/{code_name}.csv for later use by ClusterKNNClassifier.
    """
    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    docs = list(training_set["text"])
    print(f"Loaded {len(docs)} documents from training set.")

    for config in BERTOPIC_VARIATIONS:
        print(f"\n>>> Fitting: {config.name()}")
        fit_cluster(config, docs, training_set=training_set)

    print(f"\nDone. Cluster labels saved to {SAVED_CLUSTERS_DIR}")


if __name__ == "__main__":
    main()
