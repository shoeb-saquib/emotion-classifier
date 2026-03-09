"""
KNN classifier that maps embeddings to clusters, then returns per-cluster emotion probabilities.

Given a BERTopic variation (code_name), loads cluster labels and emotion proportions from
saved_clusters, fits a KNeighborsClassifier on (training_embeddings, cluster_labels),
and for a test embedding returns the emotion probability vector of the predicted cluster.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.configuration import EMOTIONS


def _default_saved_clusters_dir() -> Path:
    """Project root / saved_clusters; walk up from this file until we find a dir containing 'src'."""
    _this_dir = Path(__file__).resolve().parent
    _root = _this_dir
    while _root != _root.parent and (_root / "src").is_dir() is False:
        _root = _root.parent
    if (_root / "src").is_dir():
        return _root / "saved_clusters"
    return _this_dir.parent.parent.parent / "saved_clusters"


class ClusterKNNClassifier:
    """
    Given a clustering variation, loads its saved cluster labels and emotion proportions,
    fits a KNN on (training_embeddings, cluster_labels), and predicts emotion probabilities
    for test embeddings via the assigned cluster's emotion distribution.
    """

    def __init__(
        self,
        variation: str,
        saved_clusters_dir: Union[Path, str, None] = None,
        n_neighbors: int = 5,
    ):
        """
        Args:
            variation: Code name of the BERTopic variation (e.g. from BERTopicConfig.code_name()).
                Used to load {variation}.npy and {variation}_emotion_proportions.csv.
            saved_clusters_dir: Directory containing the .npy and _emotion_proportions.csv files.
                Defaults to project root / saved_clusters.
            n_neighbors: Number of neighbors for KNeighborsClassifier.
        """
        self.variation = variation
        self.saved_clusters_dir = Path(saved_clusters_dir) if saved_clusters_dir else _default_saved_clusters_dir()
        self.n_neighbors = n_neighbors

        self._cluster_labels: np.ndarray = None  # shape (n_train,)
        self._emotion_proportions: pd.DataFrame = None  # index = topic_id, columns = EMOTIONS
        self._knn: KNeighborsClassifier = None

        self._load_saved_data()

    def _load_saved_data(self) -> None:
        """Load cluster labels and emotion proportions from saved_clusters."""
        labels_path = self.saved_clusters_dir / f"{self.variation}.npy"
        prop_path = self.saved_clusters_dir / f"{self.variation}_emotion_proportions.csv"

        if not labels_path.exists():
            raise FileNotFoundError(f"Cluster labels not found: {labels_path}")
        if not prop_path.exists():
            raise FileNotFoundError(f"Emotion proportions not found: {prop_path}")

        self._cluster_labels = np.load(labels_path)
        self._emotion_proportions = pd.read_csv(prop_path, index_col=0)
        # Ensure columns match EMOTIONS order
        self._emotion_proportions = self._emotion_proportions.reindex(columns=EMOTIONS, fill_value=0.0)

    def fit(self, training_set: pd.DataFrame) -> "ClusterKNNClassifier":
        """
        Fit KNN: X = training embeddings, y = cluster labels from saved_clusters.

        training_set must have an "embedding" column and the same length and row order
        as the data that was used to produce the saved cluster labels.

        Args:
            training_set: DataFrame with "embedding" column (same length as saved labels).

        Returns:
            self (for chaining).
        """
        if "embedding" not in training_set.columns:
            raise ValueError("training_set must have an 'embedding' column")
        if len(training_set) != len(self._cluster_labels):
            raise ValueError(
                f"training_set length ({len(training_set)}) must match saved cluster labels ({len(self._cluster_labels)})"
            )

        X = np.vstack(np.asarray(training_set["embedding"].tolist(), dtype=np.float64))
        y = self._cluster_labels

        self._knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._knn.fit(X, y)
        return self

    def predict_cluster(self, embedding: np.ndarray) -> np.ndarray:
        """
        Predict cluster id(s) for the given embedding(s).

        Args:
            embedding: Shape (n_features,) or (n_samples, n_features).

        Returns:
            Cluster id(s), shape (n_samples,) or scalar if single embedding.
        """
        arr = np.asarray(embedding, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._knn.predict(arr)

    def predict_emotion_probas(
        self, embedding: np.ndarray, noise_fallback: str = "mean"
    ) -> dict[str, float]:
        """
        Classify the embedding to a cluster and return that cluster's emotion probabilities.

        Args:
            embedding: Shape (n_features,) or (n_samples, n_features).
            noise_fallback: How to handle cluster id -1 (noise): "mean" = use mean proportions
                over all clusters, "uniform" = 1/n_emotions. Ignored if no -1 in predictions.

        Returns:
            Dict mapping each emotion (str) to probability (float). Keys follow EMOTIONS.
            For a single embedding returns one dict; for batch, returns a list of dicts.
        """
        if self._knn is None:
            raise ValueError("Classifier must be fit before predicting")

        clusters = self.predict_cluster(embedding)
        single = np.ndim(embedding) == 1
        if single:
            clusters = np.atleast_1d(clusters)

        fallback = self._fallback_probas(noise_fallback)
        result = []
        for c in clusters:
            if c == -1 or c not in self._emotion_proportions.index:
                vec = fallback
            else:
                vec = self._emotion_proportions.loc[c].values.astype(np.float64)
            result.append(dict(zip(EMOTIONS, vec)))

        return result[0] if single else result

    def _fallback_probas(self, kind: str) -> np.ndarray:
        if kind == "mean":
            return self._emotion_proportions.values.mean(axis=0).astype(np.float64)
        if kind == "uniform":
            n = len(EMOTIONS)
            return np.full(n, 1.0 / n, dtype=np.float64)
        raise ValueError(f"noise_fallback must be 'mean' or 'uniform', got {kind!r}")
