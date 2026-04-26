"""
KNN classifier that maps embeddings to clusters, then returns per-cluster emotion probabilities.

Given a BERTopic variation (code_name), loads embeddings + cluster labels and emotion proportions
from saved_clusters, fits a KNeighborsClassifier on that data, and for a test embedding returns
the emotion probability vector of the predicted cluster.

For GoalEX-backed variations (names starting with ``goalex``), optional hybrid KNN is used:
d_total = alpha * (1 - cos(e_query, e_train)) + (1 - alpha) * (1 - p_query[c_train]),
where p_query is a softmax over cosine similarities to cluster centroids.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.configuration import EMOTIONS


def _variation_is_goalex(variation: str) -> bool:
    return (variation or "").startswith("goalex")


def _variation_is_goalex_per_emotion(variation: str) -> bool:
    return (variation or "").startswith("goalex_per_emotion")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / s if s > 0 else np.full_like(x, 1.0 / len(x))


def _default_saved_clusters_dir() -> Path:
    """Project root / saved_clusters; walk up from this file until we find a dir containing 'src'."""
    _this_dir = Path(__file__).resolve().parent
    _root = _this_dir
    while _root != _root.parent and (_root / "src").is_dir() is False:
        _root = _root.parent
    if (_root / "src").is_dir():
        return _root / "saved_clusters"
    return _this_dir.parent.parent.parent / "saved_clusters"


def _embedding_columns(df: pd.DataFrame) -> list:
    """Return emb_* column names in numeric order (emb_0, emb_1, ...)."""
    emb_cols = [c for c in df.columns if c.startswith("emb_") and c[4:].isdigit()]
    return sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))


class ClusterKNNClassifier:
    """
    Given a clustering variation, loads saved embeddings + cluster labels and emotion proportions,
    fits a KNN on the loaded data, and predicts emotion probabilities for test embeddings
    via the assigned cluster's emotion distribution.
    """

    def __init__(
        self,
        variation: str,
        saved_clusters_dir: Union[Path, str, None] = None,
        n_neighbors: int = 5,
        *,
        goalex_hybrid_alpha: float = 0.7,
        goalex_softmax_tau: float = 1.0,
        goalex_per_emotion_fusion_topk: int = 2,
    ):
        self.variation = variation
        self.saved_clusters_dir = Path(saved_clusters_dir) if saved_clusters_dir else _default_saved_clusters_dir()
        self.n_neighbors = n_neighbors
        self._goalex_hybrid_alpha = float(goalex_hybrid_alpha)
        self._goalex_softmax_tau = float(goalex_softmax_tau)

        self._fusion_topk = int(goalex_per_emotion_fusion_topk)

        self._cluster_labels: np.ndarray = None
        self._embeddings: np.ndarray = None
        self._emotion_proportions: pd.DataFrame = None
        self._source_emotions: Optional[np.ndarray] = None
        self._knn: Optional[KNeighborsClassifier] = None
        self._use_goalex_hybrid: bool = False
        self._centroids: Optional[np.ndarray] = None
        self._goalex_cluster_ids: Optional[np.ndarray] = None
        self._cluster_label_to_p_idx: Optional[dict[int, int]] = None
        self._cluster_to_source_emotion: dict[int, str] = {}

        self._load_saved_data()

    def _load_saved_data(self) -> None:
        csv_path = self.saved_clusters_dir / f"{self.variation}.csv"
        prop_path = self.saved_clusters_dir / f"{self.variation}_emotion_proportions.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Embeddings and labels not found: {csv_path}")
        if not prop_path.exists():
            raise FileNotFoundError(f"Emotion proportions not found: {prop_path}")

        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError(f"CSV must have a 'label' column: {csv_path}")
        emb_cols = _embedding_columns(df)
        if not emb_cols:
            raise ValueError(f"CSV must have emb_0, emb_1, ... columns: {csv_path}")

        self._cluster_labels = df["label"].values.astype(np.int64)
        self._embeddings = df[emb_cols].values.astype(np.float64)

        if "source_emotion" in df.columns:
            self._source_emotions = df["source_emotion"].fillna("").astype(str).values
        else:
            self._source_emotions = None

        self._emotion_proportions = pd.read_csv(prop_path, index_col=0)
        self._emotion_proportions = self._emotion_proportions.reindex(columns=EMOTIONS, fill_value=0.0)

    def _fit_goalex_hybrid(self) -> None:
        labels = self._cluster_labels
        prop_clusters = {int(i) for i in self._emotion_proportions.index}
        cluster_ids: list[int] = []
        centroids: list[np.ndarray] = []
        cluster_to_source: dict[int, str] = {}

        for c in sorted(int(x) for x in np.unique(labels) if x != -1):
            if c not in prop_clusters:
                continue
            mask = labels == c
            if not np.any(mask):
                continue
            cluster_ids.append(c)
            centroids.append(self._embeddings[mask].mean(axis=0))

            if self._source_emotions is not None:
                vals = self._source_emotions[mask]
                vals = vals[vals != ""]
                if len(vals) > 0:
                    uniq, cnt = np.unique(vals, return_counts=True)
                    cluster_to_source[int(c)] = str(uniq[np.argmax(cnt)])

        if not cluster_ids:
            self._use_goalex_hybrid = False
            self._centroids = None
            self._goalex_cluster_ids = None
            self._cluster_label_to_p_idx = None
            self._cluster_to_source_emotion = {}
            return

        self._centroids = np.asarray(centroids, dtype=np.float64)
        self._goalex_cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
        self._cluster_label_to_p_idx = {int(cid): i for i, cid in enumerate(cluster_ids)}
        self._cluster_to_source_emotion = cluster_to_source
        self._use_goalex_hybrid = True

    def fit(self, training_set: pd.DataFrame = None) -> "ClusterKNNClassifier":
        if _variation_is_goalex(self.variation):
            self._fit_goalex_hybrid()
            if self._use_goalex_hybrid:
                self._knn = None
                return self
        self._use_goalex_hybrid = False
        self._knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._knn.fit(self._embeddings, self._cluster_labels)
        return self

    def _predict_cluster_goalex(self, arr: np.ndarray) -> np.ndarray:
        n_train = self._embeddings.shape[0]
        k = min(self.n_neighbors, n_train)
        labels = self._cluster_labels
        centroids = self._centroids
        tau = max(self._goalex_softmax_tau, 1e-8)
        alpha = min(max(self._goalex_hybrid_alpha, 0.0), 1.0)

        train_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
        )
        c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        n_centroids = len(self._goalex_cluster_ids)
        uniform_p = 1.0 / max(n_centroids, 1)

        idx_map = np.array(
            [self._cluster_label_to_p_idx.get(int(lb), -1) for lb in labels],
            dtype=np.int32,
        )

        preds = []
        for q in arr:
            q = q.reshape(1, -1)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            cos_c = (q_norm @ c_norm.T).ravel()
            p = _softmax(cos_c / tau)

            cos_e = (train_norm @ q_norm.T).ravel()
            d_embed = 1.0 - cos_e

            p_train = np.where(idx_map >= 0, p[idx_map], uniform_p)
            d_cluster = 1.0 - p_train

            d_total = alpha * d_embed + (1.0 - alpha) * d_cluster
            nn = np.argpartition(d_total, k - 1)[:k]
            vote_labels = labels[nn]
            vals, counts = np.unique(vote_labels, return_counts=True)
            preds.append(int(vals[np.argmax(counts)]))
        return np.asarray(preds, dtype=np.int64)

    def _topk_mean_similarity(self, values: np.ndarray) -> float:
        """Mean of the top-k centroid similarities for one emotion (fixed aggregation)."""
        if values.size == 0:
            return -1e9
        k = max(1, min(self._fusion_topk, values.size))
        top = np.partition(values, values.size - k)[-k:]
        return float(np.mean(top))

    def _emotion_scores_from_centroids_one(self, q: np.ndarray) -> np.ndarray:
        if self._centroids is None or self._goalex_cluster_ids is None:
            return self._fallback_probas("mean")
        if not self._cluster_to_source_emotion:
            return self._fallback_probas("mean")

        q = q.reshape(1, -1)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        c_norm = self._centroids / (np.linalg.norm(self._centroids, axis=1, keepdims=True) + 1e-12)
        sims = (q_norm @ c_norm.T).ravel()

        raw = np.full(len(EMOTIONS), -1e9, dtype=np.float64)
        for i, emo in enumerate(EMOTIONS):
            idxs = [
                j for j, cid in enumerate(self._goalex_cluster_ids)
                if self._cluster_to_source_emotion.get(int(cid), "") == emo
            ]
            if idxs:
                raw[i] = self._topk_mean_similarity(sims[np.asarray(idxs, dtype=np.int64)])

        if np.all(raw <= -1e8):
            return self._fallback_probas("mean")
        return _softmax(raw)

    def predict_emotion_scores_from_centroids(
        self, embedding: np.ndarray
    ) -> Optional[Union[dict[str, float], list[dict[str, float]]]]:
        if not (_variation_is_goalex_per_emotion(self.variation) and self._use_goalex_hybrid):
            return None
        if not self._cluster_to_source_emotion:
            return None

        arr = np.asarray(embedding, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)

        result = []
        for q in arr:
            vec = self._emotion_scores_from_centroids_one(q)
            result.append(dict(zip(EMOTIONS, vec.astype(float))))
        return result[0] if single else result

    def predict_cluster(self, embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._use_goalex_hybrid:
            return self._predict_cluster_goalex(arr)
        return self._knn.predict(arr)

    def predict_emotion_probas(
        self, embedding: np.ndarray, noise_fallback: str = "mean"
    ) -> dict[str, float]:
        if self._knn is None and not self._use_goalex_hybrid:
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
