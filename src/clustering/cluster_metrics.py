"""Evaluate clustering results with purity, NMI, homogeneity, and completeness."""

import math
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

from bertopic import BERTopic


def evaluate_clustering(
    name: str,
    topics: list,
    topic_model: BERTopic,
    training_set: pd.DataFrame,
) -> dict:
    """
    Compute clustering metrics and per-topic details given topic assignments and labels.

    Args:
        name: Variation name (for the result dict).
        topics: List of topic ID per document (same order as training_set).
        topic_model: Fitted BERTopic model (for get_topic).
        training_set: DataFrame with "text" and "emotion" columns; must match len(topics).

    Returns:
        Dict with name, n_topics, n_outliers, mean_purity, weighted_purity, nmi,
        homogeneity, completeness, topic_infos, topic_sizes.
    """
    df = training_set.copy()
    df["topic"] = topics

    groups = df.groupby("topic")
    topic_infos = []
    purities = []

    for topic_id, group in groups:
        size = len(group)
        emotion_counts = group["emotion"].value_counts(normalize=True)
        dominant_emotion = emotion_counts.index[0]
        purity = float(emotion_counts.iloc[0])
        purities.append(purity)

        top_words = topic_model.get_topic(topic_id)[0][0]

        topic_infos.append({
            "topic_id": topic_id,
            "size": size,
            "dominant_emotion": dominant_emotion,
            "purity": purity,
            "emotion_dist": emotion_counts.to_dict(),
            "top_words": top_words,
        })

    n_topics = len([t for t in df["topic"].unique() if t != -1])
    n_outliers = int((df["topic"] == -1).sum())
    mean_purity = sum(purities) / len(purities) if purities else 0.0
    total = df.shape[0]
    weighted_purity = (
        sum(info["purity"] * (info["size"] / total) for info in topic_infos)
        if total else 0.0
    )

    labels_true = df["emotion"].astype(str).values
    labels_pred = df["topic"].astype(str).values
    nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    if math.isnan(nmi):
        nmi = homogeneity = completeness = 0.0
    elif math.isnan(homogeneity):
        homogeneity = 0.0
    elif math.isnan(completeness):
        completeness = 0.0

    return {
        "name": name,
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "mean_purity": mean_purity,
        "weighted_purity": weighted_purity,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "topic_infos": topic_infos,
        "topic_sizes": df["topic"].value_counts().sort_index().to_dict(),
    }
