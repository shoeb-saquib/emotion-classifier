"""
Generate a comparison report for all GoalEX-based clustering variations.

Reads:
  - saved_clusters/goalex_*.csv  (embeddings + cluster labels)
  - GoalEX cluster_result.json under GOALEX_DATA_EXPERIMENTS_DIR / <variation> / ...
  - training labels via ECFDataset

Writes:
  - reports/goalex_variation_report.txt

The report mirrors the BERTopic variation report: overall metrics table plus
per-variation details (without topic-word labels).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.ecfdataset import ECFDataset
from src.clustering.goalex_cluster import (
    GOALEX_DATA_EXPERIMENTS_DIR,
    SAVED_CLUSTERS_DIR,
    load_cluster_result,
    text_to_cluster_id,
)


REPORTS_DIR = _PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = REPORTS_DIR / "goalex_variation_report.txt"


METRICS_DOCUMENTATION = """
METRIC DEFINITIONS
-------------------------------------------------

1. PURITY (per-cluster and weighted)
   - Per-cluster purity: For each cluster, the fraction of documents that have the
     *dominant* emotion (the most frequent emotion in that cluster).
     Formula: purity(cluster k) = max_emotion count_in_cluster_k(emotion) / size(cluster k).
   - Mean purity: Average of per-cluster purities over all clusters (each cluster weighted equally).
   - Weighted purity: Purity weighted by cluster size.
     Formula: sum over clusters of [ purity(cluster k) * (size(cluster k) / total docs) ].

2. NMI (Normalized Mutual Information)
   - Measures how much knowing the cluster ID reduces uncertainty about the emotion label.
   - Formula: NMI(emotion; cluster) = MI(emotion, cluster) / average(H(emotion), H(cluster)).
     MI = mutual information, H = entropy. We use sklearn's normalized_mutual_info_score
     with average_method="arithmetic".
   - Range: 0 (cluster and emotion independent) to 1 (cluster perfectly predicts emotion).

3. HOMOGENEITY
   - Each cluster contains only one class (emotion)?
   - Formula: 1 - H(emotion | cluster) / H(emotion). Homogeneity is 1 when every cluster
     has documents of only one emotion.

4. COMPLETENESS
   - Each emotion is assigned to only one cluster?
   - Formula: 1 - H(cluster | emotion) / H(cluster). Completeness is 1 when all documents
     of a given emotion fall in a single cluster.

5. CLUSTERS AND OUTLIERS
   - Clusters: Number of distinct cluster IDs excluding -1.
   - Outliers: Number of documents assigned cluster -1 (texts not covered by any GoalEX description).
"""


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def _find_cluster_result_for_variation(variation: str) -> Path | None:
    """Find the most recent cluster_result.json for a given GoalEX variation."""
    base = GOALEX_DATA_EXPERIMENTS_DIR / variation
    # Direct file
    direct = base / "cluster_result.json"
    if direct.is_file():
        return direct
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    for d in sorted(subdirs, key=lambda p: p.name, reverse=True):
        candidate = d / "cluster_result.json"
        if candidate.is_file():
            return candidate
    return None


def evaluate_goalex_clustering(
    name: str, cluster_result_path: Path, training_set: pd.DataFrame
) -> Dict:
    """
    Compute clustering metrics and per-cluster details given a GoalEX cluster_result.json.

    Args:
        name: Variation name.
        cluster_result_path: Path to cluster_result.json for this variation.
        training_set: DataFrame with "text" and "emotion" columns.
    """
    cluster_result = load_cluster_result(cluster_result_path)
    # Map cluster ids to their natural language descriptions
    cluster_descriptions: List[str] = list(cluster_result.keys())
    text_to_cluster = text_to_cluster_id(cluster_result)

    df = training_set.copy()
    df["_text_norm"] = df["text"].astype(str).map(_normalize_text)
    df["topic"] = df["_text_norm"].map(text_to_cluster).fillna(-1).astype(int)

    groups = df.groupby("topic")
    topic_infos = []
    purities = []

    for topic_id, group in groups:
        size = len(group)
        emotion_counts = group["emotion"].value_counts(normalize=True)
        dominant_emotion = emotion_counts.index[0]
        purity = float(emotion_counts.iloc[0])
        purities.append(purity)

        # Use GoalEX description as the "topic label" when available
        label = (
            cluster_descriptions[topic_id]
            if topic_id != -1 and 0 <= topic_id < len(cluster_descriptions)
            else "(outlier / unmatched)"
        )

        topic_infos.append(
            {
                "topic_id": topic_id,
                "size": size,
                "dominant_emotion": dominant_emotion,
                "purity": purity,
                "emotion_dist": emotion_counts.to_dict(),
                "label": label,
            }
        )

    n_topics = len([t for t in df["topic"].unique() if t != -1])
    n_outliers = int((df["topic"] == -1).sum())
    mean_purity = sum(purities) / len(purities) if purities else 0.0
    total = df.shape[0]
    weighted_purity = (
        sum(info["purity"] * (info["size"] / total) for info in topic_infos)
        if total
        else 0.0
    )

    labels_true = df["emotion"].astype(str).values
    labels_pred = df["topic"].astype(str).values
    nmi = normalized_mutual_info_score(
        labels_true, labels_pred, average_method="arithmetic"
    )
    homog = homogeneity_score(labels_true, labels_pred)
    compl = completeness_score(labels_true, labels_pred)
    if math.isnan(nmi):
        nmi = homog = compl = 0.0
    elif math.isnan(homog):
        homog = 0.0
    elif math.isnan(compl):
        compl = 0.0

    return {
        "name": name,
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "mean_purity": mean_purity,
        "weighted_purity": weighted_purity,
        "nmi": nmi,
        "homogeneity": homog,
        "completeness": compl,
        "topic_infos": topic_infos,
        "topic_sizes": df["topic"].value_counts().sort_index().to_dict(),
    }


def write_goalex_report(all_results: List[Dict], out_path: Path) -> None:
    """Write all GoalEX variation results to a single text file for comparison."""
    with open(out_path, "w") as f:
        f.write("GoalEX Variation Comparisons\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(METRICS_DOCUMENTATION)
        f.write("\n")

        if not all_results:
            f.write("No GoalEX variations found in saved_clusters.\n")
            return

        # Summary table
        col_topics = 7
        col_outliers = 9
        col_purity = 11  # 10 for value + 1 for marker
        col_nmi = 8
        col_homog = 8
        col_compl = 8
        col_name = max(len(r["name"]) for r in all_results) if all_results else 40
        sep_len = (
            col_name
            + col_topics
            + col_outliers
            + col_purity
            + col_nmi
            + col_homog
            + col_compl
            + 6
        )
        valid = [r for r in all_results if "error" not in r]
        max_purity = (
            max(
                (
                    r.get("weighted_purity", r.get("mean_purity", 0))
                    for r in valid
                ),
                default=0,
            )
            if valid
            else 0
        )
        nmi_vals = [
            r.get("nmi", 0)
            for r in valid
            if not (
                isinstance(r.get("nmi"), float)
                and math.isnan(r.get("nmi", 0))
            )
        ]
        homog_vals = [
            r.get("homogeneity", 0)
            for r in valid
            if not (
                isinstance(r.get("homogeneity"), float)
                and math.isnan(r.get("homogeneity", 0))
            )
        ]
        compl_vals = [
            r.get("completeness", 0)
            for r in valid
            if not (
                isinstance(r.get("completeness"), float)
                and math.isnan(r.get("completeness", 0))
            )
        ]
        max_nmi = max(nmi_vals, default=0)
        max_homog = max(homog_vals, default=0)
        max_compl = max(compl_vals, default=0)

        f.write("-" * sep_len + "\n")
        f.write(
            f"{'Variation':<{col_name}} {'Topics':>{col_topics}} {'Outliers':>{col_outliers}} "
            f"{'Purity':>{col_purity}} {'NMI':>{col_nmi}} {'Homog':>{col_homog}} {'Compl':>{col_compl}}\n"
        )
        f.write("-" * sep_len + "\n")
        for r in all_results:
            purity = r.get("weighted_purity", r.get("mean_purity", 0))
            nmi = r.get("nmi", 0)
            homog = r.get("homogeneity", 0)
            compl = r.get("completeness", 0)
            if isinstance(nmi, float) and math.isnan(nmi):
                nmi = 0.0
            if isinstance(homog, float) and math.isnan(homog):
                homog = 0.0
            if isinstance(compl, float) and math.isnan(compl):
                compl = 0.0
            p_mark = "*" if (valid and purity == max_purity) else " "
            n_mark = "*" if (valid and nmi == max_nmi) else " "
            h_mark = "*" if (valid and homog == max_homog) else " "
            c_mark = "*" if (valid and compl == max_compl) else " "
            f.write(
                f"{r['name']:<{col_name}} {r['n_topics']:>{col_topics}} {r['n_outliers']:>{col_outliers}} "
                f"{purity:>{col_purity - 1}.4f}{p_mark} {nmi:>{col_nmi - 1}.4f}{n_mark} "
                f"{homog:>{col_homog - 1}.4f}{h_mark} {compl:>{col_compl - 1}.4f}{c_mark}\n"
            )
        f.write("(* = best in column)\n\n")

        # Per-variation detail
        for r in all_results:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"VARIATION: {r['name']}\n")
            f.write("=" * 80 + "\n")
            if "error" in r:
                f.write(f"ERROR: {r['error']}\n\n")
                continue
            _nmi = r.get("nmi", 0)
            _homog = r.get("homogeneity", 0)
            _compl = r.get("completeness", 0)
            if isinstance(_nmi, float) and math.isnan(_nmi):
                _nmi = 0.0
            if isinstance(_homog, float) and math.isnan(_homog):
                _homog = 0.0
            if isinstance(_compl, float) and math.isnan(_compl):
                _compl = 0.0
            f.write(
                f"Clusters: {r['n_topics']}  |  Outliers: {r['n_outliers']}  |  "
                f"Purity: {r['weighted_purity']:.4f}  |  NMI: {_nmi:.4f}\n"
            )
            f.write(
                f"  Homogeneity: {_homog:.4f}  |  Completeness: {_compl:.4f}\n\n"
            )

            for info in r["topic_infos"]:
                # Skip listing the synthetic "-1" outlier cluster to keep the report focused
                # on actual GoalEX descriptions. The outlier count is still reflected in
                # the summary metrics table.
                if info["topic_id"] == -1:
                    continue
                f.write(f"  --- Cluster {info['topic_id']} (n={info['size']}) ---\n")
                f.write(
                    f"      Dominant emotion: {info['dominant_emotion']} "
                    f"(purity={info['purity']:.4f})\n"
                )
                f.write(f"      Cluster description: {info['label']}\n")
                f.write(
                    f"      Emotion distribution: {info['emotion_dist']}\n\n"
                )

    print(f"GoalEX variation report written to {out_path}")


def main() -> None:
    # Find all goalex_* variations that already have saved_clusters CSVs
    if not SAVED_CLUSTERS_DIR.exists():
        print(f"No saved_clusters directory found at {SAVED_CLUSTERS_DIR}")
        return

    # Only treat the *embedding+labels* CSVs as variations. The *_emotion_proportions.csv
    # files live alongside them and should NOT be interpreted as separate variations.
    variation_csvs = sorted(
        p
        for p in SAVED_CLUSTERS_DIR.glob("goalex_*.csv")
        if p.is_file() and not p.stem.endswith("_emotion_proportions")
    )
    if not variation_csvs:
        print("No goalex_* variations found in saved_clusters.")
        return

    variations = [p.stem for p in variation_csvs]
    print(f"Found {len(variations)} GoalEX variation(s): {', '.join(variations)}")

    dataset = ECFDataset()
    training_set = dataset.load_split("train")

    all_results: List[Dict] = []
    for variation in variations:
        print(f"\n>>> Evaluating GoalEX variation: {variation}")
        cluster_result_path = _find_cluster_result_for_variation(variation)
        if cluster_result_path is None:
            print(
                f"  WARNING: No cluster_result.json found for {variation} under {GOALEX_DATA_EXPERIMENTS_DIR}."
            )
            all_results.append(
                {
                    "name": variation,
                    "error": f"cluster_result.json not found under {GOALEX_DATA_EXPERIMENTS_DIR / variation}",
                    "n_topics": 0,
                    "n_outliers": 0,
                    "mean_purity": 0.0,
                    "weighted_purity": 0.0,
                    "nmi": 0.0,
                    "homogeneity": 0.0,
                    "completeness": 0.0,
                    "topic_infos": [],
                    "topic_sizes": {},
                }
            )
            continue

        try:
            result = evaluate_goalex_clustering(
                variation, cluster_result_path, training_set
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR while evaluating {variation}: {e}")
            all_results.append(
                {
                    "name": variation,
                    "error": str(e),
                    "n_topics": 0,
                    "n_outliers": 0,
                    "mean_purity": 0.0,
                    "weighted_purity": 0.0,
                    "nmi": 0.0,
                    "homogeneity": 0.0,
                    "completeness": 0.0,
                    "topic_infos": [],
                    "topic_sizes": {},
                }
            )

    write_goalex_report(all_results, RESULTS_FILE)


if __name__ == "__main__":
    main()

