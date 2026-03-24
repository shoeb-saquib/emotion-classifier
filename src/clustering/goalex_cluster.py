"""
Unified GoalEX clustering pipeline:

1. Export training set + goal into GoalEX's `processed_data/<variation>/data.json` (+ labels.json).
2. Run GoalEX `iterative_cluster.py` to perform clustering.
3. Read the resulting `cluster_result.json` and save:
   - embeddings + cluster labels to `saved_clusters/{variation}.csv`
   - emotion proportions per cluster to `saved_clusters/{variation}_emotion_proportions.csv`

Edit the CONFIG section below to change the goal and GoalEX parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Project root: .../emotion-classifier
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.configuration import EMOTIONS
from src.data.ecfdataset import ECFDataset


# ===========================
# CONFIG
# ===========================

# Where GoalEX lives and where artifacts go.
# - **GOALEX_DATA_EXPORT_DIR** controls where `data.json`/`labels.json` are written.
# - **GOALEX_DATA_EXPERIMENTS_DIR** controls where GoalEX writes timestamped runs and `cluster_result.json`.
GOALEX_ROOT: Path = _PROJECT_ROOT / "goalex"
GOALEX_DATA_EXPORT_DIR: Path = GOALEX_ROOT / "input"
GOALEX_DATA_EXPERIMENTS_DIR: Path = GOALEX_ROOT / "output"

# Directory where emotion-classifier cluster CSVs and emotion proportions live
SAVED_CLUSTERS_DIR: Path = _PROJECT_ROOT / "saved_clusters"

GOAL: str = (
    "I would like to cluster these utterances by the emotion they express. "
    "Each cluster should have a description of the form 'expresses <emotion>' "
    "or similar."
)

# GoalEX / clustering hyperparameters (edit as needed)
# Use Purdue GenAI (or other chat model) as the proposer,
# and a local Flan-T5 model as the assigner.
PROPOSER_MODEL: str = "purdue"              # e.g. "purdue", "gpt-3.5-turbo"
ASSIGNER_NAME: str = "google/flan-t5-small"  # triggers T5Assigner in GoalEX
CLUSTER_NUM_CLUSTERS: int = len(EMOTIONS)  # usually number of emotions
PROPOSER_NUM_DESCRIPTIONS_TO_PROPOSE: int = 10
PROPOSER_NUM_DESCRIPTIONS_PER_ROUND: int = 4
PROPOSER_NUM_ROUNDS_TO_PROPOSE: Optional[int] = None  # None = derive from num_descriptions
ITERATIVE_MAX_ROUNDS: int = 1
SUBSAMPLE_FOR_GOALEX: int = 0  # 0 = use all texts in data.json (GoalEX handles --subsample)

# Final assignment: without this, only texts that get a "1" from the per-description assigner
# appear in cluster_result; the rest are unassigned. With this, every text is assigned to
# exactly one cluster via a multi-assigner pass. Use T5 template for Flan-T5 assigner.
ASSIGNER_FOR_FINAL_ASSIGNMENT_TEMPLATE: str = "templates/t5_multi_assigner_one_output.txt"


@dataclass
class GoalExPaths:
    goalex_root: Path
    data_dir: Path
    exp_base_dir: Path


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
    training_set["embedding"] = list(
        model.encode(training_set["text"].astype(str).tolist())
    )
    return training_set


def _embedding_rows_for_clusters(
    training_set: pd.DataFrame, text_to_cluster: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, embeddings) arrays for rows whose text is in a cluster."""
    rows = []
    for _, row in training_set.iterrows():
        key = _normalize_text(str(row["text"]))
        if key not in text_to_cluster:
            continue
        label = text_to_cluster[key]
        emb = np.asarray(row["embedding"], dtype=np.float64)
        rows.append((label, emb))
    if not rows:
        raise ValueError(
            "No training texts matched any cluster; check that GoalEX ran on the same data."
        )
    labels = np.array([r[0] for r in rows], dtype=np.int64)
    embeddings = np.array([r[1] for r in rows], dtype=np.float64)
    return labels, embeddings


def save_embeddings_labels_csv(
    out_path: Path,
    training_set: pd.DataFrame,
    text_to_cluster: dict,
) -> None:
    """Write CSV with columns label, emb_0, emb_1, ... for each row whose text is in a cluster."""
    labels, embeddings = _embedding_rows_for_clusters(training_set, text_to_cluster)
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


def goal_to_variation(goal: str) -> str:
    """Create a short, filesystem-safe variation name tied to the goal."""
    # Use only a short hash of the goal for uniqueness; no preview snippet.
    h = hashlib.sha1(goal.encode("utf-8")).hexdigest()[:8]
    return f"goalex_{h}"


def export_to_goalex_data(variation: str, goal: str, force_export: bool = False) -> GoalExPaths:
    """
    Export training data and labels into GoalEX processed_data/<variation>/.

    Writes:
      goalex/processed_data/<variation>/data.json
      goalex/processed_data/<variation>/labels.json
    """
    goalex_root = GOALEX_ROOT
    data_dir = GOALEX_DATA_EXPORT_DIR / variation
    data_path = data_dir / "data.json"
    labels_path = data_dir / "labels.json"

    # If GoalEX data already exists for this variation and we are not forcing a
    # re-export, reuse it.
    if not force_export and data_path.is_file() and labels_path.is_file():
        exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation
        print(f"GoalEX data already exists at {data_dir}; skipping export.")
        return GoalExPaths(
            goalex_root=goalex_root, data_dir=data_dir, exp_base_dir=exp_base_dir
        )

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    texts = training_set["text"].astype(str).tolist()
    emotions = training_set["emotion"].astype(str).tolist()

    # Use EMOTIONS order if all present, otherwise fall back to sorted unique emotions
    unique_emotions = sorted(set(emotions))
    if set(unique_emotions) == set(EMOTIONS):
        class_descriptions = EMOTIONS
    else:
        class_descriptions = unique_emotions

    example_descriptions = [f"expresses {e}" for e in class_descriptions[:5]]

    data = {
        "goal": goal,
        "texts": texts,
        "example_descriptions": example_descriptions,
    }
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    emotion_to_idx = {e: i for i, e in enumerate(class_descriptions)}
    labels = [emotion_to_idx[e] for e in emotions]
    labels_data = {
        "class_descriptions": class_descriptions,
        "labels": labels,
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)

    exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation

    print(f"Wrote GoalEX data to {data_dir}")
    return GoalExPaths(goalex_root=goalex_root, data_dir=data_dir, exp_base_dir=exp_base_dir)


def run_goalex_iterative_cluster(paths: GoalExPaths, variation: str) -> Path:
    """
    Run GoalEX iterative_cluster.py with the configured parameters.

    Returns the path to the resulting cluster_result.json.
    """
    goalex_root = paths.goalex_root
    data_dir = paths.data_dir
    exp_base = paths.exp_base_dir

    # Use data_path and exp_dir relative to goalex root if possible
    try:
        data_arg = str(data_dir.relative_to(goalex_root))
    except ValueError:
        data_arg = str(data_dir)
    try:
        exp_arg = str(exp_base.relative_to(goalex_root))
    except ValueError:
        exp_arg = str(exp_base)

    cmd = [
        sys.executable,
        str(goalex_root / "src" / "iterative_cluster.py"),
        "--data_path",
        data_arg,
        "--exp_dir",
        exp_arg,
        "--proposer_model",
        PROPOSER_MODEL,
        "--assigner_name",
        ASSIGNER_NAME,
        "--cluster_num_clusters",
        str(CLUSTER_NUM_CLUSTERS),
        "--proposer_num_descriptions_to_propose",
        str(PROPOSER_NUM_DESCRIPTIONS_TO_PROPOSE),
        "--proposer_num_descriptions_per_round",
        str(PROPOSER_NUM_DESCRIPTIONS_PER_ROUND),
        "--iterative_max_rounds",
        str(ITERATIVE_MAX_ROUNDS),
        "--turn_off_approval_before_running",
    ]
    if ASSIGNER_FOR_FINAL_ASSIGNMENT_TEMPLATE:
        cmd.extend(
            [
                "--assigner_for_final_assignment_template",
                ASSIGNER_FOR_FINAL_ASSIGNMENT_TEMPLATE,
            ]
        )
    if PROPOSER_NUM_ROUNDS_TO_PROPOSE is not None:
        cmd.extend(
            [
                "--proposer_num_rounds_to_propose",
                str(PROPOSER_NUM_ROUNDS_TO_PROPOSE),
            ]
        )
    if SUBSAMPLE_FOR_GOALEX > 0:
        cmd.extend(["--subsample", str(SUBSAMPLE_FOR_GOALEX)])

    print("\nRunning GoalEX iterative_cluster.py:")
    print("  ", " ".join(cmd))
    subprocess.run(cmd, cwd=goalex_root, check=True)

    # Find latest timestamped experiment directory under exp_base
    if not exp_base.exists():
        raise FileNotFoundError(f"Expected experiments base dir not found: {exp_base}")
    subdirs = [p for p in exp_base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No timestamped experiment dirs found under {exp_base}")
    latest = max(subdirs, key=lambda p: p.name)
    cluster_path = latest / "cluster_result.json"
    if not cluster_path.is_file():
        raise FileNotFoundError(f"cluster_result.json not found at {cluster_path}")

    print(f"GoalEX clustering finished. Results at {cluster_path}")
    return cluster_path


def find_existing_cluster_result(exp_base_dir: Path) -> Optional[Path]:
    """
    If GoalEX has already produced a cluster_result.json under exp_base_dir,
    return a path to the most recent one. Otherwise return None.

    GoalEX writes: <exp_base_dir>/<timestamp>/cluster_result.json
    """
    # Non-standard but handle direct file placement
    direct = exp_base_dir / "cluster_result.json"
    if direct.is_file():
        return direct

    if not exp_base_dir.exists():
        return None

    subdirs = [p for p in exp_base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    # Timestamp dirs are lexicographically sortable (YYYY-MM-DD-HH-MM-SS)
    for d in sorted(subdirs, key=lambda p: p.name, reverse=True):
        candidate = d / "cluster_result.json"
        if candidate.is_file():
            return candidate

    return None


def save_to_saved_clusters(cluster_result_path: Path, variation: str) -> None:
    """
    Convert GoalEX cluster_result.json into saved_clusters format:
      - embeddings + labels CSV
      - emotion proportions per cluster
    """
    cluster_result = load_cluster_result(cluster_result_path)
    text_to_cluster = text_to_cluster_id(cluster_result)

    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    training_set = ensure_embeddings(training_set)

    out_dir = SAVED_CLUSTERS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{variation}.csv"
    prop_path = out_dir / f"{variation}_emotion_proportions.csv"

    save_embeddings_labels_csv(csv_path, training_set, text_to_cluster)
    save_emotion_proportions_csv(prop_path, training_set, text_to_cluster)

    print(
        f"\nDone. Saved embeddings+labels to {csv_path} and emotion proportions to {prop_path}."
    )
    print(
        f"To use these clusters, add {variation!r} to SELECTED_CLUSTER_VARIATIONS in src/configuration.py."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GoalEX clustering end-to-end and save results for the emotion classifier.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=GOAL,
        help="Goal description for GoalEX (default: configured GOAL in this file).",
    )
    parser.add_argument(
        "--variation",
        type=str,
        default=None,
        help="Optional explicit variation name. If not set, derived from goal text.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing exported data and previous GoalEX runs; always re-export and re-cluster.",
    )
    args = parser.parse_args()

    goal = args.goal
    variation = args.variation or goal_to_variation(goal)

    print(f"Using variation name: {variation}")

    # If we already have a GoalEX result for this variation and we are not
    # forcing a rerun, reuse it.
    exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation
    existing = find_existing_cluster_result(exp_base_dir)
    if existing is not None and not args.force:
        print(
            f"Found existing GoalEX cluster result at {existing}; skipping export and clustering."
        )
        save_to_saved_clusters(existing, variation)
        return

    paths = export_to_goalex_data(variation, goal, force_export=args.force)
    cluster_result_path = run_goalex_iterative_cluster(paths, variation)
    save_to_saved_clusters(cluster_result_path, variation)


if __name__ == "__main__":
    main()

