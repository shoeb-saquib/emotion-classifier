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
"""
    "I would like to cluster these utterances by the emotion they express. "
    "Each cluster should have a description of the form 'expresses <emotion>' "
    "or similar."
"""
"""
    "I would like to cluster these utterances by social intent (requesting help, setting boundaries, conflict, reconciliation, disclosure, performance, etc.)"
"""

# GoalEX / clustering hyperparameters (edit as needed)
# Use Purdue GenAI (or other chat model) as the proposer,
# and a local Flan-T5 model as the assigner.
PROPOSER_MODEL: str = "purdue"              # e.g. "purdue", "gpt-3.5-turbo"
ASSIGNER_NAME: str = "google/flan-t5-large" #google/flan-t5-small"  # triggers T5Assigner in GoalEX
CLUSTER_NUM_CLUSTERS: int = 7
PROPOSER_NUM_DESCRIPTIONS_TO_PROPOSE: int = 15
PROPOSER_NUM_DESCRIPTIONS_PER_ROUND: int = 7
PROPOSER_NUM_ROUNDS_TO_PROPOSE: Optional[int] = None  # None = derive from num_descriptions
ITERATIVE_MAX_ROUNDS: int = 5
SUBSAMPLE_FOR_GOALEX: int = 500  # 0 = use all texts in data.json (GoalEX handles --subsample)
GOALEX_CONTEXT_UTTERANCES: int = 3  # number of previous utterances (with speakers) to include as context

# Final assignment: without this, only texts that get a "1" from the per-description assigner
# appear in cluster_result; the rest are unassigned. With this, every text is assigned to
# exactly one cluster via a multi-assigner pass. Use T5 template for Flan-T5 assigner.
ASSIGNER_FOR_FINAL_ASSIGNMENT_TEMPLATE: str = "templates/t5_multi_assigner_one_output.txt"

# Per-emotion clustering mode: when True, groups training data by emotion label and
# runs a separate GoalEX clustering job for each emotion.  When False, the original
# single-run pipeline is used.
GOALEX_PER_EMOTION: bool = True
CLUSTERS_PER_EMOTION: int = 5
SAMPLES_PER_EMOTION: int = 100
PER_EMOTION_CONTEXT_UTTERANCES: int = 2
PER_EMOTION_MAX_CLUSTER_FRACTION: float = 1.0

PER_EMOTION_GOAL_TEMPLATE: str = (
    "All of the following utterances express the emotion '{emotion}'. "
    "I want to find {num_clusters} distinct sub-categories within this emotion "
    "that capture different ways '{emotion}' manifests in conversation. "
    "Each cluster should represent a meaningfully different variation, tone, "
    "or trigger of '{emotion}' that would help a downstream classifier "
    "distinguish between similar utterances."
)

PER_EMOTION_EXAMPLE_DESCRIPTIONS: dict[str, list[str]] = {
    "anger": [
        "expresses frustrated anger due to repeated problems",
        "expresses indignant outrage at injustice or betrayal",
        "expresses irritated or sarcastic anger",
    ],
    "disgust": [
        "expresses moral disgust at someone's behavior",
        "expresses physical revulsion or repulsion",
        "expresses contemptuous dismissal",
    ],
    "fear": [
        "expresses anxious worry about future events",
        "expresses acute panic or terror",
        "expresses apprehensive uncertainty or dread",
    ],
    "joy": [
        "expresses excited elation or enthusiasm",
        "expresses warm contentment or gratitude",
        "expresses relieved happiness after tension",
    ],
    "neutral": [
        "conveys matter-of-fact information without emotion",
        "expresses polite acknowledgment or agreement",
        "asks a neutral question or gives a routine response",
    ],
    "sadness": [
        "expresses grief or mourning over a loss",
        "expresses disappointed sadness about an outcome",
        "expresses lonely or melancholic sadness",
    ],
    "surprise": [
        "expresses shocked disbelief at unexpected news",
        "expresses pleasant surprise or amazement",
        "expresses confused astonishment or bewilderment",
    ],
}


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
    training_set: pd.DataFrame,
    text_to_cluster: dict,
    cluster_to_source_emotion: Optional[dict[int, str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Return (labels, embeddings, conversation_ID, utterance_ID, source_emotion) for rows whose text is in a cluster.

    Duplicate normalized texts use the first matching row's ids (iteration order).
    """
    if "conversation_ID" not in training_set.columns or "utterance_ID" not in training_set.columns:
        raise ValueError(
            "training_set must have conversation_ID and utterance_ID columns for cluster export."
        )
    rows = []
    for _, row in training_set.iterrows():
        key = _normalize_text(str(row["text"]))
        if key not in text_to_cluster:
            continue
        label = text_to_cluster[key]
        emb = np.asarray(row["embedding"], dtype=np.float64)
        cid = int(row["conversation_ID"])
        uid = int(row["utterance_ID"])
        src_emotion = (
            str(cluster_to_source_emotion.get(label, ""))
            if cluster_to_source_emotion is not None
            else str(row.get("emotion", ""))
        )
        rows.append((label, emb, cid, uid, src_emotion))
    if not rows:
        raise ValueError(
            "No training texts matched any cluster; check that GoalEX ran on the same data."
        )
    labels = np.array([r[0] for r in rows], dtype=np.int64)
    embeddings = np.array([r[1] for r in rows], dtype=np.float64)
    conv_ids = np.array([r[2] for r in rows], dtype=np.int64)
    utt_ids = np.array([r[3] for r in rows], dtype=np.int64)
    source_emotions = [str(r[4]) for r in rows]
    return labels, embeddings, conv_ids, utt_ids, source_emotions


def save_embeddings_labels_csv(
    out_path: Path,
    training_set: pd.DataFrame,
    text_to_cluster: dict,
    cluster_to_source_emotion: Optional[dict[int, str]] = None,
) -> None:
    """Write CSV with label, conversation_ID, utterance_ID, emb_0, ... for clustered rows."""
    labels, embeddings, conv_ids, utt_ids, source_emotions = _embedding_rows_for_clusters(
        training_set, text_to_cluster, cluster_to_source_emotion
    )
    n, dim = embeddings.shape
    data = {
        "label": labels,
        "conversation_ID": conv_ids,
        "utterance_ID": utt_ids,
        "source_emotion": source_emotions,
    }
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


def _build_context_column(
    training_set: pd.DataFrame,
    context_utterances: int,
    include_emotion_labels: bool = False,
) -> list[str]:
    """
    Build one context string per row from previous utterances in the same conversation.
    Context lines are speaker-tagged and ordered oldest -> newest.

    When *include_emotion_labels* is True each line is formatted as
    ``Speaker [emotion]: text`` so the LLM can see the emotional trajectory.
    """
    if context_utterances <= 0:
        return [""] * len(training_set)
    df = training_set.copy()
    df["_orig_idx"] = np.arange(len(df))
    parts = []
    for _, conv in df.groupby("conversation_ID", sort=False):
        conv = conv.sort_values("utterance_ID")
        lines = []
        for _, row in conv.iterrows():
            speaker = str(row.get("speaker", "") or "Unknown")
            text = " ".join(str(row["text"]).split()[:24])
            if include_emotion_labels:
                emotion = str(row.get("emotion", "") or "unknown")
                lines.append(f"{speaker} [{emotion}]: {text}")
            else:
                lines.append(f"{speaker}: {text}")
        contexts = []
        for i in range(len(lines)):
            start = max(0, i - context_utterances)
            ctx = "\n".join(lines[start:i])
            contexts.append(ctx)
        conv = conv.assign(_context=contexts)
        parts.append(conv[["_orig_idx", "_context"]])
    merged = pd.concat(parts, ignore_index=True).sort_values("_orig_idx")
    return merged["_context"].fillna("").astype(str).tolist()


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
    contexts = _build_context_column(training_set, GOALEX_CONTEXT_UTTERANCES)
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
        "contexts": contexts,
        "context_utterances": GOALEX_CONTEXT_UTTERANCES,
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


def export_to_goalex_data_per_emotion(
    variation: str,
    force_export: bool = False,
) -> list[tuple[str, GoalExPaths]]:
    """
    Export per-emotion training data into GoalEX input directories.

    Contexts are built from the *full* training set (before grouping) so that
    cross-emotion conversation history is preserved.  Each context line carries
    the speaker and the emotion label of that context utterance.

    Returns a list of ``(emotion, GoalExPaths)`` tuples, one per emotion.
    """
    dataset = ECFDataset()
    training_set = dataset.load_split("train")

    contexts = _build_context_column(
        training_set,
        PER_EMOTION_CONTEXT_UTTERANCES,
        include_emotion_labels=True,
    )
    training_set = training_set.copy()
    training_set["_context"] = contexts

    result: list[tuple[str, GoalExPaths]] = []
    for emotion in EMOTIONS:
        subset = training_set[training_set["emotion"] == emotion]
        if subset.empty:
            print(f"  No data for emotion '{emotion}'; skipping.")
            continue

        data_dir = GOALEX_DATA_EXPORT_DIR / variation / emotion
        data_path = data_dir / "data.json"

        if not force_export and data_path.is_file():
            print(f"  GoalEX data for '{emotion}' already exists at {data_dir}; skipping export.")
        else:
            data_dir.mkdir(parents=True, exist_ok=True)

            goal = PER_EMOTION_GOAL_TEMPLATE.format(
                emotion=emotion,
                num_clusters=CLUSTERS_PER_EMOTION,
            )
            texts = subset["text"].astype(str).tolist()
            ctx = subset["_context"].tolist()
            example_descs = PER_EMOTION_EXAMPLE_DESCRIPTIONS.get(
                emotion,
                [f"expresses a distinct sub-type of {emotion}"],
            )

            data = {
                "goal": goal,
                "texts": texts,
                "contexts": ctx,
                "context_utterances": PER_EMOTION_CONTEXT_UTTERANCES,
                "example_descriptions": example_descs,
            }
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Wrote {len(texts)} texts for '{emotion}' to {data_dir}")

        exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation / emotion
        result.append((
            emotion,
            GoalExPaths(goalex_root=GOALEX_ROOT, data_dir=data_dir, exp_base_dir=exp_base_dir),
        ))

    return result


def run_goalex_iterative_cluster(
    paths: GoalExPaths,
    variation: str,
    *,
    cluster_num_clusters: Optional[int] = None,
    subsample: Optional[int] = None,
    context_utterances: Optional[int] = None,
    max_cluster_fraction: Optional[float] = None,
) -> Path:
    """
    Run GoalEX iterative_cluster.py with the configured parameters.

    Optional keyword arguments override the corresponding module-level
    constants for this invocation only (used by the per-emotion pipeline).

    Returns the path to the resulting cluster_result.json.
    """
    _cluster_num = cluster_num_clusters if cluster_num_clusters is not None else CLUSTER_NUM_CLUSTERS
    _subsample = subsample if subsample is not None else SUBSAMPLE_FOR_GOALEX
    _ctx = context_utterances if context_utterances is not None else GOALEX_CONTEXT_UTTERANCES
    _max_cf = max_cluster_fraction

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
        "--proposer_num_descriptions_to_propose",
        str(PROPOSER_NUM_DESCRIPTIONS_TO_PROPOSE),
        "--proposer_num_descriptions_per_round",
        str(PROPOSER_NUM_DESCRIPTIONS_PER_ROUND),
        "--iterative_max_rounds",
        str(ITERATIVE_MAX_ROUNDS),
        "--context_utterances",
        str(_ctx),
        "--turn_off_approval_before_running",
    ]
    if _cluster_num is not None:
        cmd.extend(["--cluster_num_clusters", str(_cluster_num)])
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
    if _subsample > 0:
        cmd.extend(["--subsample", str(_subsample)])
    if _max_cf is not None:
        cmd.extend(["--max_cluster_fraction", str(_max_cf)])

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


def run_goalex_per_emotion(
    emotion_paths: list[tuple[str, GoalExPaths]],
    variation: str,
    force: bool = False,
) -> list[tuple[str, Path]]:
    """
    Run GoalEX for each emotion group, reusing existing results when possible.

    Returns a list of ``(emotion, cluster_result_path)`` tuples.
    """
    results: list[tuple[str, Path]] = []
    for emotion, paths in emotion_paths:
        print(f"\n{'='*60}")
        print(f"  GoalEX clustering for emotion: {emotion}")
        print(f"{'='*60}")

        existing = find_existing_cluster_result(paths.exp_base_dir)
        if existing is not None and not force:
            print(f"  Reusing existing result at {existing}")
            results.append((emotion, existing))
            continue

        cluster_path = run_goalex_iterative_cluster(
            paths,
            variation,
            cluster_num_clusters=CLUSTERS_PER_EMOTION,
            subsample=SAMPLES_PER_EMOTION,
            context_utterances=PER_EMOTION_CONTEXT_UTTERANCES,
            max_cluster_fraction=PER_EMOTION_MAX_CLUSTER_FRACTION,
        )
        results.append((emotion, cluster_path))

    return results


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


def merge_per_emotion_results(
    emotion_results: list[tuple[str, Path]],
    variation: str,
) -> None:
    """
    Combine per-emotion cluster_result.json files into a single set of
    saved_clusters CSVs with globally unique (non-overlapping) cluster IDs.
    """
    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    training_set = ensure_embeddings(training_set)

    combined_text_to_cluster: dict[str, int] = {}
    cluster_to_source_emotion: dict[int, str] = {}
    cluster_offset = 0

    for emotion, result_path in emotion_results:
        cluster_result = load_cluster_result(result_path)
        if not cluster_result:
            print(f"  Warning: empty cluster result for '{emotion}'; skipping.")
            continue
        num_clusters_this_emotion = len(cluster_result)
        for local_id, (description, texts) in enumerate(cluster_result.items()):
            global_id = cluster_offset + local_id
            cluster_to_source_emotion[global_id] = emotion
            for t in texts:
                key = _normalize_text(t)
                if key not in combined_text_to_cluster:
                    combined_text_to_cluster[key] = global_id
        print(
            f"  {emotion}: {num_clusters_this_emotion} clusters "
            f"(IDs {cluster_offset}..{cluster_offset + num_clusters_this_emotion - 1})"
        )
        cluster_offset += num_clusters_this_emotion

    if not combined_text_to_cluster:
        raise ValueError("No texts were assigned to any cluster across all emotions.")

    out_dir = SAVED_CLUSTERS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{variation}.csv"
    prop_path = out_dir / f"{variation}_emotion_proportions.csv"

    save_embeddings_labels_csv(
        csv_path,
        training_set,
        combined_text_to_cluster,
        cluster_to_source_emotion=cluster_to_source_emotion,
    )
    save_emotion_proportions_csv(prop_path, training_set, combined_text_to_cluster)

    print(
        f"\nDone. Merged {cluster_offset} clusters across {len(emotion_results)} emotions."
    )
    print(
        f"Saved embeddings+labels to {csv_path} and emotion proportions to {prop_path}."
    )
    print(
        f"To use these clusters, add {variation!r} to SELECTED_CLUSTER_VARIATIONS "
        f"in src/configuration.py."
    )


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


def _gather_all_emotion_results(variation: str) -> list[tuple[str, Path]]:
    """
    Collect the latest existing cluster_result.json for every emotion in *variation*.
    Returns ``(emotion, path)`` tuples for emotions that have a result; skips the rest.
    """
    results: list[tuple[str, Path]] = []
    for emotion in EMOTIONS:
        exp_base = GOALEX_DATA_EXPERIMENTS_DIR / variation / emotion
        existing = find_existing_cluster_result(exp_base)
        if existing is not None:
            results.append((emotion, existing))
    return results


def _main_per_emotion(variation: str, force: bool) -> None:
    """Per-emotion clustering pipeline (all emotions)."""
    print(f"Per-emotion mode: clustering each emotion independently")
    print(f"  clusters_per_emotion={CLUSTERS_PER_EMOTION}, "
          f"samples_per_emotion={SAMPLES_PER_EMOTION}")

    emotion_paths = export_to_goalex_data_per_emotion(variation, force_export=force)
    emotion_results = run_goalex_per_emotion(emotion_paths, variation, force=force)
    merge_per_emotion_results(emotion_results, variation)


def _main_single_emotion(
    variation: str,
    emotion: str,
    samples: int,
    clusters: int,
) -> None:
    """Re-run GoalEX for one emotion, then re-merge all emotions."""
    print(f"Single-emotion update: re-clustering '{emotion}'")
    print(f"  clusters={clusters}, samples={samples}")

    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    contexts = _build_context_column(
        training_set,
        PER_EMOTION_CONTEXT_UTTERANCES,
        include_emotion_labels=True,
    )
    training_set = training_set.copy()
    training_set["_context"] = contexts

    subset = training_set[training_set["emotion"] == emotion]
    if subset.empty:
        raise ValueError(f"No training data for emotion '{emotion}'.")

    data_dir = GOALEX_DATA_EXPORT_DIR / variation / emotion
    data_dir.mkdir(parents=True, exist_ok=True)

    goal = PER_EMOTION_GOAL_TEMPLATE.format(emotion=emotion, num_clusters=clusters)
    texts = subset["text"].astype(str).tolist()
    ctx = subset["_context"].tolist()
    example_descs = PER_EMOTION_EXAMPLE_DESCRIPTIONS.get(
        emotion, [f"expresses a distinct sub-type of {emotion}"],
    )
    data = {
        "goal": goal,
        "texts": texts,
        "contexts": ctx,
        "context_utterances": PER_EMOTION_CONTEXT_UTTERANCES,
        "example_descriptions": example_descs,
    }
    with open(data_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(texts)} texts for '{emotion}' to {data_dir}")

    exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation / emotion
    paths = GoalExPaths(goalex_root=GOALEX_ROOT, data_dir=data_dir, exp_base_dir=exp_base_dir)
    new_result = run_goalex_iterative_cluster(
        paths,
        variation,
        cluster_num_clusters=clusters,
        subsample=samples,
        context_utterances=PER_EMOTION_CONTEXT_UTTERANCES,
        max_cluster_fraction=PER_EMOTION_MAX_CLUSTER_FRACTION,
    )

    all_results = _gather_all_emotion_results(variation)
    # Replace the entry for this emotion with the freshly produced result
    all_results = [
        (e, new_result if e == emotion else p) for e, p in all_results
    ]
    # If this emotion wasn't in existing results at all, append it
    if not any(e == emotion for e, _ in all_results):
        all_results.append((emotion, new_result))
    # Keep EMOTIONS order for deterministic cluster-ID assignment
    order = {e: i for i, e in enumerate(EMOTIONS)}
    all_results.sort(key=lambda t: order.get(t[0], 999))

    merge_per_emotion_results(all_results, variation)


def _main_single_run(variation: str, goal: str, force: bool) -> None:
    """Original single-run clustering pipeline."""
    exp_base_dir = GOALEX_DATA_EXPERIMENTS_DIR / variation
    existing = find_existing_cluster_result(exp_base_dir)
    if existing is not None and not force:
        print(
            f"Found existing GoalEX cluster result at {existing}; skipping export and clustering."
        )
        save_to_saved_clusters(existing, variation)
        return

    paths = export_to_goalex_data(variation, goal, force_export=force)
    cluster_result_path = run_goalex_iterative_cluster(paths, variation)
    save_to_saved_clusters(cluster_result_path, variation)


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
        help="Optional explicit variation name. If not set, derived from goal/mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing exported data and previous GoalEX runs; always re-export and re-cluster.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        choices=EMOTIONS,
        help="Re-cluster a single emotion and re-merge (per-emotion mode only).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override SAMPLES_PER_EMOTION for this run (use with --emotion).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Override CLUSTERS_PER_EMOTION for this run (use with --emotion).",
    )
    args = parser.parse_args()

    if args.emotion:
        if not GOALEX_PER_EMOTION:
            parser.error("--emotion requires GOALEX_PER_EMOTION=True in the config.")
        variation = args.variation or f"goalex_per_emotion_c{CLUSTERS_PER_EMOTION}"
        print(f"Using variation name: {variation}")
        _main_single_emotion(
            variation,
            emotion=args.emotion,
            samples=args.samples or SAMPLES_PER_EMOTION,
            clusters=args.clusters or CLUSTERS_PER_EMOTION,
        )
    elif GOALEX_PER_EMOTION:
        variation = args.variation or f"goalex_per_emotion_c{CLUSTERS_PER_EMOTION}"
        print(f"Using variation name: {variation}")
        _main_per_emotion(variation, args.force)
    else:
        goal = args.goal
        variation = args.variation or goal_to_variation(goal)
        print(f"Using variation name: {variation}")
        _main_single_run(variation, goal, args.force)


if __name__ == "__main__":
    main()

