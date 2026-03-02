"""
Run BERTopic clustering variations, evaluate them, and write a comparison report.

Orchestrates: cluster_variation (fit one model), cluster_metrics (evaluate),
cluster_report (write text file). Cache is used to skip variations already run.
"""

import sys
from pathlib import Path

# Find project root: walk up until we find a directory containing "src"
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir
while _project_root != _project_root.parent and (_project_root / "src").is_dir() is False:
    _project_root = _project_root.parent
if (_project_root / "src").is_dir():
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
else:
    _project_root = _script_dir.parent.parent.parent  # fallback

import pickle

from src.data.ecfdataset import ECFDataset
from src.clustering.bertopic.bertopic_config import BERTopicConfig, BERTOPIC_VARIATIONS
from src.clustering.bertopic.bertopic_cluster import fit_cluster
from src.clustering.cluster_metrics import evaluate_clustering
from src.clustering.bertopic.bertopic_cluster_report import write_report

# Paths (from repo root)
PROJECT_ROOT = _project_root
RESULTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "bertopic_variation_report.txt"
SAVED_DATA_DIR = PROJECT_ROOT / "saved_data"
SAVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = SAVED_DATA_DIR / "bertopic_variation_report.pkl"


def _is_invalid_cached_result(r: dict) -> bool:
    """Treat failed or bogus cached entries as missing so they get re-run."""
    if "error" in r:
        return True
    if r.get("n_topics", 0) == 0 and r.get("n_outliers", 0) == 0:
        return True
    return False


def main():
    # Load cache and drop invalid entries
    all_results = []
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                loaded = pickle.load(f)
            all_results = [r for r in loaded if not _is_invalid_cached_result(r)]
            dropped = len(loaded) - len(all_results)
            if dropped:
                print(f"Dropped {dropped} invalid cached result(s), will re-run those variations.")
            print(f"Loaded {len(all_results)} existing result(s) from {CACHE_FILE.name}")
        except Exception as e:
            print(f"Could not load cache ({e}), starting fresh.")

    existing_names = {r["name"] for r in all_results}
    to_run = [c for c in BERTOPIC_VARIATIONS if c.name() not in existing_names]
    if not to_run:
        print("All variations already in report. Nothing to run.")
        return

    print(f"Running {len(to_run)} new variation(s), skipping {len(BERTOPIC_VARIATIONS) - len(to_run)} already in report.")
    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    docs = list(training_set["text"])

    for config in to_run:
        name = config.name()
        print(f"\n>>> Running: {name}")
        try:
            topics, fitted_model = fit_cluster(config, docs, training_set=training_set)
            result = evaluate_clustering(name, topics, fitted_model, training_set)
            all_results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
            all_results.append({
                "name": name,
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
            })

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved {len(all_results)} result(s) to {CACHE_FILE.name}")

    write_report(all_results, RESULTS_FILE)


if __name__ == "__main__":
    main()
