"""
Export the emotion-classifier training set into the format required by
goalex's iterative_cluster.py (data.json + optional labels.json).
Run from project root so that goalex/ and src/ resolve correctly.
"""
from pathlib import Path
import argparse
import json
import sys

# Project root: .../emotion-classifier
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.ecfdataset import ECFDataset


GOAL = (
    "I would like to cluster these utterances by the emotion they express. "
    "Each cluster should have a description of the form 'expresses <emotion>' "
    "or similar."
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "goalex" / "processed_data"


def main():
    parser = argparse.ArgumentParser(
        description="Export training set to goalex iterative_cluster format."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write data.json and labels.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N utterances (default: use all)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not write labels.json (only data.json)",
    )
    args = parser.parse_args()

    dataset = ECFDataset()
    training_set = dataset.load_split("train", num_utterances=args.subsample)
    texts = training_set["text"].astype(str).tolist()
    emotions = training_set["emotion"].astype(str).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Unique emotions in sorted order for stable label indices
    unique_emotions = sorted(set(emotions))
    example_descriptions = [f"expresses {e}" for e in unique_emotions[:5]]

    data = {
        "goal": GOAL,
        "texts": texts,
        "example_descriptions": example_descriptions,
    }
    data_path = args.output_dir / "data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(texts)} texts to {data_path}")

    if not args.no_labels:
        emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}
        labels = [emotion_to_idx[e] for e in emotions]
        labels_data = {
            "class_descriptions": unique_emotions,
            "labels": labels,
        }
        labels_path = args.output_dir / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote labels for {len(unique_emotions)} classes to {labels_path}")

    # If output is under goalex/processed_data/, data_path from goalex dir is the relative part
    try:
        rel = args.output_dir.relative_to(_PROJECT_ROOT / "goalex")
        data_path_arg = str(rel)
    except ValueError:
        data_path_arg = str(args.output_dir)
    print(f"\nRun goalex (from the goalex/ directory):")
    print(f"  python src/iterative_cluster.py --data_path {data_path_arg} --exp_dir experiments/ecf_emotion ...")


if __name__ == "__main__":
    main()
