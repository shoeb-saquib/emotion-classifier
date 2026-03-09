"""
Generate evaluation reports for baseline methods (majority and random prediction).

Writes reports to reports/baselines/ using the same format as the main evaluation reports.
"""

import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir
while _project_root != _project_root.parent and (_project_root / "src").is_dir() is False:
    _project_root = _project_root.parent
if (_project_root / "src").is_dir() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.ecfdataset import ECFDataset
from src.evaluation.evaluate import generate_evaluation_report, get_majority_baseline, get_random_baseline
from src.configuration import BASELINES, DESCRIPTORS, EMOTIONS

REPORTS_ROOT = _project_root / "reports"
BASELINES_DIR = REPORTS_ROOT / "baselines"


def main():
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    test_set = dataset.load_split("test")

    true_labels = test_set["emotion"].tolist()
    n = len(true_labels)

    # Majority baseline: predict most frequent training emotion for every test sample
    training_emotions = training_set["emotion"].tolist()
    majority_predictions = get_majority_baseline(training_emotions, n)
    generate_evaluation_report(
        true_labels,
        majority_predictions,
        descriptors={DESCRIPTORS[0]: BASELINES[1]},  # "predict majority emotion"
        output_file=BASELINES_DIR / "majority_baseline.txt",
    )
    print(f"Wrote {BASELINES_DIR / 'majority_baseline.txt'}")

    # Random baseline: random choice from EMOTIONS for each test sample
    random_predictions = get_random_baseline(EMOTIONS, n)
    generate_evaluation_report(
        true_labels,
        random_predictions,
        descriptors={DESCRIPTORS[0]: BASELINES[0]},  # "randomly predict emotions"
        output_file=BASELINES_DIR / "random_baseline.txt",
    )
    print(f"Wrote {BASELINES_DIR / 'random_baseline.txt'}")


if __name__ == "__main__":
    main()
