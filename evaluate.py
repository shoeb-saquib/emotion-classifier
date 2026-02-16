from datetime import datetime
import random
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
from pathlib import Path
from configuration import EMOTIONS

def format_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=EMOTIONS)
    df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)

    # Add axis labels for clarity
    df.index.name = "True"
    df.columns.name = "Predicted"

    return df.to_string()


def _report_content(true_labels, predicted_labels, descriptors):
    """Build the evaluation report string (caller can write or merge it)."""
    assert len(true_labels) == len(predicted_labels), \
        f"Label/prediction length mismatch: {len(true_labels)} vs {len(predicted_labels)}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confusion_matrix_str = format_confusion_matrix(true_labels, predicted_labels).strip()
    summary_str = classification_report(true_labels, predicted_labels, digits=3).strip()
    accuracy = round(accuracy_score(true_labels, predicted_labels) * 100, 2)
    macro_f1 = round(f1_score(true_labels, predicted_labels, average="macro") * 100, 2)

    colon_pos = len(max(descriptors.keys(), key=len)) + 1
    lines = [
        "=" * 80,
        "EMOTION CLASSIFICATION EVALUATION REPORT",
        "=" * 80,
        "",
        "Timestamp" + (colon_pos - 9) * " " + ": " + timestamp,
    ]
    for k, v in descriptors.items():
        lines.append(k + (colon_pos - len(k)) * " " + ": " + str(v))
    lines.extend([
        "",
        "Accuracy" + (colon_pos - 8) * " " + ": " + str(accuracy) + "%",
        "Macro F1" + (colon_pos - 8) * " " + ": " + str(macro_f1) + "%",
        "",
        "-" * 80,
        "CONFUSION MATRIX",
        "-" * 80,
        "",
        confusion_matrix_str,
        "",
        "-" * 80,
        "CLASSIFICATION SUMMARY",
        "-" * 80,
        "",
        summary_str,
        "=" * 80,
    ])
    return "\n".join(lines)


def get_evaluation_report_content(true_labels, predicted_labels, descriptors):
    """Return the evaluation report as a string (for merging into combined files)."""
    return _report_content(true_labels, predicted_labels, descriptors)


def generate_evaluation_report(true_labels, predicted_labels, descriptors, output_file):
    content = _report_content(true_labels, predicted_labels, descriptors)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_majority_baseline(labels, num_predictions):
    counts = Counter(labels)
    return [counts.most_common(1)[0][0]] * num_predictions

def get_random_baseline(emotions, num_predictions):
    return random.choices(emotions, k=num_predictions)


