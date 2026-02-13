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


def generate_evaluation_report(true_labels, predicted_labels, descriptors, output_file):
    assert len(true_labels) == len(predicted_labels), \
        f"Label/prediction length mismatch: {len(true_labels)} vs {len(predicted_labels)}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    confusion_matrix_str = format_confusion_matrix(true_labels, predicted_labels)
    summary_str = classification_report(true_labels, predicted_labels, digits=3)
    accuracy = round(accuracy_score(true_labels, predicted_labels) * 100, 2)
    macro_f1 = round(f1_score(true_labels, predicted_labels, average="macro") * 100, 2)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("EMOTION CLASSIFICATION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        colon_pos = len(max(descriptors.keys(), key=len)) + 1
        f.write("Timestamp" + (colon_pos - 9) * ' ' + ': ' + timestamp + '\n')
        for k, v in descriptors.items():
            f.write(k + (colon_pos - len(k)) * ' ' + ': ' + str(v) + '\n')
        f.write('\n')

        f.write("Accuracy" + (colon_pos - 8) * ' ' + ': ' + str(accuracy) + "%\n")
        f.write("Macro F1" + (colon_pos - 8) * ' ' + ': ' + str(macro_f1) + "%\n\n")

        f.write("-" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n\n")
        f.write(confusion_matrix_str + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        f.write(summary_str + "\n")

        f.write("=" * 80 + "\n")


def get_majority_baseline(labels, num_predictions):
    counts = Counter(labels)
    return [counts.most_common(1)[0][0]] * num_predictions

def get_random_baseline(emotions, num_predictions):
    return random.choices(emotions, k=num_predictions)


