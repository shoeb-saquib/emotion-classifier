from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def print_confusion_matrix(true_labels, predicted_labels):
    emotions = sorted(set(true_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=emotions)
    df = pd.DataFrame(cm, index=emotions, columns=emotions)
    print(df)

def print_report(true_labels, predicted_labels, method):
    print(f"\n----------------------------BEGIN EVALUATION----------------------------\n")
    print(f"Method: '{method}'\n")
    print("Confusion Matrix:")
    print_confusion_matrix(true_labels, predicted_labels)
    print("\nSummary:")
    print(classification_report(true_labels, predicted_labels, digits=3))
    print("-----------------------------END EVALUATION----------------------------\n")


def print_majority_baseline(labels):
    counts = Counter(labels)
    for emotion in counts:
        counts[emotion] = counts[emotion] / len(labels)
    print("\nMethod: always predicting the majority emotion")
    print(counts)
    accuracy = max(counts.values())
    print(f"Accuracy: {accuracy * 100:.2f}%")

