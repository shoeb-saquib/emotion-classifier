import re
import sys
import pickle
from pathlib import Path

# Ensure project root is on path when run as script (e.g. python src/evaluation/generate_reports.py)
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir
while _project_root != _project_root.parent and (_project_root / "src").is_dir() is False:
    _project_root = _project_root.parent
if (_project_root / "src").is_dir() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.ecfdataset import ECFDataset
from src.models.emotion_model import EmotionModel
from src.evaluation.evaluate import get_evaluation_report_content, generate_evaluation_report
from src.configuration import *

# Paths relative to project root so they work regardless of cwd
TEST_EMBEDDINGS_FILENAME = _project_root / "saved_data" / "test_embeddings.pkl"
TRAIN_EMBEDDINGS_FILENAME = _project_root / "saved_data" / "train_embeddings.pkl"
REPORTS_ROOT = _project_root / "reports"
SEP = "=" * 80

# One folder per cluster config: reports/<cluster_subdir>/er0/, er1/, er2/
# e.g. reports/cv_k_7_knn5/er0/... or reports/no_cluster/er0/... when no variations
_cluster_subdir = (
    f"cv_{'_'.join(SELECTED_CLUSTER_VARIATIONS)}_knn{KNN_NEIGHBORS}"
    if SELECTED_CLUSTER_VARIATIONS
    else "no_cluster"
)
REPORTS_DIR = REPORTS_ROOT / _cluster_subdir


def _parse_context_window_from_report(block: str) -> int | None:
    """Extract Context Window value from a report block; return None if not found."""
    m = re.search(r"Context Window\s+:\s*(\d+)", block)
    return int(m.group(1)) if m else None


def _parse_reports_file(path: Path) -> dict[int, str]:
    """Read a combined reports file and return {context_window: report_content}."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text()
    # Split between reports: each report starts with SEP + "\n", so boundary is "\n\n" + SEP + "\n"
    blocks = text.split("\n\n" + SEP + "\n")
    for block in blocks:
        block = block.strip()
        if not block or "EMOTION CLASSIFICATION" not in block:
            continue
        cw = _parse_context_window_from_report(block)
        if cw is not None:
            out[cw] = block if block.startswith(SEP) else (SEP + "\n" + block)
    return out


def _write_combined_reports(path: Path, reports_by_cw: dict[int, str]) -> None:
    """Write reports to path in ascending order of context window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [reports_by_cw[cw] for cw in sorted(reports_by_cw)]
    # Join with "\n\n" only; strip each block so trailing newlines don't create extra gaps
    path.write_text("\n\n".join(s.rstrip() for s in ordered))

def _load_embeddings(filename: Path, data, model):
    """Ensure data has an 'embedding' column: use existing, load from file, or create and save."""
    if "embedding" in data.columns:
        return
    if filename.is_file():
        with open(filename, "rb") as f:
            data["embedding"] = pickle.load(f)
    else:
        data["embedding"] = model.encode(list(data["text"])).tolist()
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data["embedding"], f)

def main():
    dataset = ECFDataset()
    training_set = dataset.load_split("train")
    test_set = dataset.load_split("test")

    model = EmotionModel()
    for variation in SELECTED_CLUSTER_VARIATIONS:
        model.add_cluster_classifier(variation, KNN_NEIGHBORS)

    _load_embeddings(TRAIN_EMBEDDINGS_FILENAME, training_set, model)
    _load_embeddings(TEST_EMBEDDINGS_FILENAME, test_set, model)

    test_conversations = list(test_set.groupby(by="conversation_ID"))
    total_conversations = len(test_conversations)
    for emotion_method_id in SELECTED_EMOTION_REPRESENTATIONS:

        model.set_emotion_representation_method(emotion_method_id)
        # Methods 0 and 1 don't use context in fit; method 2 does, so we refit it per (context_method, context_window)
        if emotion_method_id in (0, 1):
            model.fit(training_set)
        one_context_method_has_finished = False
        er_dir = REPORTS_DIR / f"er{emotion_method_id}"

        # Context-window 0 doesn't use any context method, so allow running with no context methods selected.
        context_method_ids = SELECTED_CONTEXT_METHODS or [None]
        for context_method_id in context_method_ids:
            if context_method_id is not None:
                model.set_context_method(context_method_id)
            for context_window in CONTEXT_WINDOWS:
                if context_window == 0 and one_context_method_has_finished:
                    continue
                model.set_context_window(context_window)
                if emotion_method_id == 2:
                    model.fit(training_set)

                predictions = {}
                for count, (_, conversation) in enumerate(test_conversations, start=1):
                    print(
                        f"\rConversations Processed: {count}/{total_conversations}",
                        end="",
                        flush=True,
                    )
                    conversation = conversation.sort_values("utterance_ID")
                    for idx, row in conversation.iterrows():
                        utterances = conversation.loc[conversation.utterance_ID <= row.utterance_ID]
                        predictions[idx] = model.predict(embeddings=list(utterances.embedding))
                print()

                predictions = [predictions[i] for i in test_set.index]
                descriptors = {
                    DESCRIPTORS[1]: EMOTION_REPRESENTATIONS[emotion_method_id],
                    DESCRIPTORS[2]: context_window,
                    "Cluster Variations": ", ".join(SELECTED_CLUSTER_VARIATIONS),
                    "KNN Neighbors": KNN_NEIGHBORS,
                }
                if context_window == 0:
                    path = er_dir / f"er{emotion_method_id}_cw0_reports.txt"
                    generate_evaluation_report(
                        test_set.emotion, predictions, descriptors, path
                    )
                else:
                    if context_method_id is None:
                        raise ValueError(
                            "Non-zero context windows require at least one selected context method."
                        )
                    descriptors[DESCRIPTORS[3]] = CONTEXT_METHODS[context_method_id]
                    path = er_dir / f"er{emotion_method_id}_cm{context_method_id}_reports.txt"
                    content = get_evaluation_report_content(
                        test_set.emotion, predictions, descriptors
                    )
                    reports_by_cw = _parse_reports_file(path)
                    reports_by_cw[context_window] = content
                    _write_combined_reports(path, reports_by_cw)
                print(f"Report with context window {context_window} has been generated in {path}")
                one_context_method_has_finished = True

if __name__ == "__main__":
    main()









