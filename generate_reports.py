import re
import pickle
from pathlib import Path

from ecfdataset import ECFDataset
from emotion_model import EmotionModel
from evaluate import get_evaluation_report_content, generate_evaluation_report
from configuration import (
    CONTEXT_METHODS,
    CONTEXT_WINDOWS,
    DESCRIPTORS,
    EMOTION_REPRESENTATIONS,
    SELECTED_CONTEXT_METHODS,
    SELECTED_EMOTION_REPRESENTATIONS,
)

TEST_EMBEDDINGS_FILENAME = "saved_data/test_embeddings.pkl"
REPORTS_DIR = Path("reports")
SEP = "=" * 80


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
    # Split between reports only (each report has SEP inside after "EMOTION CLASSIFICATION...")
    # Boundary between reports is "\n\n" + SEP + "\n" from our join.
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
    # Use "\n\n" + SEP + "\n" so _parse_reports_file can split on the same boundary
    path.write_text(("\n\n" + SEP + "\n").join(ordered))


dataset = ECFDataset()
training_set = dataset.load_split("train")
test_set = dataset.load_split("test")

model = EmotionModel()

if Path(TEST_EMBEDDINGS_FILENAME).is_file():
    test_set["embedding"] = pickle.load(open(TEST_EMBEDDINGS_FILENAME, "rb"))
else:
    test_set["embedding"] = model.encode(list(test_set.text)).tolist()
    pickle.dump(test_set.embedding.tolist(), open(TEST_EMBEDDINGS_FILENAME, "wb"))

test_conversations = list(test_set.groupby(by="conversation_ID"))
total_conversations = len(test_conversations)
for emotion_method_id in SELECTED_EMOTION_REPRESENTATIONS:
    model.set_emotion_representation_method(emotion_method_id)
    model.fit(list(training_set.text), list(training_set.emotion))
    one_context_method_has_finished = False
    er_dir = REPORTS_DIR / f"er{emotion_method_id}"
    for context_method_id in SELECTED_CONTEXT_METHODS:
        model.set_context_method(context_method_id)
        for context_window in CONTEXT_WINDOWS:
            if context_window == 0 and one_context_method_has_finished:
                continue
            model.set_context_window(context_window)
            predictions = {}
            for count, (_, conversation) in enumerate(test_conversations, start=1):
                print(
                    f"\rConversations Processed: {count}/{total_conversations}",
                    end="",
                    flush=True,
                )
                conversation = conversation.sort_values("utterance_ID")
                for idx, row in conversation.iterrows():
                    utterances = conversation.loc[
                        conversation.utterance_ID <= row.utterance_ID
                    ]
                    predictions[idx] = model.predict(embeddings=list(utterances.embedding))
            print()
            predictions = [predictions[i] for i in test_set.index]
            descriptors = {
                DESCRIPTORS[1]: EMOTION_REPRESENTATIONS[emotion_method_id],
                DESCRIPTORS[2]: context_window,
            }
            if context_window == 0:
                path = er_dir / f"er{emotion_method_id}_cw0_reports.txt"
                generate_evaluation_report(
                    test_set.emotion, predictions, descriptors, path
                )
            else:
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









