from pathlib import Path
from datasets import load_dataset
import pandas as pd

DATA_FILENAMES = {
    "train": "saved_data/training_data.csv",
    "validation": "saved_data/validation_data.csv",
    "test": "saved_data/test_data.csv"
}

class ECFDataset:
    def __init__(self):
        self.ensure_data_saved()

    def ensure_data_saved(self):
        if not all(Path(f).is_file() for f in DATA_FILENAMES.values()):
            self.retrieve_and_save()

    @staticmethod
    def retrieve_and_save():
        dataset = load_dataset("NUSTM/ECF", streaming=True)
        for split, filename in DATA_FILENAMES.items():
            data = pd.DataFrame()
            seen_conversation_ids = set()
            for item in dataset[split]:
                cid = item["conversation_ID"]
                if cid in seen_conversation_ids:
                    continue
                seen_conversation_ids.add(cid)
                conversation = pd.DataFrame(item["conversation"])
                conversation["conversation_ID"] = cid
                conversation = conversation[["conversation_ID", "utterance_ID", "speaker", "text", "emotion"]]
                data = pd.concat([data, conversation], ignore_index=True)
            data.to_csv(filename, index=False)

    @staticmethod
    def load_split(split, num_utterances = None):
        if num_utterances is None:
            return pd.read_csv(DATA_FILENAMES[split])
        return pd.read_csv(DATA_FILENAMES[split]).head(num_utterances)
