import pickle
from pathlib import Path
from datasets import load_dataset

DATA_FILENAMES = {
    "train": "saved_data/training_data.pkl",
    "validation": "saved_data/validation_data.pkl",
    "test": "saved_data/test_data.pkl"
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
            texts, emotions = [], []
            for item in dataset[split]:
                for utterance in item["conversation"]:
                    texts.append(utterance["text"])
                    emotions.append(utterance["emotion"])
            with open(filename, "wb") as f:
                pickle.dump((texts, emotions), f)

    @staticmethod
    def load_split(split, num_utterances = None):
        with open(DATA_FILENAMES[split], "rb") as f:
            if not num_utterances: return pickle.load(f)
            texts, labels = pickle.load(f)
            if num_utterances > len(texts): num_utterances = len(texts)
            return texts[:num_utterances], labels[:num_utterances]
