from pathlib import Path
import pickle

from datasets import load_dataset
import pandas as pd

# Paths relative to project root so they work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SAVED_DATA_DIR = _PROJECT_ROOT / "saved_data"
DATA_FILENAMES = {
    "train": str(_SAVED_DATA_DIR / "training_data.csv"),
    "validation": str(_SAVED_DATA_DIR / "validation_data.csv"),
    "test": str(_SAVED_DATA_DIR / "test_data.csv"),
}
EMBEDDING_FILENAMES = {
    "train": _SAVED_DATA_DIR / "train_embeddings.pkl",
    "validation": _SAVED_DATA_DIR / "validation_embeddings.pkl",
    "test": _SAVED_DATA_DIR / "test_embeddings.pkl",
}

class ECFDataset:
    def __init__(self):
        self.ensure_data_saved()

    def ensure_data_saved(self):
        if not all(Path(f).is_file() for f in DATA_FILENAMES.values()):
            self.retrieve_and_save()

    @staticmethod
    def retrieve_and_save():
        _SAVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    def load_split(split, num_utterances=None):
        df = pd.read_csv(DATA_FILENAMES[split])
        if num_utterances is not None:
            df = df.head(num_utterances)
        emb_path = EMBEDDING_FILENAMES.get(split)
        if emb_path is not None and emb_path.is_file():
            with open(emb_path, "rb") as f:
                embeddings = pickle.load(f)
            # Match length in case pickle was saved for full split and we requested num_utterances
            n = len(df)
            df["embedding"] = list(embeddings[:n])
        return df
