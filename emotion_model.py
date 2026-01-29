import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMOTION_VECTORS_FILENAME = "emotion_vectors.npy"

class EmotionModel:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emotion_vectors = None

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def fit_words(self, labels):
        """
        Returns a dictionary of emotions and their embeddings.
        """
        emotions = list(set(labels))
        self.emotion_vectors = dict(zip(emotions, self.model.encode(emotions)))

    def fit_average(self, texts, labels):
        """
        Calculates the emotion vectors by taking the average of the embeddings of all the corresponding utterances.
        """
        if Path(EMOTION_VECTORS_FILENAME).is_file():
            self.emotion_vectors = np.load(EMOTION_VECTORS_FILENAME, allow_pickle=True).item()
            return

        embeddings = self.model.encode(texts)
        self.emotion_vectors = {}

        for emb, label in zip(embeddings, labels):
            self.emotion_vectors.setdefault(label, []).append(emb)

        for emotion in self.emotion_vectors:
            self.emotion_vectors[emotion] = np.mean(self.emotion_vectors[emotion], axis=0)
        np.save(EMOTION_VECTORS_FILENAME, self.emotion_vectors)

    def predict(self, text):
        embedding = self.model.encode(text)
        get_similarity = lambda emotion: self.cosine_similarity(embedding, self.emotion_vectors[emotion])
        return max(self.emotion_vectors, key = get_similarity)
