import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMOTION_VECTORS_FILENAME = "saved_data/emotion_vectors.npy"

class EmotionModel:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emotion_vectors = None
        self.method = "word"

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def set_method(self, method):
        self.method = method

    def fit(self, texts, labels):
        if self.method == 0:
            self.emotion_vectors = self.get_emotion_embeddings(labels)
        elif self.method == 1:
            self.emotion_vectors = self.get_averaged_emotion_embeddings(texts, labels)
        else:
            print("Invalid method")

    def get_emotion_embeddings(self, labels):
        """
        Returns a dictionary of emotions and their embeddings.
        """
        emotions = list(set(labels))
        return dict(zip(emotions, self.model.encode(emotions)))

    def get_averaged_emotion_embeddings(self, texts, labels):
        """
        Calculates the emotion vectors by taking the average of the embeddings of all the corresponding utterances.
        """
        if Path(EMOTION_VECTORS_FILENAME).is_file():
            return np.load(EMOTION_VECTORS_FILENAME, allow_pickle=True).item()

        embeddings = self.model.encode(texts)
        emotion_vectors = {}

        for emb, label in zip(embeddings, labels):
            emotion_vectors.setdefault(label, []).append(emb)
        for emotion in self.emotion_vectors:
            emotion_vectors[emotion] = np.mean(self.emotion_vectors[emotion], axis=0)
        np.save(EMOTION_VECTORS_FILENAME, emotion_vectors)
        return emotion_vectors

    def predict_embedding(self, embedding):
        get_similarity = lambda emotion: self.cosine_similarity(embedding, self.emotion_vectors[emotion])
        return max(self.emotion_vectors, key=get_similarity)

    def predict(self, texts):
        embeddings = self.model.encode(texts)
        return [self.predict_embedding(embedding) for embedding in embeddings]

