import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from configuration import *

EMOTION_VECTORS_FILENAME = "saved_data/averaged_emotion_vectors.npy"

class EmotionModel:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emotion_vectors = None
        self.emotion_method_id = next(iter(EMOTION_REPRESENTATIONS))
        self.context_method_id = next(iter(CONTEXT_METHODS))
        self.context_window = 0

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def encode(self, texts):
        return self.model.encode(texts)

    def set_emotion_representation_method(self, method_id):
        if method_id not in EMOTION_REPRESENTATIONS:
            raise ValueError("Emotion representation method not recognized")
        self.emotion_method_id = method_id

    def get_emotion_representation_method(self):
        return EMOTION_REPRESENTATIONS[self.emotion_method_id]

    def set_context_method(self, method_id):
        if method_id not in CONTEXT_METHODS:
            raise ValueError("Context method not recognized")
        self.context_method_id = method_id

    def get_context_method(self):
        return CONTEXT_METHODS[self.context_method_id]

    def set_context_window(self, context_window):
        self.context_window = context_window

    def fit(self, training_text=None, training_labels=None):
        if self.emotion_method_id == 0:
            self.emotion_vectors = dict(zip(EMOTIONS, self.model.encode(EMOTIONS)))
        elif self.emotion_method_id == 1:
            self.emotion_vectors = self.get_averaged_emotion_embeddings(training_text, training_labels)

    def get_averaged_emotion_embeddings(self, text, labels):
        """
        Calculates the emotion vectors by taking the average of the embeddings of all the corresponding utterances.
        """
        if Path(EMOTION_VECTORS_FILENAME).is_file():
            return np.load(EMOTION_VECTORS_FILENAME, allow_pickle=True).item()
        embeddings = self.model.encode(text)
        emotion_vectors = {}
        for emb, label in zip(embeddings, labels):
            emotion_vectors.setdefault(label, []).append(emb)
        for emotion in emotion_vectors:
            emotion_vectors[emotion] = np.mean(emotion_vectors[emotion], axis=0)
        np.save(EMOTION_VECTORS_FILENAME, emotion_vectors)
        return emotion_vectors

    def predict(self, utterances=None, embeddings=None):
        if self.emotion_vectors is None:
            raise ValueError("Model must be fit before predicting")
        if embeddings is None:
            if utterances is None:
                raise ValueError("No utterances provided")
            embeddings = self.model.encode(utterances)
        if self.context_window < len(embeddings) - 1:
            embeddings = embeddings[-(self.context_window+1):]
        if self.context_method_id == 0:
            average_embedding = np.mean(embeddings, axis=0)
            get_similarity = lambda key: self.cosine_similarity(average_embedding, self.emotion_vectors[key])
            return max(self.emotion_vectors, key=get_similarity)
        if self.context_method_id == 1:
            max_emotion = None
            max_similarity = -2
            for emotion in self.emotion_vectors:
                total_similarity = 0
                for i, embedding in enumerate(embeddings):
                    similarity = self.cosine_similarity(embedding, self.emotion_vectors[emotion])
                    total_similarity += similarity*(2**i)
                if total_similarity > max_similarity:
                    max_similarity = total_similarity
                    max_emotion = emotion
            if max_emotion is None: print("Max emotion was not found.")
            return max_emotion
        return None


