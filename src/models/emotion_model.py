import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.configuration import *
from src.clustering.cluster_knn_classifier import ClusterKNNClassifier


class EmotionModel:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emotion_vectors = None
        self.emotion_method_id = next(iter(EMOTION_REPRESENTATIONS))
        self.context_method_id = next(iter(CONTEXT_METHODS))
        self.context_window = 0
        self.cluster_classifiers = []

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

    def add_cluster_classifier(self, variation, n_neighbors):
        self.cluster_classifiers.append(ClusterKNNClassifier(variation, n_neighbors=n_neighbors))

    def fit(self, training_set):
        if "embedding" not in training_set.columns:
            training_set.embedding = self.model.encode(training_set.text)

        for classifier in self.cluster_classifiers:
            classifier.fit(training_set)

        if self.emotion_method_id == 0:
            self.emotion_vectors = dict(zip(EMOTIONS, self.model.encode(EMOTIONS)))
        elif self.emotion_method_id == 1:
            self.emotion_vectors = self.get_averaged_emotion_embeddings(training_set)
        elif self.emotion_method_id == 2:
            emotion_vectors = {}
            conversations = training_set.groupby(by="conversation_ID")
            for _, conversation in conversations:
                conversation = conversation.sort_values("utterance_ID")
                for idx, row in conversation.iterrows():
                    utterances = conversation.loc[conversation.utterance_ID <= row.utterance_ID]
                    embeddings = list(utterances.embedding)
                    if self.context_window < len(embeddings) - 1:
                        embeddings = embeddings[-(self.context_window + 1):]
                    embeddings = np.asarray(embeddings)
                    if self.context_method_id == 0:
                        weighted_embedding = np.mean(embeddings, axis=0)
                    elif self.context_method_id == 1:
                        weights = np.array([2**i for i in range(len(embeddings))], dtype=float)
                        weighted_embedding = np.average(embeddings, axis=0, weights=weights)
                        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)
                    else:
                        raise ValueError("Context method not recognized")
                    emotion_vectors.setdefault(row.emotion, []).append(weighted_embedding)
            for emotion in emotion_vectors:
                emotion_vectors[emotion] = np.mean(emotion_vectors[emotion], axis=0)
            self.emotion_vectors = emotion_vectors

        else:
            raise ValueError("Emotion representation method not recognized")

    def get_averaged_emotion_embeddings(self, utterances):
        """
        Calculates the emotion vectors by taking the average of the embeddings of all the corresponding utterances.
        """
        embeddings = utterances.embedding
        emotion_vectors = {}
        for emb, emotion in zip(embeddings, utterances.emotion):
            emotion_vectors.setdefault(emotion, []).append(emb)
        for emotion in emotion_vectors:
            emotion_vectors[emotion] = np.mean(emotion_vectors[emotion], axis=0)
        return emotion_vectors

    def predict(self, utterances=None, embeddings=None):
        if self.emotion_vectors is None:
            raise ValueError("Model must be fit before predicting")
        if embeddings is None:
            if utterances is None:
                raise ValueError("No utterances provided")
            embeddings = self.model.encode(utterances)
        embeddings = np.asarray(embeddings)
        if self.context_window < len(embeddings) - 1:
            embeddings = embeddings[-(self.context_window + 1) :]


        if self.context_method_id == 0:
            context_embedding = np.mean(embeddings, axis=0)

        elif self.context_method_id == 1:
            weights = np.array([2**i for i in range(len(embeddings))], dtype=float)
            context_embedding = np.average(embeddings, axis=0, weights=weights)
            context_embedding = context_embedding / np.linalg.norm(context_embedding)

        else:
            raise ValueError("Context method not recognized")

        # Call each cluster classifier once; probas[i] is (n_emotions,) in EMOTIONS order
        cluster_probas = [clf.predict_emotion_probas(context_embedding) for clf in self.cluster_classifiers]

        def modified_similarity(emotion):
            sim = self.cosine_similarity(context_embedding, self.emotion_vectors[emotion])
            if not cluster_probas:
                return sim
            # Multiply by the probability for this emotion from each cluster classifier
            proba_product = 1.0
            for probas in cluster_probas:
                proba_product *= probas[emotion]
            return sim * proba_product

        return max(self.emotion_vectors, key=modified_similarity)


