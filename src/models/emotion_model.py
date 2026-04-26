import numpy as np
from sentence_transformers import SentenceTransformer
from src.configuration import *
from src.clustering.cluster_knn_classifier import ClusterKNNClassifier
from src.models.context_embedding import pool_utterance_embeddings


def _softmax_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    denom = np.sum(ex)
    return ex / denom if denom > 0 else np.full_like(x, 1.0 / len(x))


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
        self.cluster_classifiers.append(
            ClusterKNNClassifier(
                variation,
                n_neighbors=n_neighbors,
                goalex_hybrid_alpha=GOALEX_HYBRID_ALPHA,
                goalex_per_emotion_fusion_topk=GOALEX_PER_EMOTION_FUSION_TOPK,
            )
        )

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
                    emb_seq = np.asarray(list(utterances.embedding), dtype=np.float64)
                    weighted_embedding = pool_utterance_embeddings(
                        emb_seq, self.context_window, self.context_method_id
                    )
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
        context_embedding = pool_utterance_embeddings(
            embeddings, self.context_window, self.context_method_id
        )

        # Base emotion-representation similarities (always available)
        sim_vec = np.array(
            [self.cosine_similarity(context_embedding, self.emotion_vectors[e]) for e in EMOTIONS],
            dtype=np.float64,
        )
        if not self.cluster_classifiers:
            return EMOTIONS[int(np.argmax(_softmax_vec(sim_vec)))]

        n_emotions = len(EMOTIONS)
        prop_product = np.ones(n_emotions, dtype=np.float64)
        centroid_product = np.ones(n_emotions, dtype=np.float64)
        n_prop = 0
        n_centroid = 0

        for clf in self.cluster_classifiers:
            centroid_scores = clf.predict_emotion_scores_from_centroids(context_embedding)
            if centroid_scores is not None:
                n_centroid += 1
                centroid_product *= np.array(
                    [centroid_scores[e] for e in EMOTIONS], dtype=np.float64
                )
            else:
                n_prop += 1
                probas = clf.predict_emotion_probas(context_embedding)
                prop_product *= np.array([probas[e] for e in EMOTIONS], dtype=np.float64)

        if n_centroid == 0:
            # Proportion-based cluster evidence only (legacy).

            def modified_similarity(emotion):
                sim = self.cosine_similarity(context_embedding, self.emotion_vectors[emotion])
                idx = EMOTIONS.index(emotion)
                return sim * prop_product[idx]

            return max(self.emotion_vectors, key=modified_similarity)

        c_sum = float(np.sum(centroid_product))
        if c_sum > 0:
            c_norm = centroid_product / c_sum
        else:
            c_norm = np.full(n_emotions, 1.0 / n_emotions, dtype=np.float64)

        # Mirror proportion path: raw cosine × cluster evidence (per emotion).
        centroid_scores_vec = sim_vec * c_norm

        if n_prop == 0:
            return EMOTIONS[int(np.argmax(centroid_scores_vec))]

        final_vec = centroid_scores_vec * prop_product
        return EMOTIONS[int(np.argmax(final_vec))]


