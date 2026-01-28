import pickle
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

EMOTION_VECTORS_FILENAME = "emotion_vectors.npy"
DATA_FILENAMES = ["training_data.pkl", "validation_data.pkl", "test_data.pkl"]
SPLITS = ["train", "validation", "test"]

def retrieve_dataset():
    """
    Get the dataset and save the train, validation, and test data to pickle files.
    """
    dataset = load_dataset("NUSTM/ECF", streaming=True)
    for split, filename in zip(SPLITS, DATA_FILENAMES):
        data = dataset[split]
        texts = []
        emotions = []
        for item in data:
            for utterance in item["conversation"]:
                texts.append(utterance["text"])
                emotions.append(utterance["emotion"])
        with open(filename, "wb") as f:
            pickle.dump((texts, emotions), f)

def compute_similarity(vector1, vector2):
    """
    Compute the similarity between two vectors using cosine similarity.
    """
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class Classifier:
    def __init__(self):
        """
        Load data into class attributes and calculate the emotion vectors.
        """
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.training_labels = None
        self.validation_labels = None
        self.test_labels = None
        self.load_data()

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.emotion_vectors = None
        self.load_emotion_vectors()

    def check_accuracy(self):
        """
        Check the accuracy of the model with the test data.
        """
        num_texts = len(self.test_data)
        successes = 0
        embeddings = self.model.encode(self.test_data)
        for embedding, label in zip(embeddings, self.test_labels):
            if label == self.classify_vector(embedding): successes += 1
        print(f"Tests Passed: {successes}/{num_texts}")
        print(f"Accuracy: {successes/num_texts * 100:.2f}%")


    def classify_utterance(self, utterance):
        """
        Takes in a sentence and returns the most likely emotion.
        """
        return self.classify_vector(self.model.encode(utterance))

    def classify_vector(self, vector):
        """
        Takes in an embedding vector and returns the most likely emotion.
        """
        max_similarity = -2
        closest_emotion = None
        for emotion in self.emotion_vectors:
            similarity = compute_similarity(vector, self.emotion_vectors[emotion])
            if similarity > max_similarity:
                max_similarity = similarity
                closest_emotion = emotion
        return closest_emotion

    def load_data(self):
        """
        Load the data from the pickle files into the class attributes.
        If the files don't exist, then it retrieves the dataset.
        """
        for filename in DATA_FILENAMES:
            if not Path(filename).is_file():
                retrieve_dataset()
            with open(filename, "rb") as f:
                if filename == DATA_FILENAMES[0]:
                    self.training_data, self.training_labels = pickle.load(f)
                if filename == DATA_FILENAMES[1]:
                    self.validation_data, self.validation_labels = pickle.load(f)
                if filename == DATA_FILENAMES[2]:
                    self.test_data, self.test_labels = pickle.load(f)

    def load_emotion_vectors(self):
        """
        Loads the emotion vectors from the numpy file into the class attribute.
        If the file doesn't exist, then it calculates the emotion vectors.'
        """
        if not Path(EMOTION_VECTORS_FILENAME).is_file():
            self.calculate_average_emotion_vectors()
        self.emotion_vectors = np.load(EMOTION_VECTORS_FILENAME, allow_pickle=True).item()

    def calculate_average_emotion_vectors(self):
        """
        Calculates the vectors representing the emotions by taking the average of the embeddings from the training data.
        """
        embeddings = self.model.encode(self.training_data)

        emotion_vectors = {}
        for embedding, label in zip(embeddings, self.training_labels):
            if label not in emotion_vectors:
                emotion_vectors[label] = []
            emotion_vectors[label].append(embedding)

        for emotion in emotion_vectors:
            emotion_vectors[emotion] = np.mean(emotion_vectors[emotion], axis=0)

        np.save(EMOTION_VECTORS_FILENAME, emotion_vectors)
