# Configure below before running generate_reports.py or delete_reports.py
SELECTED_EMOTION_REPRESENTATIONS = [2]

SELECTED_CONTEXT_METHODS = [1]

CONTEXT_WINDOWS = [2]

SELECTED_CLUSTER_VARIATIONS = ["h_default"]
KNN_NEIGHBORS = 25
GOALEX_HYBRID_ALPHA = 1.0
GOALEX_PER_EMOTION_FUSION_TOPK = 1

# Add new emotion representations here
EMOTION_REPRESENTATIONS = {
    0 : "direct emotion label embeddings",
    1 : "averaged emotion utterance embeddings",
    2 : "averaged emotion utterance embeddings with context"
}

# Add new context methods here
CONTEXT_METHODS = {
    0 : "average with previous utterance embeddings",
    1 : "exponentially weight previous utterance embeddings"
}

# General constants
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

BASELINES = ["randomly predict emotions", "predict majority emotion"]

DESCRIPTORS = ["Baseline Method", "Emotion Representation", "Context Window", "Context Method"]