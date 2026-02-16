EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

BASELINES = ["randomly predict emotions", "predict majority emotion"]

DESCRIPTORS = ["Baseline Method", "Emotion Representation", "Context Window", "Context Method"]

EMOTION_REPRESENTATIONS = {
    0 : "direct emotion label embeddings",
    1 : "averaged emotion utterance embeddings"
}

CONTEXT_METHODS = {
    0 : "average with previous utterance embeddings",
    1 : "exponentially weight previous utterance embeddings"
}

SELECTED_EMOTION_REPRESENTATIONS = [1]

SELECTED_CONTEXT_METHODS = [1]

CONTEXT_WINDOWS = [7, 8, 9, 10]


