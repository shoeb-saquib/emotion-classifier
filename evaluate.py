
def get_accuracy(model, texts, labels):
    correct = 0
    for text, label in zip(texts, labels):
        if model.predict(text) == label:
            correct += 1
    return correct / len(labels), correct
