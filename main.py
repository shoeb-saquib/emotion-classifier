from ecfdataset import ECFDataset
from emotion_model import EmotionModel
from evaluate import get_accuracy

dataset = ECFDataset()

train_texts, train_labels = dataset.load_split("train")
test_texts, test_labels = dataset.load_split("test", 1000)

model = EmotionModel()
model.fit_average(train_texts, train_labels)

accuracy, successes = get_accuracy(model, test_texts, test_labels)
print("Method: taking the average of the utterance embeddings")
print(f"Tests Passed: {successes} / {len(test_texts)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

model.fit_words(train_labels)

accuracy, successes = get_accuracy(model, test_texts, test_labels)
print("\nMethod: embing the emotions")
print(f"Tests Passed: {successes} / {len(test_texts)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

