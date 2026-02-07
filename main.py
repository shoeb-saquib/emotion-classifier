from ecfdataset import ECFDataset
from emotion_model import EmotionModel
from evaluate import *

METHODS = {
    0 : "embed emotion words directly",
    1 : "average utterance embeddings for each emotion"
}
METHOD_ID = 0

dataset = ECFDataset()
train_texts, train_labels = dataset.load_split("train")
test_texts, test_labels = dataset.load_split("test")

print_majority_baseline(test_labels)

model = EmotionModel()
model.set_method(METHOD_ID)
model.fit(test_texts, test_labels)
predicted_labels = model.predict(test_texts)
print_report(test_labels, predicted_labels, METHODS[METHOD_ID])



