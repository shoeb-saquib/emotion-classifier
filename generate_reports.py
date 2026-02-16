from ecfdataset import ECFDataset
from emotion_model import EmotionModel
from evaluate import *
from configuration import *
from pathlib import Path
import pickle

TEST_EMBEDDINGS_FILENAME = "saved_data/test_embeddings.pkl"

dataset = ECFDataset()
training_set = dataset.load_split("train")
test_set = dataset.load_split("test")

model = EmotionModel()

if Path(TEST_EMBEDDINGS_FILENAME).is_file():
    test_set["embedding"] = pickle.load(open(TEST_EMBEDDINGS_FILENAME, "rb"))
else:
    test_set["embedding"] = model.encode(list(test_set.text)).tolist()
    pickle.dump(test_set.embedding.tolist(), open(TEST_EMBEDDINGS_FILENAME, "wb"))

test_conversations = test_set.groupby(by="conversation_ID")
for emotion_method_id in SELECTED_EMOTION_REPRESENTATIONS:
    model.set_emotion_representation_method(emotion_method_id)
    model.fit(list(training_set.text), list(training_set.emotion))
    one_context_method_has_finished = False
    for context_method_id in SELECTED_CONTEXT_METHODS:
        model.set_context_method(context_method_id)
        for context_window in CONTEXT_WINDOWS:
            if context_window == 0 and one_context_method_has_finished: continue
            model.set_context_window(context_window)
            predictions = {}
            for _, conversation in test_conversations:
                conversation = conversation.sort_values("utterance_ID")
                for idx, row in conversation.iterrows():
                    utterances = conversation.loc[conversation.utterance_ID <= row.utterance_ID]
                    predictions[idx] = model.predict(embeddings=list(utterances.embedding))
            predictions = [predictions[i] for i in test_set.index]
            descriptors = {
                DESCRIPTORS[1] : EMOTION_REPRESENTATIONS[emotion_method_id],
                DESCRIPTORS[2] : context_window
            }
            filename = "reports/"
            if context_window == 0:
                filename += f'er{emotion_method_id}_cw0_report.txt'
            else:
                method_tag = f'er{emotion_method_id}_cm{context_method_id}'
                filename += f'{method_tag}/{method_tag}_cw{context_window}_report.txt'
                descriptors[DESCRIPTORS[3]] = CONTEXT_METHODS[context_method_id]
            generate_evaluation_report(test_set.emotion, predictions, descriptors, filename)
            one_context_method_has_finished = True









