import argparse
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer

from src.data.ecfdataset import ECFDataset, EMBEDDING_FILENAMES


def _ensure_embeddings(df, split: str, model_name: str):
    if "embedding" in df.columns:
        return df
    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(df["text"]))
    df = df.copy()
    df["embedding"] = list(embeddings)
    out = EMBEDDING_FILENAMES[split]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(embeddings, f)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="KNN baseline: classify test embeddings from train embeddings."
    )
    parser.add_argument("--k", type=int, default=25, help="Number of neighbors.")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric passed to sklearn KNeighborsClassifier.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="distance",
        choices=["uniform", "distance"],
        help="Neighbor weighting scheme.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name, used only if saved embeddings are missing.",
    )
    args = parser.parse_args()

    dataset = ECFDataset()
    train_df = dataset.load_split("train")
    test_df = dataset.load_split("test")

    train_df = _ensure_embeddings(train_df, "train", args.model_name)
    test_df = _ensure_embeddings(test_df, "test", args.model_name)

    x_train = list(train_df["embedding"])
    y_train = list(train_df["emotion"])
    x_test = list(test_df["embedding"])
    y_test = list(test_df["emotion"])

    clf = KNeighborsClassifier(
        n_neighbors=args.k, metric=args.metric, weights=args.weights
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"KNN embedding baseline")
    print(f"k={args.k}, metric={args.metric}, weights={args.weights}")
    print(f"accuracy={acc:.4f}")
    print(f"macro_f1={macro_f1:.4f}")


if __name__ == "__main__":
    main()
