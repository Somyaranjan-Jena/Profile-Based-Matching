import os
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.preprocess import clean_text

DATA_PATH = "data/users.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)


def load_users():
    users = pd.read_csv(DATA_PATH)

    users["combined_text"] = (
        users["professional_summary"].fillna("") +
        " " +
        users["about_me"].fillna("")
    )

    users["processed_text"] = users["combined_text"].apply(clean_text)

    return users


users = load_users()


def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loaded cached embeddings.")
        return np.load(EMBEDDINGS_PATH)

    print("Generating embeddings...")

    embeddings = model.encode(
        users["processed_text"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print("Embeddings saved.")

    return embeddings


embeddings = load_embeddings()

similarity_matrix = cosine_similarity(embeddings)


def get_similarity(user_index, other_index):
    return float(similarity_matrix[user_index][other_index] * 100)


def get_top_matches(user_id, top_n=5):

    if user_id not in users["user_id"].values:
        raise ValueError(f"User '{user_id}' not found.")

    idx = users.index[users["user_id"] == user_id][0]

    scores = similarity_matrix[idx]

    result = users.copy()

    result["Semantic"] = scores * 100

    result = result[result["user_id"] != user_id]

    result = result.sort_values(
        "Semantic",
        ascending=False
    )

    return result.head(top_n).reset_index(drop=True)


if __name__ == "__main__":

    sample = users.iloc[0]["user_id"]

    print(f"\nTop matches for {sample}\n")

    matches = get_top_matches(sample)

    print(
        matches[
            [
                "user_id",
                "name",
                "profession",
                "Semantic",
            ]
        ]
    )