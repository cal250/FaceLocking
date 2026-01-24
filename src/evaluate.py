# src/evaluate.py
"""
Evaluate face embeddings using cosine similarity.
"""

import os
import numpy as np


DB_DIR = "data/enroll"


def load_database(db_dir=DB_DIR):
    """
    Load embeddings from enrollment directory.

    Returns:
        dict: {person_name: [embeddings]}
    """
    database = {}

    for person in os.listdir(db_dir):
        person_dir = os.path.join(db_dir, person)
        if not os.path.isdir(person_dir):
            continue

        embs = []
        for f in os.listdir(person_dir):
            if f.endswith(".npy"):
                emb = np.load(os.path.join(person_dir, f))
                embs.append(emb)

        if embs:
            database[person] = np.stack(embs)

    return database


def cosine_similarity(a, b):
    """
    a: (D,)
    b: (N,D)
    returns: (N,)
    """
    return np.dot(b, a)


def match(embedding, database, threshold=0.5):
    """
    Match embedding against database.

    Returns:
        (name, score) or ("unknown", best_score)
    """
    best_score = -1
    best_name = "unknown"

    for name, embs in database.items():
        scores = cosine_similarity(embedding, embs)
        score = scores.max()

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "unknown", best_score

    return best_name, best_score


# standalone test
if __name__ == "__main__":
    db = load_database()
    print("Loaded identities:", list(db.keys()))

    # sanity test (compare same-person embeddings)
    for name, embs in db.items():
        if len(embs) > 1:
            s = cosine_similarity(embs[0], embs[1:])
            print(f"{name} self-similarity:", s.mean())
