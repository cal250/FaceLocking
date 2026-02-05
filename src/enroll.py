# src/enroll.py
"""
Face enrollment:
 - capture face
 - align
 - embed
 - store embedding
"""

import cv2
import os
import json
import numpy as np
from src.embed import ArcFaceEmbedder
from src.detect import detect_faces
from src.align import align_face


DB_DIR = "book/db"
NPZ_PATH = os.path.join(DB_DIR, "face_db.npz")
JSON_PATH = os.path.join(DB_DIR, "face_db.json")

def enroll_from_camera(name, num_samples=20):
    os.makedirs(DB_DIR, exist_ok=True)

    embedder = ArcFaceEmbedder("models/embedder_arcface.onnx")
    cap = cv2.VideoCapture(0)

    embeddings = []
    print("[INFO] Press SPACE to capture face | ESC to quit")

    while len(embeddings) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for face in faces:
            aligned = align_face(frame, face)
            emb = embedder.get_embedding(aligned)
            embeddings.append(emb)

            print(f"[INFO] Captured {len(embeddings)}/{num_samples}")
            break

        cv2.imshow("Enroll Face", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    embeddings = np.array(embeddings)

    # Load existing DB
    if os.path.exists(NPZ_PATH):
        data = np.load(NPZ_PATH, allow_pickle=True)
        db_embeddings = dict(data)
    else:
        db_embeddings = {}

    db_embeddings[name] = embeddings

    np.savez(NPZ_PATH, **db_embeddings)

    # Save metadata
    with open(JSON_PATH, "w") as f:
        json.dump({k: len(v) for k, v in db_embeddings.items()}, f, indent=2)

    print("[SUCCESS] Enrollment completed!")

if __name__ == "__main__":
    person_name = input("Enter your name: ")
    enroll_from_camera(person_name)
