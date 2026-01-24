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
import numpy as np
import mediapipe as mp

from align import align_face
from embed import ArcFaceEmbedder


DATA_DIR = "data/enroll"
IDX = [33, 263, 1, 61, 291]


def enroll(person_name, samples=10):
    os.makedirs(DATA_DIR, exist_ok=True)
    person_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    )
    embedder = ArcFaceEmbedder()

    count = 0
    print(f"Enrolling '{person_name}' — press SPACE to capture")

    while count < samples:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm[i].x * W, lm[i].y * H] for i in IDX], dtype=np.float32)

            if pts[0, 0] > pts[1, 0]:
                pts[[0, 1]] = pts[[1, 0]]
            if pts[3, 0] > pts[4, 0]:
                pts[[3, 4]] = pts[[4, 3]]

            aligned = align_face(frame, pts)

            cv2.imshow("Aligned", aligned)
            cv2.putText(
                frame,
                f"{count}/{samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Enroll", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32 and res.multi_face_landmarks:
            emb = embedder.embed(aligned)
            np.save(
                os.path.join(person_dir, f"{count}.npy"),
                emb
            )
            print(f"Saved sample {count}")
            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Enrollment complete.")


if __name__ == "__main__":
    name = input("Enter person name: ").strip()
    enroll(name)
