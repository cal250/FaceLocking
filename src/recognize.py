# src/recognize.py
"""
Real-time face recognition using:
Haar + MediaPipe 5pt + ArcFace + cosine similarity
"""

import cv2
import numpy as np
import mediapipe as mp

from align import align_face
from embed import ArcFaceEmbedder
from evaluate import load_database, match


IDX = [33, 263, 1, 61, 291]
THRESHOLD = 0.5


def main():
    database = load_database()
    if not database:
        raise RuntimeError("No enrolled faces found. Run enroll.py first.")

    embedder = ArcFaceEmbedder()

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    print("Real-time recognition. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        label = "no face"
        score = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm[i].x * W, lm[i].y * H] for i in IDX], dtype=np.float32)

            # enforce landmark order
            if pts[0, 0] > pts[1, 0]:
                pts[[0, 1]] = pts[[1, 0]]
            if pts[3, 0] > pts[4, 0]:
                pts[[3, 4]] = pts[[4, 3]]

            aligned = align_face(frame, pts)
            emb = embedder.embed(aligned)

            label, score = match(emb, database, threshold=THRESHOLD)

            cv2.imshow("Aligned", aligned)

        # overlay result
        text = f"{label} ({score:.2f})"
        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
