

import cv2
import time
import numpy as np
import mediapipe as mp

from align import align_face
from embed import ArcFaceEmbedder
from evaluate import load_database, match


IDX = [33, 263, 1, 61, 291]
THRESHOLD = 0.5
LOCK_TIMEOUT = 30  # frames


def main():
    selected_identity = input("Enter identity to lock: ").strip().lower()

    database = load_database()
    if selected_identity not in database:
        raise RuntimeError("Selected identity not enrolled")

    embedder = ArcFaceEmbedder()
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(0)

    locked = False
    locked_name = None
    last_seen = 0
    frame_id = 0

    history = []
    start_time = time.strftime("%Y%m%d%H%M%S")
    history_file = f"data/{selected_identity}_history_{start_time}.txt"

    prev_x = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        H, W = frame.shape[:2]

        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm[i].x * W, lm[i].y * H] for i in IDX], dtype=np.float32)

            # enforce order
            if pts[0, 0] > pts[1, 0]:
                pts[[0, 1]] = pts[[1, 0]]
            if pts[3, 0] > pts[4, 0]:
                pts[[3, 4]] = pts[[4, 3]]

            aligned = align_face(frame, pts)
            emb = embedder.embed(aligned)

            name, score = match(emb, database, THRESHOLD)

            # ---- LOCK LOGIC ----
            if not locked and name == selected_identity:
                locked = True
                locked_name = name
                last_seen = frame_id
                print("🔒 FACE LOCKED")

            if locked:
                if name == locked_name:
                    last_seen = frame_id

                # ---- ACTION DETECTION ----
                cx = pts[2, 0]  # nose x
                if prev_x is not None:
                    if cx - prev_x > 5:
                        record(history, "move_right")
                    elif prev_x - cx > 5:
                        record(history, "move_left")
                prev_x = cx

                cv2.putText(frame, f"LOCKED: {locked_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- UNLOCK ----
        if locked and frame_id - last_seen > LOCK_TIMEOUT:
            print("🔓 FACE UNLOCKED")
            save_history(history_file, history)
            break

        cv2.imshow("Face Locking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def record(history, action):
    ts = time.strftime("%H:%M:%S")
    history.append((ts, action))
    print(ts, action)


def save_history(path, history):
    with open(path, "w") as f:
        for ts, act in history:
            f.write(f"{ts} | {act}\n")
    print("History saved:", path)


if __name__ == "__main__":
    main()
