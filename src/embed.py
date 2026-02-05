# src/embed.py
"""
ArcFace embedding using ONNX Runtime.

Input:
 - aligned face (112x112 BGR)

Output:
 - L2-normalized embedding (1D vector)
"""

import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceEmbedder:
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, img):
        """
        img: 112x112 BGR
        returns: (1,3,112,112) float32
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def embed(self, img):
        """
        img: aligned face (112x112 BGR)
        returns: normalized embedding vector
        """
        blob = self.preprocess(img)
        emb = self.sess.run(
            [self.output_name],
            {self.input_name: blob}
        )[0]

        # L2 normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0]


# standalone test
if __name__ == "__main__":
    from align import align_face
    import mediapipe as mp

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    )

    IDX = [33, 263, 1, 61, 291]
    embedder = ArcFaceEmbedder()

    while True:
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
            emb = embedder.embed(aligned)

            cv2.imshow("Aligned", aligned)
            print("Embedding shape:", emb.shape)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
