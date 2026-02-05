# src/align.py
import cv2
import numpy as np

# Reference 5-point landmarks for ArcFace (112x112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align(img, landmarks, image_size=(112, 112)):
    landmarks = np.array(landmarks, dtype=np.float32)

    transform = cv2.estimateAffinePartial2D(
        landmarks,
        REFERENCE_LANDMARKS,
        method=cv2.LMEDS
    )[0]

    aligned = cv2.warpAffine(
        img,
        transform,
        image_size,
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )

    return aligned
