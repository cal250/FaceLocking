# 🧠 Face Locking System
Face Recognition with ArcFace ONNX + Face Locking + Action Tracking (CPU-Only)

---

## 1. Project Overview

This project implements a **Face Locking system** built on top of a **Face Recognition pipeline using ArcFace ONNX and 5-Point Face Alignment**.

The system first recognizes **who the person is**, then **locks onto that identity** and tracks the same person and their actions over time.

Once a face is locked:
- the system stays focused on the selected identity
- other faces are ignored
- brief recognition failures are tolerated
- actions are detected and recorded to a history file

This project demonstrates how face recognition can be extended into **identity-aware behavior tracking**, which is a real-world computer vision feature.

---

## 2. Assignment Goal

The goal of this assignment is to extend an existing face recognition system by adding **Face Locking**, such that:

- one enrolled identity is manually selected
- the system locks onto that identity when recognized
- tracking remains stable over time
- simple face actions are detected
- action history is recorded
- the system runs on CPU only
- no model retraining is performed

---

## 3. What Is Face Locking?

**Face locking** means:

> Recognize who the person is first, then track that same person and their actions across time.

Important clarifications:
- Face locking is **not face detection**
- Face locking is **not repeated recognition**
- Face locking **depends on recognition to initialize**
- After locking, tracking becomes the main behavior

Even if recognition briefly fails, the lock remains active.

---

## 4. High-Level System Pipeline

The system operates as the following pipeline:

1. Camera input  
2. Face detection  
3. 5-point facial landmark extraction  
4. Face alignment (112×112)  
5. ArcFace embedding extraction (ONNX)  
6. Face recognition (cosine similarity)  
7. Face locking (state management)  
8. Action detection (while locked)  
9. Action history recording  

---

## 5. Technologies Used

- Python 3
- OpenCV
- MediaPipe
- NumPy
- ONNX Runtime
- ArcFace pretrained ONNX model

---

## 6. Constraints (Mandatory)

- CPU-only execution  
- No retraining of models  
- Pretrained ArcFace ONNX model  
- Recognition pipeline must not be broken  

---

## 7. Project Structure

Face-Locking/
├── data/
│ ├── enroll/
│ │ └── <person_name>/
│ │ ├── 0.npy
│ │ ├── 1.npy
│ │ └── ...
│ └── <face>history<timestamp>.txt
│
├── models/
│ └── embedder_arcface.onnx
│
├── src/
│ ├── camera.py # Camera test
│ ├── detect.py # Face detection
│ ├── landmarks.py # 5-point landmarks
│ ├── align.py # Face alignment
│ ├── embed.py # ArcFace embeddings
│ ├── enroll.py # Face enrollment
│ ├── evaluate.py # Similarity & matching
│ ├── recognize.py # Real-time recognition
│ └── face_locking.py # Face locking + actions
│
└── README.md


---

## 8. Installation & Requirements

Install required Python packages:

```bash
pip install opencv-python mediapipe numpy onnxruntime


python -m src.camera

python -m src.enroll


python -m src.face_locking
