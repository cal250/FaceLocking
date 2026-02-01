
# ARCFACE ONNX — Face Locking

High-performance face recognition project using an ArcFace ONNX embedder and lightweight Python tooling to detect, align, enroll, and recognize faces. Designed for offline, local use with simple scripts for enrollment, recognition, and evaluation.

## Key Features
- ONNX ArcFace embedding model (`models/embedder_arcface.onnx`) for fast, device-agnostic embedding extraction.
- Face detection, alignment, and landmarks utilities.
- Enrollment pipeline to build a searchable face database (`book/db/face_db.*`).
 # ARCFACE ONNX — Face Locking (Face-Based Lock/Unlock Toolkit)

Face-locking toolkit using ArcFace ONNX embeddings to enable local, offline face-based locking and access control. Provides detection, alignment, enrollment, and real-time recognition for device or application lock/unlock workflows. See `src/face_locking.py` for integration helpers.

## Key Features

- ONNX ArcFace embedding model (`models/embedder_arcface.onnx`) for fast, device-agnostic embedding extraction.
- Face detection, alignment, and landmarks utilities.
- Enrollment pipeline to build a searchable face database (`book/db/face_db.*`).
- Recognition and face-locking scripts for live or recorded input.
- Evaluation tools to measure accuracy on enrolled datasets.

## Repository Structure

- `models/` — ONNX model(s) used for embedding extraction.
- `src/` — Main Python scripts and modules:
  - `align.py`, `landmarks.py` — face alignment and landmark utilities
  - `detect.py`, `haar.py` — face detectors
  - `enroll.py` — enroll identities into the face DB
  - `recognize.py` — run recognition (camera or image)
  - `evaluate.py` — evaluation utilities
  - `camera.py`, `face_locking.py` — helpers for live capture and locking
  - `ambed.py` — embedding helper utilities
  - `README.md` — module-level notes
- `book/` — dataset and enrollment storage
  - `db/face_db.json`, `db/face_db.npz` — enrolled identities and embeddings
  - `enroll/` — folders containing enrolled face images per identity
- `data/` — ancillary data (if used)

## Requirements

- Python 3.8+ (Windows supported)

Install dependencies (minimal example):

```bash
python -m pip install -r requirements.txt
```

If a `requirements.txt` is not present, typical dependencies include:

```bash
python -m pip install numpy opencv-python onnxruntime scikit-learn
```

Use `onnxruntime-gpu` for GPU acceleration if desired and available.

## Quick Start

1. Ensure `models/embedder_arcface.onnx` is present in `models/`.
2. Enroll identities from images (one folder per identity):

```bash
python src/enroll.py --input book/enroll/Calvin --name Calvin
```

3. Run recognition (camera):

```bash
python src/recognize.py --db book/db/face_db.npz --model models/embedder_arcface.onnx --source 0
```

4. Run face-locking helper (example integration):

```bash
python src/face_locking.py --db book/db/face_db.npz --model models/embedder_arcface.onnx
```

5. Evaluate using the provided script:

```bash
python src/evaluate.py --db book/db/face_db.npz
```

Refer to each script's header or `--help` for supported CLI flags and options.

## Typical Workflows

- Enroll a new person:
  - Add images to `book/enroll/<PersonName>/`
  - Run `python src/enroll.py --input book/enroll/<PersonName> --name <PersonName>`

- Recognize from camera or video file:
  - `python src/recognize.py --db book/db/face_db.npz --model models/embedder_arcface.onnx --source 0`

- Integrate face locking into an application:
  - Use `src/face_locking.py` or import `src/face_locking` functions to trigger lock/unlock actions based on recognition confidence.

## Implementation Notes

- Detection, alignment, and embedding are separated so you can swap detectors or embedders.
- Embeddings and metadata are stored in `book/db/` as `.npz` and `.json` for fast search and easy inspection.
- Use `onnxruntime` for model inference; switch to `onnxruntime-gpu` for GPU acceleration.

## Troubleshooting

- If faces are not detected reliably, try the Haar detector (`src/haar.py`) or adjust detector thresholds.
- For poor recognition accuracy, verify alignment and use high-quality, frontal enrollment images.

## Contributing

1. Fork the repo and create a feature branch.
2. Add tests or a demonstration script for new functionality.
3. Open a pull request with a clear description of changes.

## Attribution & References

- ArcFace / InsightFace research inspired the embedding approach used here.
- This project uses an ONNX-format ArcFace embedder for portability.

## License

Add your preferred license file (e.g., `LICENSE` with MIT) if desired.

---

If you want, I can also:
- Add a `requirements.txt` with pinned packages.
- Expand `src/` READMEs with per-script usage and examples.
- Add an `examples/` folder with a small enrollment+recognition demo.






