
# ARCFACE ONNX — Face Recognition with ArcFace Embeddings

High-performance face recognition project using an ArcFace ONNX embedder and lightweight Python tooling to detect, align, enroll, and recognize faces. Designed for offline, local use with simple scripts for enrollment, recognition, and evaluation.

## Key Features
- ONNX ArcFace embedding model (`models/embedder_arcface.onnx`) for fast, device-agnostic embedding extraction.
- Face detection, alignment, and landmarks utilities.
- Enrollment pipeline to build a searchable face database (`book/db/face_db.*`).
- Recognition script for live or recorded input.
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
- Install dependencies (minimal example):

```bash
python -m pip install -r requirements.txt
```

If a `requirements.txt` is not present, typical dependencies include:

```bash
python -m pip install numpy opencv-python onnxruntime scikit-learn
```

Adjust packages to your environment (GPU-backed `onnxruntime-gpu` can be used for faster embedding).

## Quick Start

1. Place the ONNX embedder model in `models/embedder_arcface.onnx` (already present).
2. Enroll identities from images (one folder per identity). Example:

```bash
python src/enroll.py --input book/enroll/Calvin --name Calvin
```

3. Run recognition with your camera:

```bash
python src/recognize.py --db book/db/face_db.npz --model models/embedder_arcface.onnx
```

4. Evaluate using the provided script:

```bash
python src/evaluate.py --db book/db/face_db.npz
```

Refer to the individual scripts in `src/` for supported CLI flags and options.

## Typical Workflows

- Enroll new person:
	- Add images to `book/enroll/<PersonName>/`
	- Run `python src/enroll.py --input book/enroll/<PersonName> --name <PersonName>`

- Recognize from a camera or video file:
	- `python src/recognize.py --db book/db/face_db.npz --model models/embedder_arcface.onnx --source 0`

- Rebuild database from enroll directory (batch enroll):
	- `python src/enroll.py --input book/enroll --batch --model models/embedder_arcface.onnx`

## Implementation Notes

- The repository separates detection, alignment, and embedding to allow swapping detectors or embedders.
- Embeddings are stored in `book/db/` as `.npz`/`.json` for fast search and human-readable metadata.
- Use `onnxruntime` for model inference. For GPU acceleration, switch to `onnxruntime-gpu` and ensure proper CUDA/cuDNN drivers.

## Troubleshooting

- If faces are not detected reliably, try the Haar detector in `src/haar.py` or adjust detection thresholds.
- For poor recognition accuracy, check alignment and ensure enrollment images are high-quality and frontal.

## Contributing

1. Fork the repo and create a feature branch.
2. Add tests or a demonstration script if adding new functionality.
3. Open a pull request with a clear description of changes.

## Attribution & References

- ArcFace: InsightFace / ArcFace papers and official implementations inspired the embedding approach.
- This project uses an ONNX version of an ArcFace embedder for portability.

## License

Specify your preferred license here (e.g., MIT). If you want me to add a LICENSE file, tell me which license to use and I will add it.

---

If you'd like, I can also:
- Add a `requirements.txt` with pinned packages.
- Expand individual `src/` README notes with usage examples per script.
- Add a sample `examples/` directory with enrollment and recognition demos.

Updated: project README
