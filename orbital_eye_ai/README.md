# ORBITAL EYE AI — starter application

A hackathon-ready **software scaffold** for longitudinal ophthalmic monitoring during spaceflight.
The current version supports OCT image upload, basic image quality/file inspection, and scan history.
It does **not** yet run an AI model or diagnose SANS.

## 1. Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## 2. Where your datasets and model go

- `data/datasets/` — place downloaded public OCT datasets here. Keep the original dataset license and citation.
- `models/` — place trained model weights here (for example `retinal_segmentation.pt`).
- `training/` — training and evaluation code belongs here.
- `uploads/` — images uploaded through the app (local prototype storage).

Do not commit private patient data or large datasets to a public repository.

## 3. Current functionality

- OCT image upload (PNG, JPG, TIFF, BMP; 20 MB maximum).
- Basic image readability check, dimensions, and mean grayscale intensity.
- Local scan history stored in SQLite.
- Clearly marked model-integration placeholder.

The basic image statistics are technical quality metadata only. They are not clinical measurements.

## 4. Model integration plan

The intended next step is to add a validated OCT segmentation/inference pipeline:

1. Confirm dataset modality, image format, labels and license.
2. Split data by patient/subject, not by image, to avoid leakage.
3. Train and evaluate a segmentation model (e.g. U-Net) using expert-annotated retinal boundaries.
4. Report metrics such as Dice score and boundary error on a held-out subject-level test set.
5. Add inference in `app.py` and render predicted boundaries in the analysis page.
6. Compare measurements with the astronaut's own pre-flight baseline.

Do not train a SANS classifier unless the dataset contains suitable, clinically meaningful labels and enough independent subjects.

## 5. Suggested project structure

```text
orbital_eye_ai/
├── app.py
├── requirements.txt
├── orbital_eye.db          # created automatically
├── app/
│   ├── templates/
│   └── static/
├── data/
│   └── datasets/           # ADD YOUR DATA HERE
├── models/                 # ADD TRAINED WEIGHTS HERE
├── training/               # ADD TRAINING CODE HERE
└── uploads/
```

## Medical and scientific limitations

This is a research/hackathon prototype, not a medical device. It must not be used to make clinical decisions.
SANS is multifactorial. OCT changes alone do not establish a diagnosis. The app currently does not estimate SANS risk, identify edema, or interpret retinal layers.
