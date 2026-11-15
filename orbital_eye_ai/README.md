# ORBITAL EYE AI

**AI-assisted OCT monitoring prototype for astronaut eye health** — NASA Space Apps Challenge 2026,
challenge *"Create Health Monitoring Software for Astronauts on Space Missions"*.

> **Research prototype. Not a medical device. Does not diagnose Spaceflight-Associated Neuro-ocular Syndrome (SANS).**
> All astronaut profiles and demo scans in this repository are fictional / synthetic.

## What it does

| Area | Status |
|---|---|
| Astronaut profiles (anonymised IDs, mission, status) | ✅ working |
| OCT upload linked to astronaut, exam date, phase (baseline / in-flight / post-flight), eye | ✅ working |
| Validation: extension **and** real file format, size, corrupt files, dimensions, decompression bombs | ✅ working |
| 8-bit, 16-bit and multi-page TIFF, RGB → grayscale | ✅ working |
| Technical quality report (format, bit depth, intensity stats, contrast, saturation, noise, status) | ✅ working |
| Preprocessing (percentile contrast stretch + 3×3 median; CLAHE deliberately not default) | ✅ working |
| Boundary detection → ILM–RPE thickness in **pixels** + overlay | ✅ heuristic (non-AI) fallback |
| U-Net retinal layer segmentation (PyTorch) | 🟡 code complete, **untrained** — train in Colab |
| Longitudinal comparison with baseline, Chart.js trends, side-by-side compare | ✅ working |
| Medical-review alerts with configurable **demonstration** threshold | ✅ working |
| AI model status page (honest: shows "untrained" until a real checkpoint exists) | ✅ working |
| Unit tests (24) | ✅ passing |

## Run it (GitHub Codespaces or locally)

```bash
cd orbital_eye_ai            # the app lives in this sub-folder
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask seed-demo              # optional: 4 fictional astronauts with synthetic OCT scans
python app.py                # or: flask run
```

Open http://127.0.0.1:5000. In Codespaces the port is forwarded automatically (Ports tab).
If the preview does not open, run `HOST=0.0.0.0 python app.py`.

Other commands: `flask init-db`, `flask remove-demo`, `pytest -q`.

Configuration is via environment variables (see `.env.example`): `SECRET_KEY`, `FLASK_DEBUG`, `HOST`, `PORT`,
`MAX_UPLOAD_MB`, `ALERT_THRESHOLD_PCT`, `MODEL_PATH`. Never commit `.env`.

## Project structure

```text
orbital_eye_ai/
├── app.py                  Flask routes + CLI commands (create_app factory)
├── orbital/
│   ├── config.py           settings from environment variables
│   ├── db.py               SQLite schema (astronauts, scans, alerts) + automatic upgrade of old DBs
│   ├── imaging.py          validation, 16-bit handling, preprocessing, quality report
│   ├── boundaries.py       boundary detection (U-Net or heuristic) + pixel thickness + overlay
│   ├── pipeline.py         upload → quality → preprocess → segment → measure
│   ├── monitoring.py       baseline comparison, timelines, alerts
│   ├── demo.py             fictional astronauts + synthetic OCT generator
│   └── ml/                 unet.py (PyTorch U-Net), inference.py, metrics.py (Dice, IoU, …)
├── app/templates, app/static   UI (Chart.js from cdnjs)
├── training/               Colab notebook, Duke DME loader, train/evaluate script
├── tests/                  pytest suite
├── data/datasets/          local datasets only (git-ignored)
├── models/                 model checkpoints only (git-ignored)
└── uploads/                uploaded images + derived previews (git-ignored, not public static)
```

## How measurements work (and their limits)

1. **Quality metrics** (dimensions, intensity, contrast…) are technical image properties, never clinical values.
2. **Thickness** is the vertical distance in **pixels** between the ILM and the outer retinal boundary, per image column.
   Pixels are not micrometres: conversion needs the scanner's axial pixel spacing, which uploaded PNG/JPEG files do not carry.
3. Until a trained U-Net checkpoint is connected, boundaries come from a **classical intensity heuristic** (first strong
   reflectivity rise = ILM, brightest outer band = RPE). It works on clean macular B-scans and on the synthetic demo data,
   but it is **not validated** and fails on pathology, shadows and artefacts. Every result is labelled with its method.
4. Heuristic and U-Net measurements are never compared with each other; comparisons also require the same eye.
5. **Alerts** fire only when a usable measurement, a usable baseline and a threshold breach all exist. The default
   threshold (±8 %) is **illustrative only**. An alert means "recommend ophthalmological review" — not SANS.

## Training the U-Net

See `training/README.md` and `training/colab_train.ipynb`. In short: put the Duke DME `.mat` files on Google Drive,
run the notebook on a Colab GPU, download `retinal_unet.pt` into `models/`, `pip install -r requirements-ml.txt`, restart.
Test-set metrics stored in the checkpoint then appear on the *AI model status* page. **No model has been trained yet,
so no performance figures exist.**

## Scientific background and data sources

The app's **NASA research** page (`/nasa`) summarises the NASA evidence base (SANS findings, ISS Ocular Health OCT
study, OSDR OSD-816, LSDA access path), maps it to the prototype's design, and lists exactly which data were used.

- **SANS** – optic disc edema, globe flattening, choroidal folds and peripapillary retinal thickening reported in
  long-duration crew members (Mader et al., *Ophthalmology* 2011; Lee et al., *npj Microgravity* 2020).
  Most SANS OCT findings concern the **optic nerve head**, while public annotated datasets are mostly **macular** scans.
  This prototype therefore demonstrates the *pipeline* (segmentation → thickness → change vs. baseline); applying it to
  astronaut optic-disc OCT would need appropriate data and validation.
- **Duke DME OCT dataset** – Chiu et al., "Kernel regression based segmentation of optical coherence tomography images
  with diabetic macular edema", *Biomedical Optics Express* 6(4), 2015. Manual layer boundaries by two experts.
  Check the dataset's terms before use; do not redistribute it.
- **NASA Open Science Data Repository (OSDR)** – https://osdr.nasa.gov/bio/ . *OSD-816* is an experimental human eye
  tissue model related to SANS mechanisms; it is **not** an astronaut clinical OCT dataset and is used here only as
  scientific background.
- **Space Omics and Medical Atlas (SOMA)** – https://soma.weill.cornell.edu/ (background on spaceflight biology).
- **U-Net** – Ronneberger, Fischer, Brox, MICCAI 2015. **ReLayNet** – Roy et al., *Biomed. Opt. Express* 2017
  (OCT layer segmentation on Duke DME).

Data categories are kept separate: experimental research (OSDR), clinical OCT datasets (Duke), synthetic demo data
(generated in `orbital/demo.py`), and real astronaut observations (published literature only — none in this app).

## Public demo (Hugging Face Spaces)

Codespaces is for development, not hosting. The public demo runs on **Hugging Face Spaces** (Docker, free CPU)
in **public demo mode** (`PUBLIC_DEMO=1`):

- demo astronauts are seeded at start-up;
- visitors can browse everything and try an upload — the result is shown only to them, nothing is saved,
  the original file is deleted immediately and derived images after `DEMO_UPLOAD_TTL_MIN` (30 min);
- deleting examinations and creating profiles are disabled;
- the trained weights are downloaded at start-up from a private link stored as the Space secret `MODEL_URL`
  (e.g. a Google Drive link shared as "Anyone with the link"), so they never enter the public repository.

Publish or update the Space (run in `orbital_eye_ai/`):

```bash
pip install huggingface_hub
python deploy/push_to_hf.py --space YOUR_HF_USERNAME/orbital-eye-ai
```

The script asks for a Hugging Face write token and the model link (both hidden), stores them as Space secrets and
uploads the code. Every push to GitHub also builds and smoke-tests the same Docker image in GitHub Actions.
