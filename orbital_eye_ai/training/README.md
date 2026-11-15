# Model training workspace

Train the retinal-layer U-Net in **Google Colab (GPU)**. Codespaces has no GPU.

| File | Purpose |
|---|---|
| `colab_train.ipynb` | Step-by-step Colab notebook (mount Drive → inspect data → train → evaluate) |
| `config.py` | Paths, patient-level split, hyper-parameters |
| `dataset.py` | Duke DME `.mat` loader, boundary → mask conversion, augmentation, `--inspect` |
| `train.py` | Training (CE + Dice loss), validation, single held-out test evaluation, checkpoint |
| `requirements.txt` | Training dependencies |

Shared code: `orbital/ml/unet.py` (architecture) and `orbital/ml/metrics.py` (Dice, IoU, precision, recall, boundary error).

## Before training
- Download the Duke DME dataset yourself and read its terms of use. Keep it on Google Drive, **never in GitHub**.
- Run `python training/dataset.py --inspect <folder>` and confirm the keys/shapes match the docstring in `dataset.py` (`images`, `manualLayers1`, 8 boundaries). Adjust the loader if they differ.
- The split is by **subject** (6 train / 2 validation / 2 test). Never move B-scans of one subject across splits.
- The test set is evaluated **once**, after model selection on the validation set.

## Known limitations
- Duke DME is small (10 subjects, ~11 annotated B-scans each) and contains diabetic macular edema; results will not generalise to all devices or to optic-disc scans.
- Training resizes the annotated central region (~500 px wide) to 512×512; the app scales the whole B-scan to 512 px height keeping its proportions. Only the central ~2/3 of each Duke B-scan is annotated, so expect weaker boundaries near the image edges.
- Verified on the real files (Nov 2026): 10 subjects × 11 annotated B-scans = 110; variables `images` (496×768×61), `manualLayers1/2` (8×768×61). A few columns inside the annotated range miss some boundaries; the loader interpolates them.
- Report only the metrics printed by `train.py` / stored in the checkpoint. Do not quote numbers from papers as your own.
