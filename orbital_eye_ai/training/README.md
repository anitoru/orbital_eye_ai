# Model training workspace

Add the dataset-specific preprocessing, training and evaluation scripts here.

Before training:
- Confirm whether images are OCT B-scans, fundus photographs, or another modality.
- Confirm annotation format and class definitions.
- Keep all scans from the same subject in only one data split.
- Record preprocessing steps and model version.
- Evaluate on held-out subjects and report appropriate metrics.

A U-Net is a candidate for retinal layer segmentation only if pixel-level boundary annotations are available.
