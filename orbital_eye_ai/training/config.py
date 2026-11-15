"""Training configuration. Edit paths for your Google Drive layout."""
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # Folder with the Duke DME .mat files (Subject_01.mat ... Subject_10.mat)
    data_dir: str = "/content/drive/MyDrive/orbital_eye/datasets/duke_dme"
    out_dir: str = "/content/drive/MyDrive/orbital_eye/checkpoints"

    # Patient-level split (subject numbers). No subject appears in two splits.
    train_subjects: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    val_subjects: List[int] = field(default_factory=lambda: [7, 8])
    test_subjects: List[int] = field(default_factory=lambda: [9, 10])
    annotator: int = 1                 # manualLayers1 or manualLayers2

    num_classes: int = 9               # above ILM, 7 layers, below Bruch's membrane
    input_height: int = 512
    input_width: int = 512
    base: int = 32
    depth: int = 4

    epochs: int = 80
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 1.0
    seed: int = 42
    num_workers: int = 2
    version: str = "0.1.0"

    def to_dict(self):
        return asdict(self)
