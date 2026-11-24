"""
config.py
=========
Semua konstanta global di SATU tempat, supaya setting eksperimen
konsisten di semua fase dan mudah dilaporkan di paper.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"    # dataset mentah (l7irish/, l8biome/)
DATA_DIR = PROJECT_ROOT / "data"          # artefak fase 0 (index, stats, split)
RUNS_DIR = PROJECT_ROOT / "runs"          # hasil training (json + checkpoint)
ANALYSIS_DIR = PROJECT_ROOT / "analysis"  # tabel & statistik
FIGURES_DIR = PROJECT_ROOT / "figures"

NUM_CLASSES = 5
CLASS_NAMES = ["Fill", "Cloud Shadow", "Clear", "Thin Cloud", "Cloud"]
FILL, SHADOW, CLEAR, THIN, CLOUD = 0, 1, 2, 3, 4
EVAL_CLASSES = [SHADOW, CLEAR, THIN, CLOUD]  # Fill selalu di-ignore

DATASETS = {
    "l7irish": {
        "torchgeo_class": "L7Irish",
        "root": str(DATASET_DIR / "l7irish"),
        "in_channels": 9,
        "rgb_bands": (2, 1, 0),  # B3, B2, B1
        "pretrained_weights": "LANDSAT_ETM_TOA_MOCO",
    },
    "l8biome": {
        "torchgeo_class": "L8Biome",
        "root": str(DATASET_DIR / "l8biome"),
        "in_channels": 11,
        "rgb_bands": (3, 2, 1),  # B4, B3, B2
        "pretrained_weights": "LANDSAT_OLI_TIRS_TOA_MOCO",
    },
}

# Split 60/20/20 dengan seed TETAP.
#   "location" : scene dikelompokkan per lokasi WRS-2 (path/row); satu lokasi utuh masuk
#                ke SATU split, sehingga tidak ada lokasi sama (beda tanggal) di train & test.
#                Fraksi dihitung atas jumlah scene, per grup
#                dataset (biome L8 / zona lintang L7) supaya setiap grup ada di train/val/test
#                (data.location_split). Mode "scene" lama tidak didukung loader native.
SPLIT_MODE = "location"
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_SEED = 123
VAL_SEED = 999   # seed patch validasi per-epoch (dibuat SEKALI, lalu tetap)

# Bobot kelas hand-picked yang dipakai di SEMUA run versi lama (train.py baris 115).
# Dipertahankan sebagai loss "wce" supaya hasil baru sebanding dengan versi lama,
# dan WAJIB dilaporkan di paper.
LEGACY_CLASS_WEIGHTS = [0.0, 5.0, 1.0, 1.5, 1.0]

# Partial label untuk scene yang shadow-nya tidak dianotasi (scripts/make_partial_masks.py).
#   none : mask asli
#   all  : semua Clear di scene tanpa anotasi shadow = "Clear atau Shadow" (Francis 2024)
#   geom : hanya Clear di jalur proyeksi bayangan awan (sudut matahari dari MTL) yang ambigu
PARTIAL_MODES = ("all", "geom")
PARTIAL_DIR = DATASET_DIR / "partial"
AMBIG = 5          # label training "Clear atau Shadow" (bukan kelas output; eval -> Clear)
AMBIG_RAW = 96     # nilai di file mask turunan
SHADOW_ANNOT_MIN_PX = 1            # scene dianggap berlabel shadow bila mask berisi >= 1 piksel shadow
CLOUD_HEIGHT_RANGE_M = (200, 12000)  # rentang tinggi awan untuk proyeksi bayangan (Fmask)
GEOM_DOWNSAMPLE = 4                # proyeksi dihitung di grid 4x lebih kasar (120 m)
GEOM_BUFFER_PX = 1                 # dilasi jalur di grid kasar (1 px = 120 m)


def partial_root(dataset: str, mode: str) -> Path:
    return PARTIAL_DIR / mode / dataset


# Seed: 3 seed lama + 2 baru
SEEDS = [42, 123, 2024, 7, 2025]

# Default training (sama dengan paper versi lama, kecuali augmentasi/normalisasi)
TRAIN_DEFAULTS = {
    "epochs": 50,
    "batch_size": 8,
    "patch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patches_per_epoch": 2000,
    "val_patches": 400,
    "plateau_patience": 5,
    "early_stop_patience": 10,
    # Early stopping baru boleh menghentikan run setelah epoch ini (training minimal).
    "min_epochs": 20,
    # Checkpoint hanya dipilih dari epoch >= nilai ini; val mIoU baseline random
    # naik-turun ±10 poin di epoch awal sehingga puncak kebetulan di epoch 1 bisa terpilih.
    "select_from_epoch": 5,
}


# Kriteria pemilihan checkpoint & early stopping (SERAGAM untuk semua run).
#   "mIoU"      : val mIoU 4 kelas (default; peka terhadap shadow)
#   "shadow_f1" : val F1 kelas shadow
#   "loss"      : val loss (perilaku versi lama; buta terhadap shadow yang sangat langka)
# LR scheduler tetap memakai val loss.
SELECT_METRIC = "mIoU"


def artefact_paths(dataset: str) -> dict:
    """Lokasi artefak fase 0 untuk satu dataset."""
    return {
        "index": DATA_DIR / f"{dataset}_patch_index.pkl",
        "stats": DATA_DIR / f"{dataset}_train_stats.json",
        "split": DATA_DIR / f"{dataset}_split.json",
    }
