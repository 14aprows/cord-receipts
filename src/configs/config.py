from pathlib import Path  

ROOT_DIR = Path(__file__).resolve().parents[2]
DATAROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = DATAROOT_DIR / "cordrcv2/data"

print(DATA_DIR)
OUTPUT_DIR = ROOT_DIR / "outputs"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
PREDICTION_DIR = ROOT_DIR / "predictions"
METRICS_DIR = ROOT_DIR / "metrics"

SEED = 42

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5

IMAGE_SIZE = 384