from pathlib import Path  

ROOT_DIR = Path(__file__).resolve().parents[2]
DATAROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = DATAROOT_DIR / "cordrcv2/data"

print(DATA_DIR)
OUTPUT_DIR = ROOT_DIR / "outputs"

if Path("/kaggle/working").is_dir():
    OUTPUT_DIR = Path("/kaggle/working/cord-receipts")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"

SEED = 42

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5

IMAGE_SIZE = 384