import argparse
import time
import torch

from src.configs.config import (
    DATA_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    IMAGE_SIZE,
    SEED,
    PATIENCE,
    MIN_DELTA
)

from src.data.dataset import CordDataset
from src.data.loader import create_dataloader
from src.data.collate import create_collate_fn
from src.data.preprocessing import ByteTokenizer, ImageProcessor
from src.models.custom.encoder import ImageEncoder
from src.models.custom.decoder import TextDecoder
from src.models.custom.model import CordModel
from src.models.donut.model import DonutModel
from src.training.trainer import train_one_epoch
from scripts.evaluate import evaluate
from src.configs.config import OUTPUT_DIR
from src.utils.csv_logger import save_csv
from src.utils.visualization import plot_training

def main(
    model,
    model_name,
    image_processor, 
    start_token_id,
    tokenizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    print(f"Model: {model_name}") 
    print(f"Device: {device}")

    collate_fn = create_collate_fn(
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=start_token_id
    )

    train_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No training files found in {DATA_DIR}")

    train_dataset = CordDataset(parquet_files=train_files)
    print(f"Train records: {len(train_dataset)}")
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_files = sorted(DATA_DIR.glob("validation-*.parquet"))
    if not val_files:
        raise FileNotFoundError("Validation files were not found")

    val_dataset = CordDataset(parquet_files=val_files)
    print(f"Validation records: {len(val_dataset)}")
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    output_dir = OUTPUT_DIR / model_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_history.csv"
    figure_path = output_dir / "figures" / "training.png"
    
    history = []
    best_val_loss = float("inf")
    stopping_best = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        started_at = time.perf_counter()

        print(f"Starting epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        results = evaluate(model, val_loader, device)
        val_loss = results["loss"]
        token_accuracy = results["token_accuracy"]
        
        if not torch.isfinite(torch.tensor(val_loss)):
            raise ValueError("Validation loss is not finite.")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_token_accuracy": token_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started_at
        })
        save_csv(history, csv_path)
        plot_training(csv_path, figure_path)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train loss: {train_loss:.4f} "
            f"| Val loss: {val_loss:.4f} "
            f"| Val token accuracy: {token_accuracy:.2%}"
        )

        checkpoint = {
            "model_name": model_name,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "val_loss": val_loss
        }

        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print("Best model saved.")

        if val_loss < stopping_best - MIN_DELTA:
            stopping_best = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    print(f"CSV saved to: {csv_path}")
    print(f"Figure saved to: {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["custom", "donut"],
        required=True
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    args = parser.parse_args()

    torch.manual_seed(SEED)

    if args.model == "custom":
        tokenizer = ByteTokenizer()
        image_processor = ImageProcessor(
            image_size=IMAGE_SIZE
        )

        encoder = ImageEncoder(hidden_size=256)

        decoder = TextDecoder(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            start_token_id=tokenizer.bos_token_id,
            hidden_size=256,
            max_length=2048
        )

        model = CordModel(
            encoder=encoder,
            decoder=decoder
        )

        start_token_id = decoder.start_token_id
    else:
        model = DonutModel()
        tokenizer = model.processor.tokenizer
        image_processor = model.processor.image_processor
        start_token_id = model.network.config.decoder_start_token_id

        model.processor.save_pretrained(OUTPUT_DIR / "donut" / "processor")

    main(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        start_token_id=start_token_id,
        model_name=args.model,
        epochs=args.epochs,
    )