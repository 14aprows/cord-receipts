import argparse

import torch

from src.configs.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
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
from src.models.encoder import ImageEncoder
from src.models.decoder import TextDecoder
from src.models.model import CordModel
from src.training.trainer import train_one_epoch
from scripts.evaluate import evaluate

def main(
    encoder, 
    decoder, 
    image_processor, 
    tokenizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    collate_fn = create_collate_fn(
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=decoder.start_token_id
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

    model = CordModel(
        encoder=encoder,
        decoder=decoder
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    stopping_best = float("inf")
    bad_epochs = 0
    for epoch in range(epochs):
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
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train loss: {train_loss:.4f} "
            f"| Val loss: {val_loss:.4f} "
            f"| Val token accuracy: {token_accuracy:.2%}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "val_loss": val_loss,
            "tokenizer_type": "utf8_bytes_v1",
            "hidden_size": decoder.token_embedding.embedding_dim,
            "max_length": decoder.max_length,
            "image_size": image_processor.image_size,
        }

        torch.save(checkpoint, CHECKPOINT_DIR / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
            print("Best model saved.")

        if val_loss < stopping_best - MIN_DELTA:
            stopping_best = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the receipt model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be at least 1.")

    torch.manual_seed(SEED)
    tokenizer = ByteTokenizer()
    image_processor = ImageProcessor(image_size=IMAGE_SIZE)
    encoder = ImageEncoder(hidden_size=256)
    decoder = TextDecoder(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        start_token_id=tokenizer.bos_token_id,
        hidden_size=256,
        max_length=2048,
    )

    main(
        encoder=encoder,
        decoder=decoder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )