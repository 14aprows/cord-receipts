import argparse

import torch

from src.configs.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    IMAGE_SIZE,
    SEED
)

from src.data.dataset import CordDataset
from src.data.loader import create_dataloader
from src.data.collate import create_collate_fn
from src.data.preprocessing import ByteTokenizer, ImageProcessor
from src.models.encoder import ImageEncoder
from src.models.decoder import TextDecoder
from src.models.model import CordModel
from src.training.trainer import train_one_epoch

def main(
    encoder, 
    decoder, 
    image_processor, 
    tokenizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    smoke_test=False,
):
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    collate_fn = create_collate_fn(
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=decoder.start_token_id
    )

    train_files = sorted(DATA_DIR.glob("train-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No training files found in {DATA_DIR}")

    train_dataset = CordDataset(parquet_files=train_files)
    print(f"Train samples: {len(train_dataset)}", flush=True)
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=not smoke_test,
        collate_fn=collate_fn
    )

    model = CordModel(
        encoder=encoder,
        decoder=decoder
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if smoke_test:
        print("Testing one batch...", flush=True)
        batch = next(iter(train_loader))
        loss = train_one_epoch(model, [batch], optimizer, device)
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError("The loss is not finite.")
        print(f"Test passed | Loss: {loss:.4f}", flush=True)
        return model

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}", flush=True)
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f}", flush=True)
        torch.save(
            {
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "tokenizer_type": "utf8_bytes_v1",
                "hidden_size": decoder.token_embedding.embedding_dim,
                "max_length": decoder.max_length,
                "image_size": image_processor.image_size,
            },
            CHECKPOINT_DIR / "last.pt"
        )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the receipt model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--smoke-test", action="store_true")
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
        smoke_test=args.smoke_test,
    )
