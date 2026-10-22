import torch 

from src.configs.config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE
)
from src.data.dataset import CordDataset
from src.data.loader import create_dataloader
from src.data.collate import create_collate_fn
from src.models.model import CordModel

@torch.no_grad()
def evaluate(
    model, 
    dataloader, 
    device
):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch.get("input_ids")
        if input_ids is not None:
            input_ids = input_ids.to(device)

        outputs = model(
            pixel_values = batch["image"].to(device),
            input_ids = input_ids,
            labels = batch["labels"].to(device)
        )

        total_loss += outputs["loss"].item()

    return total_loss / len(dataloader)

def main(
    encoder, 
    decoder, 
    image_procesor,
    tokenizer
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = CordModel(encoder, decoder).to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "last.pt",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    collate_fn = create_collate_fn(
        image_processor=image_procesor,
        tokenizer=tokenizer,
        decoder_start_token_id=decoder.start_token_id
    )

    val_files = sorted(DATA_DIR.glob("validation-*.parquet"))

    dataset = CordDataset(parquet_files=val_files)
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    loss = evaluate(model, dataloader, device)
    print(f"Validation loss: {loss:.4f}")
    return loss
    