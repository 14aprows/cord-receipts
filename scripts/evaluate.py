import argparse
import io
import json

import torch
from src.configs.config import DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, IMAGE_SIZE
from src.data.dataset import CordDataset
from src.data.loader import create_dataloader
from src.data.collate import create_collate_fn
from src.data.preprocessing import ByteTokenizer, ImageProcessor
from src.evaluation.metrics import calculate_text_metrics
from src.models.custom.encoder import ImageEncoder
from src.models.custom.decoder import TextDecoder
from src.models.custom.model import CordModel
from scripts.inference import predict

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_tokens = 0
    total_tokens = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=images,
            input_ids=input_ids,
            labels=labels,
        )

        total_loss += outputs["loss"].item()
        total_batches += 1

        predictions = outputs["logits"].argmax(dim=-1)
        valid_tokens = labels != -100
        correct_tokens += ((predictions == labels) & valid_tokens).sum().item()
        total_tokens += valid_tokens.sum().item()

    if total_batches == 0 or total_tokens == 0:
        raise ValueError("Evaluation data is empty.")

    return {
        "loss": total_loss / total_batches,
        "token_accuracy": correct_tokens / total_tokens,
    }

def evaluate_text(model, dataset, image_processor, tokenizer, device):
    predictions = []
    targets = []
    token_limit_count = 0

    for index in range(len(dataset)):
        record = dataset[index]
        image = record["image"]

        if isinstance(image, dict):
            if image.get("bytes") is not None:
                image = io.BytesIO(image["bytes"])
            else:
                image = image["path"]

        prediction, reached_eos = predict(
            model=model,
            image_path=image,
            image_processor=image_processor,
            tokenizer=tokenizer,
            decoder_start_token_id=model.decoder.start_token_id,
            device=device,
            max_new_tokens=model.decoder.max_length,
        )

        predictions.append(prediction)
        targets.append(
            json.dumps(record["ground_truth"]["gt_parse"], ensure_ascii=False)
        )
        token_limit_count += int(not reached_eos)

        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"Predicted {index + 1}/{len(dataset)} records")

    results = calculate_text_metrics(predictions, targets)
    results["token_limit_count"] = token_limit_count
    return results

def main(
    encoder,
    decoder,
    image_processor,
    tokenizer,
    split="validation",
    batch_size=BATCH_SIZE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CordModel(encoder, decoder).to(device)

    checkpoint_path = CHECKPOINT_DIR / "best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = CHECKPOINT_DIR / "last.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    data_files = sorted(DATA_DIR.glob(f"{split}-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No {split} files found in {DATA_DIR}")

    dataset = CordDataset(parquet_files=data_files)
    collate_fn = create_collate_fn(
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=decoder.start_token_id,
    )
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Split: {split} | Records: {len(dataset)}")
    print(f"Checkpoint: {checkpoint_path}")

    results = evaluate(model, dataloader, device)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Token accuracy: {results['token_accuracy']:.2%}")

    results.update(
        evaluate_text(model, dataset, image_processor, tokenizer, device)
    )
    print(f"CER: {results['cer']:.2%}")
    print(f"JSON valid rate: {results['json_valid_rate']:.2%}")
    print(f"JSON exact match: {results['json_exact_match']:.2%}")
    print(f"Token limit reached: {results['token_limit_count']}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the receipt model.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be at least 1.")

    checkpoint_path = CHECKPOINT_DIR / "best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = CHECKPOINT_DIR / "last.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden_size = checkpoint.get("hidden_size", 256)
    max_length = checkpoint.get("max_length", 2048)
    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    del checkpoint

    tokenizer = ByteTokenizer()
    image_processor = ImageProcessor(image_size=image_size)
    encoder = ImageEncoder(hidden_size=hidden_size)
    decoder = TextDecoder(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        start_token_id=tokenizer.bos_token_id,
        hidden_size=hidden_size,
        max_length=max_length,
    )

    main(
        encoder=encoder,
        decoder=decoder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
    )