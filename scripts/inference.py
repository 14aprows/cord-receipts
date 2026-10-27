import torch
from PIL import Image

from src.configs.config import CHECKPOINT_DIR
from src.models.custom.model import CordModel

@torch.no_grad()
def predict(
    model,
    image_path,
    image_processor,
    tokenizer,
    decoder_start_token_id,
    device,
    max_new_tokens=512
):
    model.eval()
    if decoder_start_token_id is None:
        raise ValueError("Decoder start token ID is missing.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS token ID is missing.")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        pixel_values = image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(device)

    input_ids = torch.tensor(
        [[decoder_start_token_id]],
        dtype=torch.long,
        device=device
    )

    reached_eos = False

    for _ in range(max_new_tokens):
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids
        )

        next_token = outputs["logits"][:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            reached_eos = True
            break

    text = tokenizer.decode(
        input_ids[0, 1:].tolist(),
        skip_special_tokens=True
    )

    return text, reached_eos

def main(
    encoder,
    decoder,
    image_processor,
    tokenizer,
    image_path
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

    text, reached_eos = predict(
        model=model,
        image_path=image_path,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=decoder.start_token_id,
        device=device,
    )

    print(text)
    if not reached_eos:
        print("Token limit reached. The result may be incomplete.")

    return text
