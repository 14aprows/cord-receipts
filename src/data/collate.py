import io
import json

import torch 
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

def create_collate_fn(
    image_processor,
    tokenizer,
    decoder_start_token_id
):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad token ID is missing.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS token ID is missing.")
    if decoder_start_token_id is None:
        raise ValueError("Decoder start token ID is missing.")

    def collate_fn(samples):
        if not samples:
            raise ValueError("The batch is empty.")

        images = []
        targets = []

        for sample in samples:
            image = sample["image"]
            if isinstance(image, dict):
                if image.get("bytes") is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])
            elif not isinstance(image, Image.Image):
                image = Image.open(image)

            images.append(image.convert("RGB"))

            target_next = json.dumps(
                sample["ground_truth"]["gt_parse"],
                ensure_ascii=False
            )

            token_ids = tokenizer.encode(
                target_next,
                add_special_tokens=False
            )

            token_ids.append(tokenizer.eos_token_id)
            targets.append(
                torch.tensor(token_ids, dtype=torch.long)
            )

        pixel_values = image_processor(
            images=images,
            return_tensors="pt"
        )["pixel_values"]

        labels = pad_sequence(
            targets,
            batch_first=True,
            padding_value=-100
        )

        input_ids = torch.full_like(
            labels,
            tokenizer.pad_token_id,
        )

        input_ids[:, 0] = decoder_start_token_id
        input_ids[:, 1:] = labels[:, :-1]

        input_ids.masked_fill_(
            input_ids == -100,
            tokenizer.pad_token_id,
        )

        return {
            "image": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

    return collate_fn
