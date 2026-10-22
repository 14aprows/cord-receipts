import json

import numpy as np
import torch
from PIL import Image

def parse_ground_truth(ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    return ground_truth["gt_parse"]

class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        # 256 UTF-8 byte values and three special tokens.
        return 259

    def encode(self, text, add_special_tokens=False):
        tokens = [value + 3 for value in text.encode("utf-8")]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        result = bytearray()
        special_tokens = {0: b"[PAD]", 1: b"[START]", 2: b"[END]"}
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id >= 3:
                result.append(token_id - 3)
            elif not skip_special_tokens:
                result.extend(special_tokens[token_id])
        return result.decode("utf-8", errors="replace")


class ImageProcessor:
    def __init__(self, image_size=384):
        self.image_size = image_size

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("Only PyTorch tensors are supported.")
        if isinstance(images, Image.Image):
            images = [images]

        tensors = []
        for image in images:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = np.array(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(pixels).permute(2, 0, 1))

        return {"pixel_values": torch.stack(tensors)}
