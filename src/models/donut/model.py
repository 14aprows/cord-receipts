import torch.nn as nn
from transformers import DonutProcessor, VisionEncoderDecoderModel

class DonutModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "naver-clova-ix/donut-base"
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.network = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.network.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.network.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.network.config.eos_token_id = self.processor.tokenizer.eos_token_id

        # memory
        self.network.encoder.gradient_checkpointing_enable()
        self.network.decoder.gradient_checkpointing_enable()
        self.network.decoder.config.use_cache = False

    def forward(
        self,
        pixel_values,
        input_ids=None,
        labels=None
    ):
        if input_ids is None:
            raise ValueError("Input token IDs are missing")
        if input_ids.size(1) > self.network.decoder.config.max_position_embeddings:
            raise ValueError("The text exceeds the decoder limit.")

        outputs = self.network(
            pixel_values = pixel_values,
            decoder_input_ids = input_ids,
            decoder_attention_mask = input_ids.ne(self.network.config.pad_token_id).long(),
            labels=labels
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }
