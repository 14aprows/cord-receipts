import torch 
import torch.nn as nn

class CordModel(nn.Module):
    def __init__(
        self, 
        encoder,
        decoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        pixel_values,
        input_ids=None,
        labels=None
    ):
        image_features = self.encoder(pixel_values)
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=image_features,
            labels=labels
        )
        return outputs