import torch 
import torch.nn as nn
import torch.nn.functional as F

class TextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        start_token_id,
        hidden_size=256,
        max_length=1024
    ):
        super().__init__()
        if pad_token_id is None:
            raise ValueError("pad token id is missing")
        if start_token_id is None:
            raise ValueError("start token id is missing")

        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.max_length = max_length 

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_size
        )

        layers = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer=layers,
            num_layers=3
        )

        self.output_layers = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

    def forward(
        self, 
        input_ids, 
        encoder_hidden_states,
        labels=None
    ):
        if input_ids is None:
            raise ValueError("input ids is missing")
        
        length = input_ids.size(1)
        if length == 0:
            raise ValueError("The text is empty")
        if length > self.max_length:
            raise ValueError("The text is too long")

        position = torch.arange(
            length,
            device=input_ids.device
        ).unsqueeze(0)

        embeddings = self.token_embedding(input_ids) + self.position_embedding(position)

        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=input_ids.device
            ),
            diagonal=1
        )

        padding_mask = input_ids.eq(self.pad_token_id)

        hidden_states = self.transformer(
            tgt=embeddings,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )

        logits = self.output_layers(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
        }