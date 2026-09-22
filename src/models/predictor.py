import torch
import torch.nn as nn


class IJEPA_Predictor(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 196, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.predictor_head = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(self, x):

        x = x + self.pos_embed

        x = self.blocks(x)

        x = self.norm(x)

        x = self.predictor_head(x)

        return x