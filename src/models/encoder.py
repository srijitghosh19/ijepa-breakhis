import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16,
                 in_channels=3, embed_dim=384):

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        # (B, C, H, W)
        # -> (B, N, C)

        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x


class VisionTransformerEncoder(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6
    ):

        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
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

    def forward(self, x):

        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = self.blocks(x)

        x = self.norm(x)

        return x