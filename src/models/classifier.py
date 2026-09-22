import torch
import torch.nn as nn


class FineTuneClassifier(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dim=384,
        num_classes=2
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = self.encoder(x)

        # Mean-pool the patch tokens
        features = features.mean(dim=1)

        logits = self.classifier(features)

        return logits