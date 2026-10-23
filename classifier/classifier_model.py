import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    """
    分类器 MLP

    输入:
        [B, 256, 256]

    输出:
        [B, 1]
    """

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2),
            nn.ReLU(inplace=True),

            nn.Linear(2, 1)
        )

    def forward(self, x):
        """
        x: [B,256,256]
        """

        # Figure (a): H x W -> H x 1, taking the maximum over W.
        # Keep one value per token; never pool over the patient's slice axis.
        if x.ndim != 3 or x.shape[-2:] != (256, 256):
            raise ValueError("ClassifierMLP expects [slices, 256 tokens, 256 features]")
        x = x.amax(dim=-1)

        # MLP分类
        x = self.mlp(x)

        return x
