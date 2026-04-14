import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalStructureGuidance(nn.Module):
    """
    DSM structure guides RGB features (lightweight, stable).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)

        # learnable blur (depthwise) for high-pass structure = dsm - blur(dsm)
        self.dsm_blur = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        # refine structure
        self.structure_refine = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # conditional attention: use both rgb & structure
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        # high-pass structure
        blur = self.dsm_blur(dsm)
        structure = dsm - blur
        structure = self.structure_refine(structure)

        # conditional attention
        a = self.attn(torch.cat([rgb, structure], dim=1))

        # residual guidance
        return rgb + rgb * a