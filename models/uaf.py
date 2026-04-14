import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwareFusion(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, temperature: float = 1.5):
        super().__init__()
        mid = max(8, channels // reduction)
        self.temperature = temperature

        def conf_head():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),  # depthwise
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, mid, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            )

        self.rgb_conf = conf_head()
        self.dsm_conf = conf_head()

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        rgb_score = self.rgb_conf(rgb)  # [B,1,H,W]
        dsm_score = self.dsm_conf(dsm)  # [B,1,H,W]

        scores = torch.cat([rgb_score, dsm_score], dim=1)  # [B,2,H,W]
        weights = F.softmax(scores / self.temperature, dim=1)

        w_rgb = weights[:, 0:1, :, :]
        w_dsm = weights[:, 1:2, :, :]

        fusion = w_rgb * rgb + w_dsm * dsm
        return fusion