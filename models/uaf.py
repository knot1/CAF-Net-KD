import torch
import torch.nn as nn
import torch.nn.functional as F

EQUAL_WEIGHT = 0.5
CONFIDENCE_SCALE = 2.0


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

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor, return_conf: bool = False):
        rgb_score = self.rgb_conf(rgb)  # [B,1,H,W]
        dsm_score = self.dsm_conf(dsm)  # [B,1,H,W]

        scores = torch.cat([rgb_score, dsm_score], dim=1)  # [B,2,H,W]
        weights = F.softmax(scores / self.temperature, dim=1)

        w_rgb = weights[:, 0:1, :, :]
        w_dsm = weights[:, 1:2, :, :]

        fusion = w_rgb * rgb + w_dsm * dsm
        if return_conf:
            # Use max modality weight as confidence: weights sum to 1, so max=0.5 means equal preference
            # (maximum uncertainty) while max=1.0 means full preference for one modality (high confidence).
            # Map max weight from [0.5, 1.0] -> [0, 1] so 0.5 (equal weights) becomes 0.0 confidence.
            conf = (weights.max(dim=1).values - EQUAL_WEIGHT) * CONFIDENCE_SCALE
            # 0.5 -> 0.0 (uncertain), 1.0 -> 1.0 (confident)
            conf = conf.clamp(0.0, 1.0)
            return fusion, conf
        return fusion
