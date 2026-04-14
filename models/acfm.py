import torch
import torch.nn as nn
import torch_dct as DCT

class AdaptiveCrossFrequencyModule(nn.Module):
    """
    Stage-4 recommended (small H,W) due to O(N^2) attention.
    Frequency gating:
      - compute DCT
      - split into low/high frequency by radius mask
      - use two scalars (g_low, g_high) from global pooled features to weight low/high
    """
    def __init__(self, channels: int, low_radius: float = 0.35):
        super().__init__()
        self.channels = channels
        self.low_radius = low_radius  # normalized radius threshold in [0,1]

        self.rgb_proj = nn.Conv2d(channels, channels, 1)
        self.dsm_proj = nn.Conv2d(channels, channels, 1)

        # low/high frequency gates (2 scalars per-sample)
        mid = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1),  # -> (g_low, g_high)
        )

        # cross attention
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # spatial refinement
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1)

        # channel attention
        ca_mid = max(8, channels // 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, ca_mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ca_mid, channels, 1),
            nn.Sigmoid()
        )

        # cache for low-frequency mask (registered buffers should be tensors)
        self.register_buffer("_cached_low_mask", torch.empty(0), persistent=False)
        self._cached_hw = None  # (H,W)

    def _build_low_mask(self, H: int, W: int, device, dtype):
        # DCT low freq is near (0,0). We'll use normalized radius from (0,0).
        yy = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        xx = torch.arange(W, device=device, dtype=dtype).view(1, W)
        rr = torch.sqrt((yy / max(H - 1, 1)) ** 2 + (xx / max(W - 1, 1)) ** 2)  # [H,W] in [0,~1.4]
        rr = rr / rr.max().clamp_min(1e-6)  # normalize to [0,1]

        low_mask = (rr <= self.low_radius).to(dtype=dtype)  # [H,W]
        low_mask = low_mask.view(1, 1, H, W)                # broadcastable
        return low_mask

    def freq_transform(self, x: torch.Tensor):
        """
        Return gated spatial feature.
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # (re)build cached mask if needed
        if self._cached_hw != (H, W) or self._cached_low_mask.numel() == 0:
            self._cached_low_mask = self._build_low_mask(H, W, device=device, dtype=dtype)
            self._cached_hw = (H, W)

        low_mask = self._cached_low_mask  # [1,1,H,W]
        high_mask = 1.0 - low_mask

        # gates per sample: [B,2,1,1]
        g = torch.sigmoid(self.gate(x))
        g_low = g[:, 0:1, :, :]   # [B,1,1,1]
        g_high = g[:, 1:2, :, :]  # [B,1,1,1]

        freq = DCT.dct_2d(x, norm='ortho')

        freq_low = freq * low_mask
        freq_high = freq * high_mask
        freq = g_low * freq_low + g_high * freq_high

        out = DCT.idct_2d(freq, norm='ortho')
        return out

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor):
        rgb = self.rgb_proj(rgb)
        dsm = self.dsm_proj(dsm)
        rgb_freq = self.freq_transform(rgb)
        dsm_freq = self.freq_transform(dsm)

        # cross attention (token attention -> O((HW)^2), so use stage-4)
        q = self.query(rgb_freq)
        k = self.key(dsm_freq)
        v = self.value(dsm_freq)

        B, C, H, W = q.shape
        q = q.view(B, C, -1)  # [B,C,N]
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        attn = torch.softmax((q.transpose(1, 2) @ k) / (C ** 0.5), dim=-1)  # [B,N,N]
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).contiguous()       # [B,C,N]
        out = out.view(B, C, H, W)

        spatial = self.spatial(out)
        weight = self.channel_attn(spatial)
        out = spatial * weight

        return rgb + out