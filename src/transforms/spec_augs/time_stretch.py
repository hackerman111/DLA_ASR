import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, min_rate: float = 1.0, max_rate: float = 1.15, p: float = 0.5):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def forward(self, data: Tensor) -> Tensor:
        if not self.training or data.numel() == 0 or torch.rand(()).item() >= self.p:
            return data
        rate = torch.empty(()).uniform_(self.min_rate, self.max_rate).item()
        frames = max(1, round(data.shape[-1] / rate))
        result = F.interpolate(
            data.reshape(-1, 1, data.shape[-1]),
            size=frames,
            mode="linear",
            align_corners=False,
        )
        return result.reshape(*data.shape[:-1], frames)
