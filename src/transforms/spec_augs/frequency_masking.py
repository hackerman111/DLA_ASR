import torch
import torchaudio
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(
        self, freq_mask_param: int = 12, p: float = 0.5, iid_masks: bool = False
    ):
        super().__init__()
        self.p = p
        self._aug = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param, iid_masks=iid_masks
        )

    def forward(self, data: Tensor) -> Tensor:
        if not self.training or data.numel() == 0 or torch.rand(()).item() >= self.p:
            return data
        return self._aug(data)
