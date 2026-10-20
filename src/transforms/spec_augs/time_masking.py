import torch
import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(
        self,
        time_mask_param: int = 20,
        p: float = 0.5,
        iid_masks: bool = False,
        max_mask_fraction: float = 0.1,
    ):
        super().__init__()
        self.p = p
        self._aug = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param,
            iid_masks=iid_masks,
            p=max_mask_fraction,
        )

    def forward(self, data: Tensor) -> Tensor:
        if not self.training or data.numel() == 0 or torch.rand(()).item() >= self.p:
            return data
        return self._aug(data)
