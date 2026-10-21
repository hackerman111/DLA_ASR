import torch
from torch import nn

from src.model.conformer_modules import ConformerBlock, ConvSubsampling


class Conformer(nn.Module):
    """Доктестик.
    >>> model = Conformer(8, 6, d_model=8, num_layers=1, num_heads=2,
    ...                   ffn_dim=16, conv_kernel_size=3, dropout=0.0)
    >>> result = model(torch.rand(2, 8, 17), torch.tensor([17, 13]))
    >>> result["log_probs"].shape
    torch.Size([2, 5, 6])
    >>> result["log_probs_length"].tolist()
    [5, 4]
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        d_model=256,
        num_layers=8,
        num_heads=4,
        ffn_dim=1024,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.subsampling = ConvSubsampling(n_feats, d_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, num_heads, ffn_dim, conv_kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        if spectrogram.ndim != 3 or spectrogram.shape[1] != self.n_feats:
            raise ValueError("Ожидается спектрограмма размерности [B, n_feats, T]")
        if (
            spectrogram_length.shape != (spectrogram.shape[0],)
            or spectrogram_length.dtype not in (torch.int32, torch.int64)
            or (spectrogram_length <= 0).any()
            or (spectrogram_length > spectrogram.shape[-1]).any()
        ):
            raise ValueError(
                "Длины должны быть целыми числами от 1 до T для каждого примера"
            )

        lengths = spectrogram_length.to(spectrogram.device)
        mask = (
            torch.arange(spectrogram.shape[-1], device=spectrogram.device)[None, :]
            >= lengths[:, None]
        )
        x = spectrogram.float().clamp_min(1e-5).log().masked_fill(mask[:, None, :], 0)
        count = lengths[:, None, None]
        mean = x.sum(-1, keepdim=True) / count
        x = (x - mean).masked_fill(mask[:, None, :], 0)
        x = x / (x.square().sum(-1, keepdim=True) / count + 1e-5).sqrt()

        x, output_lengths = self.subsampling(x, spectrogram_length)
        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            >= output_lengths.to(x.device)[:, None]
        )
        for block in self.blocks:
            x = block(x, mask)
        log_probs = nn.functional.log_softmax(
            self.output(x), dim=-1, dtype=torch.float32
        )
        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        return self.subsampling.transform_input_lengths(input_lengths)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            super().__str__()
            + f"\nВсего параметров: {all_parameters}"
            + f"\nОбучаемых параметров: {trainable_parameters}"
        )
