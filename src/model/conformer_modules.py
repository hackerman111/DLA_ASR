import math

import torch
from torch import nn


class ConvSubsampling(nn.Module):
    """Доктестик
    >>> module = ConvSubsampling(n_feats=7, d_model=8)
    >>> output, lengths = module(torch.rand(2, 7, 17), torch.tensor([17, 13]))
    >>> output.shape
    torch.Size([2, 5, 8])
    >>> lengths.tolist()
    [5, 4]
    """

    def __init__(self, n_feats, d_model):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.projection = nn.Linear(d_model * ((n_feats + 3) // 4), d_model)

    def forward(self, spectrogram, lengths):
        x = spectrogram.transpose(1, 2).unsqueeze(1)
        for conv in self.convs:
            mask = (
                torch.arange(x.shape[2], device=x.device)[None, :]
                >= lengths.to(x.device)[:, None]
            )
            x = x.masked_fill(mask[:, None, :, None], 0)
            x = nn.functional.relu(conv(x))
            lengths = (lengths + 1) // 2
        x = self.projection(x.transpose(1, 2).flatten(2))
        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            >= lengths.to(x.device)[:, None]
        )
        return x.masked_fill(mask.unsqueeze(-1), 0), lengths

    @staticmethod
    def transform_input_lengths(lengths):
        """
        >>> lengths = torch.tensor([1, 4, 5, 17])
        >>> ConvSubsampling.transform_input_lengths(lengths).tolist()
        [1, 1, 2, 5]
        """
        return (lengths + 3) // 4


class FeedForwardModule(nn.Module):
    """
    >>> module = FeedForwardModule(d_model=8, ffn_dim=16, dropout=0.0)
    >>> module(torch.rand(2, 9, 8)).shape
    torch.Size([2, 9, 8])
    """

    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RelativeMultiHeadAttention(nn.Module):
    """
    >>> module = RelativeMultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    >>> mask = torch.tensor([[False, False, True], [False, False, False]])
    >>> output = module(torch.rand(2, 3, 8), mask)
    >>> output.shape
    torch.Size([2, 3, 8])
    >>> output[0, 2].count_nonzero().item()
    0
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model должен делиться на положительное num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.position_projection = nn.Linear(d_model, d_model, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "inv_freq",
            torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)),
            persistent=False,
        )

    def forward(self, x, padding_mask=None):
        batch_size, time_steps, d_model = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(batch_size, time_steps, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        positions = torch.arange(1 - time_steps, time_steps, device=x.device)
        angles = positions[:, None] * self.inv_freq
        encoding = x.new_zeros(2 * time_steps - 1, d_model)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, : d_model // 2].cos()
        relative = (
            self.position_projection(encoding)
            .reshape(2 * time_steps - 1, self.num_heads, self.head_dim)
            .permute(1, 2, 0)
        )
        indices = torch.arange(time_steps, device=x.device)
        offsets = indices[:, None] - indices[None, :] + time_steps - 1
        with torch.autocast(device_type=x.device.type, enabled=False):
            content = (
                q.float() + self.content_bias[:, None].float()
            ) @ k.float().transpose(-1, -2)
            position = (
                q.float() + self.position_bias[:, None].float()
            ) @ relative.float()
            position = position.gather(
                -1, offsets.expand(batch_size, self.num_heads, -1, -1)
            )
            scores = (content + position) / math.sqrt(self.head_dim)
            if padding_mask is not None:
                scores = scores.masked_fill(
                    padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
                )
            weights = scores.softmax(dim=-1)
            if padding_mask is not None:
                weights = weights.masked_fill(padding_mask[:, None, None, :], 0)
        attended = self.dropout(weights.to(v.dtype)) @ v
        attended = attended.transpose(1, 2).reshape(batch_size, time_steps, d_model)
        output = self.dropout(self.output(attended))
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
        return output


class ConvolutionModule(nn.Module):
    """
    >>> module = ConvolutionModule(d_model=8, kernel_size=3, dropout=0.0)
    >>> module(torch.rand(2, 9, 8)).shape
    torch.Size([2, 9, 8])
    """

    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть положительным и нечётным")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=1)
        )
        self.net = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size,
                padding=kernel_size // 2,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        x = self.pointwise(self.norm(x).transpose(1, 2))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0)
        x = self.net(x).transpose(1, 2)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x


class ConformerBlock(nn.Module):
    """
    >>> block = ConformerBlock(8, 2, 16, 3, 0.0)
    >>> block(torch.rand(2, 9, 8)).shape
    torch.Size([2, 9, 8])
    """

    def __init__(self, d_model, num_heads, ffn_dim, conv_kernel_size, dropout):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention(x, padding_mask)
        x = x + self.conv(x, padding_mask)
        x = self.norm(x + 0.5 * self.ffn2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x
