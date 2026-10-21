import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer


class BPE:
    EMPTY_TOK = ""

    def __init__(self, tokenizer_path, beam_width=1, **kwargs):
        if type(beam_width) is not int or beam_width < 1:
            raise ValueError("Ширини фигня")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model_sha256 = hashlib.sha256(
            Path(tokenizer_path).read_bytes()
        ).hexdigest()
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

        self.alphabet = [
            self.tokenizer.id_to_token(i)
            for i in range(self.tokenizer.get_vocab_size())
        ]
        self.vocab = [self.EMPTY_TOK] + self.alphabet
        self.beam_width = beam_width
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    @property
    def metadata(self):
        return {
            "type": "huggingface_bpe",
            "model_sha256": self.model_sha256,
            "blank_id": 0,
            "tokenizer_id_offset": 1,
            "pieces": len(self) - 1,
        }

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        inds = self.tokenizer.encode(text, add_special_tokens=False).ids
        # Shift tokenizer IDs to reserve zero for the CTC blank.
        return torch.tensor([ind + 1 for ind in inds], dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC, preserving repeated tokens.

        Args:
            inds (list): list of tokens with CTC blank at index zero.
        Returns:
            raw_text (str): text with blank tokens removed.
        """
        inds = [int(ind) - 1 for ind in inds if int(ind) != 0]
        return self.tokenizer.decode(inds).strip()

    def ctc_decode(self, inds) -> str:
        res = []
        prev = 0

        for ind in inds:
            cur = int(ind)
            if cur != 0 and cur != prev:
                res.append(cur)
            prev = cur

        return self.decode(res)

    def beam_decode(self, log_prob: torch.Tensor) -> str:
        if log_prob.ndim != 2 or log_prob.shape[1] != len(self):
            raise ValueError("Beam Search: ошибка размерностей")

        probs = log_prob.detach().to(device="cpu", dtype=torch.float64).exp().tolist()
        beam = {(): (1.0, 0.0)}

        for frame in probs:
            next_beam = defaultdict(lambda: [0.0, 0.0])

            for prefix, (pb, pnb) in beam.items():
                next_beam[prefix][0] += (pb + pnb) * frame[0]

                for token_id, p in enumerate(frame[1:], start=1):
                    new_prefix = prefix + (token_id,)
                    if prefix and prefix[-1] == token_id:
                        next_beam[prefix][1] += pnb * p
                        next_beam[new_prefix][1] += pb * p
                    else:
                        next_beam[new_prefix][1] += (pb + pnb) * p

            beam = dict(
                sorted(next_beam.items(), key=lambda item: sum(item[1]), reverse=True)[
                    : self.beam_width
                ]
            )
            scale = sum(sum(scores) for scores in beam.values())
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Beam Search: вероятности фигня")
            beam = {
                prefix: (pb / scale, pnb / scale) for prefix, (pb, pnb) in beam.items()
            }

        best_prefix = max(beam, key=lambda prefix: sum(beam[prefix]))
        return self.decode(best_prefix)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
