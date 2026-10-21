import hashlib
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer


class BPE:
    EMPTY_TOK = ""

    def __init__(self, tokenizer_path, beam_width=1, lm=None, **kwargs):
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
        self.lm = lm
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
        if isinstance(inds, torch.Tensor) and inds.ndim == 2:
            return self.beam_decode(inds)
        res = []
        prev = 0

        for ind in inds:
            cur = int(ind)
            if cur != 0 and cur != prev:
                res.append(cur)
            prev = cur

        return self.decode(res)

    def beam_decode(self, log_prob: torch.Tensor, **kwargs) -> str:
        from src.text_encoder.beam_search import BeamSearch

        return BeamSearch(
            text_encoder=self, beam_width=self.beam_width, lm=self.lm, **kwargs
        ).beam_decode(log_prob)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
