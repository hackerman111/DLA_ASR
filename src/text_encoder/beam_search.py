import math
import re
from collections import defaultdict
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_width=1, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.beam_width = beam_width
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(
                [self.char2ind[char] for char in text], dtype=torch.long
            ).unsqueeze(0)
        except KeyError:
            unknown_chars = {char for char in text if char not in self.char2ind}
            raise Exception(  # noqa: TRY002
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def beam_decode(self, log_prob: torch.Tensor) -> str:
        probs = log_prob.detach().to(device="cpu", dtype=torch.float64).exp().tolist()
        beam = {"": (1.0, 0.0)}

        for t in range(len(probs)):
            next_beam = defaultdict(lambda: [0.0, 0.0])

            for prefix, (pb, pnb) in beam.items():
                p = probs[t][0]
                next_beam[prefix][0] += (pb + pnb) * p

                for token_id, c in enumerate(self.vocab[1:], start=1):
                    p = probs[t][token_id]

                    if prefix.endswith(c):
                        next_beam[prefix][1] += pnb * p

                        new_prefix = prefix + c
                        next_beam[new_prefix][1] += pb * p

                    else:
                        new_prefix = prefix + c
                        next_beam[new_prefix][1] += (pb + pnb) * p

            beam = dict(
                sorted(next_beam.items(), key=lambda item: sum(item[1]), reverse=True)[
                    : self.beam_width
                ]
            )
            scale = sum(sum(scores) for scores in beam.values())
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Токен фигня")
            beam = {
                prefix: (pb / scale, pnb / scale) for prefix, (pb, pnb) in beam.items()
            }

        return max(beam, key=lambda prefix: sum(beam[prefix])).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
