import math
import re
from collections import defaultdict
from functools import lru_cache

import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class BeamSearch:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        beam_width=1,
        text_encoder=None,
        lm=None,
        token_top_k=None,
        token_min_logp=-math.inf,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if type(beam_width) is not int or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if token_top_k is not None and (
            type(token_top_k) is not int or token_top_k < 1
        ):
            raise ValueError("token_top_k must be a positive integer")
        self.text_encoder = text_encoder or CTCTextEncoder(alphabet=alphabet)
        self.lm = lm
        self.token_top_k = token_top_k
        self.token_min_logp = token_min_logp
        self.alphabet = self.text_encoder.alphabet
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
        return self.text_encoder.encode(text)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return self.text_encoder.decode(inds)

    def beam_decode(self, log_prob: torch.Tensor) -> str:
        if log_prob.ndim != 2 or log_prob.shape[1] != len(self):
            raise ValueError(
                "Expected frame log probabilities with shape [T, vocabulary]"
            )
        scores = log_prob.detach().to(device="cpu", dtype=torch.float64)
        if scores.isnan().any() or scores.isposinf().any():
            raise ValueError("Frame log probabilities contain NaN or +Inf")
        probs = scores.exp().tolist()
        if self.token_top_k is None:
            candidates = [range(1, len(self)) for _ in probs]
        else:
            candidates = scores.topk(
                min(self.token_top_k, len(self)), dim=-1
            ).indices.tolist()
        cutoffs = (scores.max(dim=-1).values + self.token_min_logp).tolist()
        beam = {(): (1.0, 0.0)}

        @lru_cache(maxsize=4096)
        def prefix_text(prefix):
            if hasattr(self.text_encoder, "tokenizer"):
                return self.text_encoder.tokenizer.decode([i - 1 for i in prefix])
            return "".join(self.ind2char[i] for i in prefix)

        def rank(item, is_final=False):
            prefix, values = item
            acoustic = math.log(sum(values))
            return acoustic + (
                self.lm.score(prefix_text(prefix), is_final) if self.lm else 0
            )

        for frame, token_ids, cutoff in zip(probs, candidates, cutoffs):
            next_beam = defaultdict(lambda: [0.0, 0.0])
            tokens = [
                (i, frame[i])
                for i in token_ids
                if i != 0 and frame[i] > 0 and math.log(frame[i]) >= cutoff
            ]
            for prefix, (pb, pnb) in beam.items():
                if frame[0] > 0:
                    next_beam[prefix][0] += (pb + pnb) * frame[0]
                for token_id, p in tokens:
                    new_prefix = prefix + (token_id,)
                    if prefix and prefix[-1] == token_id:
                        next_beam[prefix][1] += pnb * p
                        next_beam[new_prefix][1] += pb * p
                    else:
                        next_beam[new_prefix][1] += (pb + pnb) * p
            live = [
                (prefix, values)
                for prefix, values in next_beam.items()
                if sum(values) > 0
            ]
            beam = dict(sorted(live, key=rank, reverse=True)[: self.beam_width])
            scale = sum(sum(values) for values in beam.values())
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("No finite CTC path survives")
            beam = {
                prefix: (pb / scale, pnb / scale) for prefix, (pb, pnb) in beam.items()
            }

        best_prefix = max(beam.items(), key=lambda item: rank(item, is_final=True))[0]
        return self.decode(best_prefix)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
