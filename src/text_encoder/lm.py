import math
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import kenlm


class LM:
    def __init__(
        self, model_path, alpha=0.1, beta=1.0, uppercase=True, unk_score_offset=-10
    ):
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)

        self.model = kenlm.Model(str(model_path))
        self.alpha = alpha
        self.beta = beta
        self.uppercase = uppercase
        self.unk_score_offset = unk_score_offset
        self.unigrams = []
        if str(model_path).endswith(".arpa"):
            with Path(model_path).open() as stream:
                reading = False
                for line in stream:
                    if line.strip() == r"\1-grams:":
                        reading = True
                        continue
                    if reading:
                        if line.startswith("\\"):
                            break
                        fields = line.split()
                        if len(fields) >= 2 and not fields[1].startswith("<"):
                            self.unigrams.append(fields[1])
            self.unigrams.sort()
        self._word_score = lru_cache(maxsize=65536)(self._word_score)
        self._is_word_prefix = lru_cache(maxsize=65536)(self._is_word_prefix)

    def _is_word_prefix(self, word):
        index = bisect_left(self.unigrams, word)
        return index < len(self.unigrams) and self.unigrams[index].startswith(word)

    def _word_score(self, words, is_final):
        text = " ".join(words)
        if self.uppercase:
            text = text.upper()
        unknown = sum(word not in self.model for word in text.split())
        score = (
            self.model.score(text, bos=True, eos=is_final)
            + self.unk_score_offset * unknown
        )
        return score * math.log(10)

    def score(self, text, is_final=False):
        words = tuple(text.split())
        partial = None
        if not is_final and text and not text[-1].isspace():
            partial = words[-1]
            words = words[:-1]
        score = self.alpha * self._word_score(words, is_final) if self.alpha else 0.0
        if partial and self.unigrams:
            word = partial.upper() if self.uppercase else partial
            if not self._is_word_prefix(word):
                score += self.alpha * self.unk_score_offset * math.log(10)
        return score + self.beta * len(words)

    def ctc_decode(self, log_prob, text_encoder=None, beam_width=32, **kwargs):
        from src.text_encoder.beam_search import BeamSearch

        return BeamSearch(
            text_encoder=text_encoder, beam_width=beam_width, lm=self, **kwargs
        ).beam_decode(log_prob)

    beam_decode = ctc_decode
