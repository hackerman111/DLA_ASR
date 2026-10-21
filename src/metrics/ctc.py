import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import levi_dist


class CorpusCTCMetric(BaseMetric):
    def __init__(self, text_encoder, unit="word", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if unit not in ("word", "char"):
            raise ValueError("CTC: слово фигня")
        self.text_encoder = text_encoder
        self.unit = unit
        self.weight = 0

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        predictions = torch.argmax(log_probs, dim=-1).detach().cpu()
        errors = self.weight = 0
        for tokens, length, reference in zip(predictions, log_probs_length, text):
            reference = self.text_encoder.normalize_text(reference)
            prediction = self.text_encoder.ctc_decode(tokens[: int(length)])
            if self.unit == "word":
                reference, prediction = reference.split(), prediction.split()
            errors += levi_dist(reference, prediction)
            self.weight += len(reference)
        if self.weight == 0:
            raise ValueError("Корпус пустой и фигня")
        return errors / self.weight
