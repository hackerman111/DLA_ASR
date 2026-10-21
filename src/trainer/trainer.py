import math
import random
from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.beam_search import BeamSearch
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        input_lengths = batch["log_probs_length"].detach().cpu()
        target_lengths = batch["text_encoded_length"].detach().cpu()
        targets = batch["text_encoded"].detach().cpu()
        repeated = (targets[:, 1:] == targets[:, :-1]) & (
            torch.arange(targets.shape[1])[1:] < target_lengths[:, None]
        )
        minimum_lengths = target_lengths + repeated.sum(dim=1)
        invalid_lengths = (
            (input_lengths <= 0)
            | (input_lengths > batch["log_probs"].shape[1])
            | (input_lengths < minimum_lengths)
            | (target_lengths < 0)
            | (target_lengths > targets.shape[1])
        )
        if invalid_lengths.any():
            self.logger.error(
                "Invalid CTC lengths; audio_path=%s; log_probs_length=%s; "
                "text_encoded_length=%s; minimum_ctc_length=%s; output_frames=%s",
                batch["audio_path"],
                input_lengths.tolist(),
                target_lengths.tolist(),
                minimum_lengths.tolist(),
                batch["log_probs"].shape[1],
            )
            raise ValueError("Invalid CTC lengths; see batch diagnostics above")

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if not torch.isfinite(batch["loss"]).all():
            lengths = {
                name: batch[name].detach().cpu().tolist()
                for name in (
                    "spectrogram_length",
                    "log_probs_length",
                    "text_encoded_length",
                )
                if name in batch
            }
            self.logger.error(
                "Non-finite loss=%s; audio_path=%s; lengths=%s",
                batch["loss"].detach().cpu().tolist(),
                batch.get("audio_path"),
                lengths,
            )
            raise FloatingPointError(
                "Loss contains NaN or Inf; see batch diagnostics above"
            )

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            batch["grad_norm"] = self._get_grad_norm()
            if not math.isfinite(batch["grad_norm"]):
                self.logger.error(
                    "Non-finite gradient norm=%s; audio_path=%s",
                    batch["grad_norm"],
                    batch["audio_path"],
                )
                raise FloatingPointError("Gradient norm contains NaN or Inf")
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=len(batch["text"]))

        for met in metric_funcs:
            metrics.update(met.name, met(**batch), n=len(batch["text"]))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        count = min(self.cfg_trainer.get("examples_to_log", 3), len(batch["text"]))
        indices = random.sample(range(len(batch["text"])), max(0, count))
        self.log_audio(**batch, sample_rate=self.sample_rates[mode], indices=indices)
        self.log_spectrogram(**batch, indices=indices)
        if mode != "train":
            self.log_predictions(**batch, indices=indices)

    def log_audio(self, audio, audio_length, sample_rate, indices=None, **batch):
        if indices is None:
            indices = [0]
        for index in indices:
            audio_for_log = audio[index, : int(audio_length[index])].detach().cpu()
            self.writer.add_audio(
                f"audio/{index}", audio_for_log.unsqueeze(0), sample_rate=sample_rate
            )

    def log_spectrogram(self, spectrogram, spectrogram_length, indices=None, **batch):
        if indices is None:
            indices = [0]
        for index in indices:
            spectrogram_for_plot = (
                spectrogram[index, :, : int(spectrogram_length[index])].detach().cpu()
            )
            image = plot_spectrogram(spectrogram_for_plot)
            self.writer.add_image(f"spectrogram/{index}", image)

    def log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        examples_to_log=10,
        beam_width=10,
        indices=None,
        **batch,
    ):
        if indices is None:
            count = min(examples_to_log, len(text))
            indices = random.sample(range(len(text)), max(0, count))
        if not indices:
            return
        log_probs = log_probs[indices].detach().cpu()
        lengths = log_probs_length[indices].detach().cpu().tolist()
        text = [text[index] for index in indices]
        audio_path = [audio_path[index] for index in indices]
        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds, lengths)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        beam_search = BeamSearch(
            alphabet=self.text_encoder.alphabet, beam_width=beam_width
        )

        beam_texts = [
            beam_search.beam_decode(scores[: int(length)])
            for scores, length in zip(log_probs, lengths)
        ]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path, beam_texts))

        rows = {}
        for index, (pred, target, raw_pred, path, beam_pred) in zip(indices, tuples):
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(path).name] = {
                "sample index": index,
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
                "beam predictions": beam_pred,
                "beam wer": calc_wer(target, beam_pred) * 100,
                "beam cer": calc_cer(target, beam_pred) * 100,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
