from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.text_encoder.beam_search import BeamSearch
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """CTC inference with one UTF-8 <utterance_id>.txt per recording."""

    def __init__(self, model, config, device, dataloaders, text_encoder,
                 save_path, metrics=None, batch_transforms=None, skip_model_load=False):
        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.model = model
        self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms or {}
        self.evaluation_dataloaders = dataloaders
        self.save_path = Path(save_path) if save_path is not None else None
        self.metrics = metrics or {"inference": []}
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics["inference"]], writer=None
        ) if self.metrics["inference"] else None
        self.is_train = False
        decoding = self.cfg_trainer.get("decoding", "greedy")
        if decoding not in {"greedy", "beam"}:
            raise ValueError("inferencer.decoding must be greedy or beam")
        self.decoder = BeamSearch(
            text_encoder=text_encoder,
            lm=getattr(text_encoder, "lm", None),
            **dict(self.cfg_trainer.get("beam", {})),
        ) if decoding == "beam" else None
        if not skip_model_load:
            path = self.cfg_trainer.get("from_pretrained")
            if not path:
                raise ValueError("Set inferencer.from_pretrained to a checkpoint path")
            self._from_pretrained(path)

    def _from_pretrained(self, pretrained_path):
        # Legacy training checkpoints contain OmegaConf; load only trusted weights.
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "vocab" in checkpoint and checkpoint["vocab"] != self.text_encoder.vocab:
            raise ValueError("Checkpoint vocabulary does not match the text encoder")
        saved_hash = checkpoint.get("text_encoder", {}).get("model_sha256")
        if saved_hash and saved_hash != getattr(self.text_encoder, "model_sha256", None):
            raise ValueError("Checkpoint tokenizer SHA256 does not match the text encoder")
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        print(f"Loaded checkpoint: {pretrained_path}")

    def run_inference(self):
        return {part: self._inference_part(part, loader)
                for part, loader in self.evaluation_dataloaders.items()}

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        batch.update(self.model(**batch))
        log_probs = batch["log_probs"].detach().cpu()
        lengths = batch["log_probs_length"].detach().cpu().tolist()
        if len(lengths) != len(batch["audio_path"]) or len(log_probs) != len(lengths):
            raise ValueError("CTC batch dimensions do not match audio paths")
        predictions = []
        for scores, length, audio_path in zip(log_probs, lengths, batch["audio_path"]):
            if not 0 < length <= scores.shape[0]:
                raise ValueError(f"Invalid CTC output length for {audio_path}: {length}")
            scores = scores[:int(length)]
            if not torch.isfinite(scores).all():
                raise ValueError(f"Non-finite model output for {audio_path}")
            text = (self.decoder.beam_decode(scores) if self.decoder else
                    self.text_encoder.ctc_decode(scores.argmax(dim=-1)))
            uid = Path(audio_path).stem
            if uid in self._seen_ids:
                raise ValueError(f"Duplicate utterance ID in partition {part}: {uid}")
            self._seen_ids.add(uid)
            predictions.append(text)
            if self.save_path is not None:
                (self.save_path / part / f"{uid}.txt").write_text(text + "\n", encoding="utf-8")
        batch["predictions"] = predictions
        if metrics is not None:
            for metric in self.metrics["inference"]:
                value = metric(**batch)
                metrics.update(metric.name, value, n=getattr(metric, "weight", len(predictions)))
        return batch

    def _inference_part(self, part, dataloader):
        self.model.eval()
        self._seen_ids = set()
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()
            for metric in self.metrics["inference"]:
                if hasattr(metric, "reset"):
                    metric.reset()
        if self.save_path is not None:
            directory = self.save_path / part
            directory.mkdir(exist_ok=True, parents=True)
            if any(directory.glob("*.txt")):
                raise FileExistsError(f"Predictions already exist in {directory}; choose a new save_path")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=part)):
                self.process_batch(batch_idx, batch, self.evaluation_metrics, part)
        logs = self.evaluation_metrics.result() if self.evaluation_metrics is not None else {}
        logs["utterances"] = len(self._seen_ids)
        return logs
